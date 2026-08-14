"""Packaging contract for a PairLogit ranker. Run before any rank GPU job.

A ranker is the one member whose raw output is **not** a probability. The
trainer's own test path knows that -- it reads `model._rank_calib` and pushes
the score through a sigmoid -- but that attribute lives only in the fitting
process. Verified here: a CatBoost object does **not** carry custom attributes
through joblib, so a reloaded ranker has no calibration, and the generic
fallbacks in `train_gbdt2` would then reach
`np.clip(model.predict(Xt), 0, 1)` and ship a clipped pairwise score as
P(success).

So the calibration travels in the pack dict and `rank_probability_from_pack` is
the only sanctioned inference path. These are the eight checks the overnight
plan asks for, plus the negative one that explains why the key exists.

  python tests/test_rank_contract.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train_gbdt2 as T                                          # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def build(n=6000, seed=0):
    """A small ranker with one genuine categorical, fitted on CPU."""
    from catboost import CatBoostRanker, Pool

    rng = np.random.default_rng(seed)
    hand = rng.choice(["L", "R"], size=n)
    X = pd.DataFrame({
        "f0": rng.normal(size=n), "f1": rng.normal(size=n),
        "f2": rng.integers(0, 4, n).astype(float),
        "pitcher_hand": hand,
    })
    lin = 0.9 * X.f0 - 0.6 * X.f1 + 0.4 * (hand == "L")
    y = (rng.random(n) < 1 / (1 + np.exp(-lin))).astype(float)
    feats = ["f0", "f1", "f2", "pitcher_hand"]
    cats = ["pitcher_hand"]
    g = np.arange(n) // 16                     # group16, the only allowed size
    m = CatBoostRanker(iterations=60, depth=4, learning_rate=0.1,
                       loss_function="PairLogitPairwise", verbose=0,
                       task_type="CPU", random_seed=seed)
    m.fit(Pool(X[feats], y, cat_features=cats, group_id=g))
    raw = m.predict(X[feats])
    calib = T._fit_rank_sigmoid(raw, y)
    m._rank_calib = calib
    pack = {"model": m, "features": feats, "cat_cols": cats,
            "rank_calib": {"a": calib[0], "b": calib[1],
                           "mu": calib[2], "sd": calib[3]}}
    return pack, X, y, calib


def main():
    print("rank packaging contract")
    pack, X, y, calib = build()
    p_live = T._rank_probability(pack["model"].predict(X[pack["features"]]),
                                 calib)

    path = os.path.join(tempfile.gettempdir(), "rank_contract.pkl")
    joblib.dump(pack, path, compress=3)
    got = joblib.load(path)

    # 0. the negative result the pack key exists for
    check("a CatBoost attribute does NOT survive joblib (why rank_calib is "
          "in the pack)", getattr(got["model"], "_rank_calib", None) is None)

    # 1. saved pkl replay
    p_pack = T.rank_probability_from_pack(got, X)
    check("1 saved pkl replays the live probabilities exactly",
          np.array_equal(p_live, p_pack),
          f"max|diff| {np.max(np.abs(p_live - p_pack)):.3g}")

    # 2. same frame, twice
    check("2 same frame predicts identically on repeat",
          np.array_equal(p_pack, T.rank_probability_from_pack(got, X)))

    # 3. subset
    idx = np.arange(0, len(X), 7)
    sub = T.rank_probability_from_pack(got, X.iloc[idx].reset_index(drop=True))
    check("3 a row subset scores identically", np.array_equal(sub, p_pack[idx]),
          f"{len(idx):,} rows")

    # 4. one row alone -- the criterion test.csv is judged on
    one = T.rank_probability_from_pack(got, X.iloc[[0]].reset_index(drop=True))
    check("4 a single row scores identically",
          float(one[0]) == float(p_pack[0]))

    # 5. calibration round trip
    rc = got["rank_calib"]
    check("5 calibration survives serialisation exactly",
          (rc["a"], rc["b"], rc["mu"], rc["sd"]) == tuple(calib),
          f"a={rc['a']:.10f} b={rc['b']:.10f} mu={rc['mu']:.10f} "
          f"sd={rc['sd']:.10f}")

    # 6. schema
    check("6 feature names and cat cols survive exactly",
          list(got["features"]) == list(pack["features"])
          and list(got["cat_cols"]) == list(pack["cat_cols"]))

    # 7. bias -- a calibrated ranker must sit near the base rate
    bias = float(p_pack.mean() - y.mean())
    check("7 mean prediction is near the target mean", abs(bias) < 0.03,
          f"bias {bias:+.5f} (pred {p_pack.mean():.5f} vs {y.mean():.5f})")

    # 8. finite and in range
    check("8 probabilities are finite and inside [0, 1]",
          bool(np.isfinite(p_pack).all() and (p_pack >= 0).all()
               and (p_pack <= 1).all()),
          f"[{p_pack.min():.6f}, {p_pack.max():.6f}]")

    # 9. an uncalibrated pack must refuse, not degrade
    bare = dict(got)
    bare["rank_calib"] = None
    try:
        T.rank_probability_from_pack(bare, X)
        check("9 a pack without calibration refuses", False, "it returned")
    except ValueError:
        check("9 a pack without calibration refuses", True)

    # 10. clipping a raw score really is a different answer -- the failure this
    #     whole contract prevents, quantified rather than asserted
    rawp = np.clip(got["model"].predict(X[got["features"]]), 0, 1)
    check("10 clipped raw score is NOT the calibrated probability",
          not np.allclose(rawp, p_pack, atol=1e-6),
          f"mean {rawp.mean():.4f} vs {p_pack.mean():.4f}, "
          f"max|diff| {np.max(np.abs(rawp - p_pack)):.4f}")

    os.remove(path)
    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

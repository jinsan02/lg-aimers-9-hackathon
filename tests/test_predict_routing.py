"""`fpipe.predict` must know what kind of pack it is holding.

This is the test whose absence let two defects live. `fpipe.predict` is the
submission path — the shipped `script.py` calls it — and it used to end in
`return proba[:, 1]` for anything it did not recognise, while recognising only
xgboost, a baseline column and `fm_success`. So:

  * a **multilabel** pack returned column 1, `P(middle)`, as `P(success)`.
    Measured on the real `cat_ML2_s3.pkl`: 0.130428 against a true 0.493087,
    with no exception.
  * a **ranker** pack has no `predict_proba`, so it raised `AttributeError`
    here — loud, but a runtime error in `script.py` costs a submission.

`tests/test_rank_contract.py` passed throughout, because it exercises
`rank_probability_from_pack`, which the submission never calls. A contract test
has to test the path that ships.

What is asserted:

  1. the champion's own packs are **bit-identical** to the pre-fix behaviour;
  2. a multilabel pack returns head 0, not head 1;
  3. a ranker pack returns the calibrated probability, not a crash;
  4. a ranker pack with no calibration **raises** rather than guessing;
  5. a multiclass pack that declares neither kind **raises** rather than
     reading column 1 as if it were binary.

Run: python tests/test_predict_routing.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fpipe                                                     # noqa: E402

FAIL = []
N = 400


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def frames():
    hdr = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                           encoding="utf-8-sig").columns)
    raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      encoding="utf-8-sig",
                      usecols=hdr + ["control_success"])
    test = raw[raw.season == 2024].head(N)[hdr].reset_index(drop=True)
    fit = raw[raw.season == 2023].head(4000).reset_index(drop=True)
    return test, fit


def main():
    print("fpipe.predict routing")
    test, fit = frames()

    base = joblib.load(os.path.join(ROOT, "model", "cat_B1S_base_s3.pkl"))
    cell = joblib.load(os.path.join(ROOT, "model", "cat_B1S_cell_s3.pkl"))

    # 1. the champion, unchanged. Reference values come from the models
    #    themselves rather than a stored constant, so this stays true if the
    #    frame changes but breaks the moment routing changes.
    src = fpipe.transform(test.copy(), base["fpipe"])
    X = fpipe.design(src, base)
    ref_base = base["model"].predict_proba(X)[:, 1]
    got_base = fpipe.predict(dict(base), test.copy())
    check("1a binary pack == predict_proba[:, 1] exactly",
          np.array_equal(ref_base, got_base), f"mean {got_base.mean():.6f}")

    src_c = fpipe.transform(test.copy(), cell["fpipe"])
    Xc = fpipe.design(src_c, cell)
    ref_cell = cell["model"].predict_proba(Xc)[:, cell["fm_success"]].sum(axis=1)
    got_cell = fpipe.predict(dict(cell), test.copy())
    check("1b cell pack == sum of success cells exactly",
          np.array_equal(ref_cell, got_cell),
          f"cells {cell['fm_success']}, mean {got_cell.mean():.6f}")

    # 2. multilabel -> head 0
    ml_path = os.path.join(ROOT, "model", "cat_ML2_s3.pkl")
    if os.path.exists(ml_path):
        ml = joblib.load(ml_path)
        pr = ml["model"].predict_proba(
            fpipe.design(fpipe.transform(test.copy(), ml["fpipe"]), ml))
        got = fpipe.predict(dict(ml), test.copy())
        check("2a multilabel pack returns head 0 (P(success))",
              np.array_equal(got, pr[:, 0]),
              f"head0 {pr[:, 0].mean():.6f} vs head1 {pr[:, 1].mean():.6f}")
        check("2b and NOT head 1 (P(middle)) -- the defect this replaces",
              not np.allclose(got, pr[:, 1]),
              f"difference {abs(pr[:, 0].mean() - pr[:, 1].mean()):.6f}")
        bare = dict(ml)
        bare["fm_multilabel"] = False
        try:
            fpipe.predict(bare, test.copy())
            check("2c a multiclass pack declaring neither kind raises", False,
                  "it returned a number")
        except ValueError:
            check("2c a multiclass pack declaring neither kind raises", True)
    else:
        print("  SKIP multilabel arm -- model/cat_ML2_s3.pkl not present")

    # 3/4. ranker
    from catboost import CatBoostRanker, Pool
    fr = fpipe.transform(fit, base["fpipe"])
    Xr = fr[base["features"]].copy()
    for c in base["cat_cols"]:
        Xr[c] = Xr[c].astype(str)
    m = CatBoostRanker(iterations=30, depth=4, loss_function="PairLogitPairwise",
                       task_type="CPU", verbose=0, random_seed=1)
    m.fit(Pool(Xr, fr["control_success"].to_numpy(float),
               cat_features=base["cat_cols"],
               group_id=np.arange(len(Xr)) // 16))
    rank = dict(base)
    rank["model"] = m
    rank["fm_success"] = None
    rank["rank_calib"] = {"a": 0.05, "b": 1.1, "mu": 0.0, "sd": 1.0}
    rank["rank_ntree_end"] = 0
    got_r = fpipe.predict(dict(rank), test.copy())
    raw_r = m.predict(fpipe.design(fpipe.transform(test.copy(), rank["fpipe"]),
                                   rank))
    want = 1.0 / (1.0 + np.exp(-np.clip(0.05 + 1.1 * raw_r, -30, 30)))
    check("3a ranker pack returns the calibrated probability",
          np.allclose(got_r, want), f"mean {got_r.mean():.6f}")
    check("3b and is finite inside [0, 1]",
          bool(np.isfinite(got_r).all() and (got_r >= 0).all()
               and (got_r <= 1).all()))
    check("3c and is NOT the clipped raw score",
          not np.allclose(got_r, np.clip(raw_r, 0, 1)),
          f"clipped mean {np.clip(raw_r, 0, 1).mean():.6f}")

    nocal = dict(rank)
    nocal["rank_calib"] = None
    try:
        fpipe.predict(nocal, test.copy())
        check("4 a ranker with no calibration raises", False, "it returned")
    except ValueError:
        check("4 a ranker with no calibration raises", True)

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

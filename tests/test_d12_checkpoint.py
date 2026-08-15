"""D12 picks the checkpoint by the fixed-core Brier, and only by that.

CPU sanity before any GPU is spent, per AGENTS.md. A stub classifier stands in
for CatBoost so the selection rule can be checked against a curve whose answer
is known by construction -- with a real model the "right" checkpoint is exactly
what is in question.

What is asserted:

  1. the base predictions are re-attached by row_id, not by position;
  2. a row_id set that does not cover the validation season is refused rather
     than silently producing NaNs;
  3. the maximum of the fixed-core curve wins;
  4. the pre-registered 0.05 tie-break takes the SMALLER iteration, which is the
     conservative direction;
  5. the grid spacing is what was pre-registered, and the last point is the
     model's real tree count rather than a rounded-up multiple;
  6. selection reads the validation season only.

Run: python tests/test_d12_checkpoint.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import train_gbdt2 as T                                          # noqa: E402

FAIL = []
N, NCELL = 600, 4
SUCC = [0, 1]


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


class StubFM:
    @staticmethod
    def success_prob(proba, succ):
        return np.asarray(proba)[:, list(succ)].sum(axis=1)


class StubClf:
    """staged_predict_proba with a known *skill* per checkpoint.

    `curve` maps iteration -> a, and the success probability is
    `0.5 + a*(y - 0.5)`. Skill rises monotonically with a, so the checkpoint
    with the largest a has the best fixed-core Brier by construction.

    An earlier version of this stub emitted a constant probability per
    checkpoint and expected the largest one to win. That was wrong: against a
    balanced target a constant 0.9 scores *worse* than a constant 0.5, so the
    test was asserting its own error rather than the selector's.
    """

    def __init__(self, curve, trees, y_val):
        self.curve = curve
        self.tree_count_ = trees
        self.y = np.asarray(y_val, float)
        self.seen = []

    def staged_predict_proba(self, X, ntree_start, ntree_end, eval_period):
        self.seen.append(len(X))
        i = 0
        while True:
            i += 1
            it = min(i * eval_period, ntree_end)
            ps = np.clip(0.5 + self.curve[it] * (self.y - 0.5), 0.001, 0.999)
            pr = np.zeros((len(X), NCELL))
            pr[:, 0] = ps                      # success cells 0 and 1
            pr[:, 2] = 1.0 - ps
            yield pr
            if it >= ntree_end:
                break


def frame():
    rng = np.random.default_rng(0)
    rid = np.array([f"TRAIN_{i:07d}" for i in range(N)])
    y = (rng.random(N) < 0.5).astype(float)
    df = pd.DataFrame({"row_id": rid, T.TARGET: y, "f": rng.normal(size=N)})
    is_val = pd.Series([False] * (N // 2) + [True] * (N // 2))
    return df, is_val


def run(curve, trees, step, base_pred=None, base_rid=None, tag="t"):
    df, is_val = frame()
    vid = df.loc[is_val, "row_id"].to_numpy()
    y = df.loc[is_val, T.TARGET].to_numpy(float)
    if base_pred is None:
        base_pred, base_rid = np.full(len(vid), y.mean()), vid
    p = os.path.join(ROOT, "out", f"_d12test_{tag}.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p, pred=base_pred, row_id=base_rid)
    args = SimpleNamespace(d12_base_preds=p, d12_step=step, tag=f"_d12test_{tag}")
    clf = StubClf(curve, trees, y)
    try:
        got = T._d12_checkpoint(args, clf, df, ["f"], is_val, SUCC, StubFM(),
                                mc_iter=trees)
    finally:
        for f in (p, os.path.join(ROOT, "out", f"d12_curve__d12test_{tag}.npz")):
            if os.path.exists(f):
                os.remove(f)
    return got, clf, len(vid)


def main():
    print("selection rule")
    # Skill peaks at iteration 50 and falls away either side -- the shape of a
    # real overfitting curve.
    curve = {25: 0.30, 50: 0.80, 75: 0.55, 100: 0.20}
    got, clf, nval = run(curve, 100, 25, tag="peak")
    check("1 the fixed-core maximum wins", got == 50, f"chose {got}")
    check("2 selection reads the validation season only",
          clf.seen and all(n == nval for n in clf.seen),
          f"{clf.seen} rows vs {nval} validation rows")

    # Identical skill at 25 and 50: inside the 0.05 band, so the smaller wins.
    flat = {25: 0.80, 50: 0.80, 75: 0.30, 100: 0.20}
    got, _, _ = run(flat, 100, 25, tag="tie")
    check("3 a tie inside 0.05 BSS takes the SMALLER iteration", got == 25,
          f"chose {got} from a flat top at 25 and 50")

    print("\ngrid")
    got, clf, _ = run({20: 0.3, 40: 0.9, 55: 0.3}, 55, 20, tag="grid")
    check("4 the last grid point is the real tree count, not a rounded multiple",
          got == 40, "grid 20/40/55 over 55 trees")

    print("\nbase predictions are joined by row_id")
    df, is_val = frame()
    vid = df.loc[is_val, "row_id"].to_numpy()
    y = df.loc[is_val, T.TARGET].to_numpy(float)
    # Same values, shuffled order: a positional join would corrupt the core,
    # a row_id join is invariant.
    perm = np.random.default_rng(1).permutation(len(vid))
    got_a, _, _ = run(curve, 100, 25, np.full(len(vid), y.mean()), vid, "ord")
    got_b, _, _ = run(curve, 100, 25, np.full(len(vid), y.mean())[perm],
                      vid[perm], "shuf")
    check("5 shuffling the base npz changes nothing", got_a == got_b,
          f"{got_a} vs {got_b}")

    try:
        run(curve, 100, 25, np.zeros(3),
            np.array(["NOPE_1", "NOPE_2", "NOPE_3"]), "bad")
        check("6 a base npz that does not cover the season is refused", False,
              "it returned")
    except SystemExit as e:
        check("6 a base npz that does not cover the season is refused", True,
              str(e)[:60])

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

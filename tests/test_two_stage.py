"""Audit contract 1/2 -- the selection and deployment artifacts are separate.

The selection model's artifact may only see rows before validation. The refit
model's artifact is allowed the whole final-train partition. Before this split
one `is_fit` artifact served both, which pinned the TE shrink prior at
<= val_season-1 for the deployment model too. Success rates fall every season
(.5495 in 2019 -> .4861 in 2024), so that stale prior cost -2.41 on unseen 2024
while gaining +3.56 on validation -- the two-surface sign flip that made a
half-applied P1 look like a regression.

Run: python tests/test_two_stage.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import skill as sk                                              # noqa: E402
import target_enc as te                                         # noqa: E402

VAL, TEST = 2023, 2024
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    cols = ["row_id", "season", "pitcher_id", "batter_id", "balls_before",
            "strikes_before", "control_success"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    d = d[d.season <= TEST].reset_index(drop=True)

    sel = (d.season < VAL).to_numpy()                    # selection fit  <= 2022
    dep = (d.season <= VAL).to_numpy()                   # deployment fit <= 2023

    # --- the two stages must actually differ ------------------------------
    p_sel = float(d.control_success[sel].mean())
    p_dep = float(d.control_success[dep].mean())
    p_test = float(d.control_success[d.season == TEST].mean())
    check("the two stage priors are different numbers",
          abs(p_sel - p_dep) > 1e-6, f"sel {p_sel:.6f} vs dep {p_dep:.6f}")
    # The whole point: the deployment prior is the one closer to the season it
    # will actually score. If this ever reverses, the drift argument is void.
    check("the deployment prior is closer to the unseen season",
          abs(p_dep - p_test) < abs(p_sel - p_test),
          f"|dep-2024| {abs(p_dep - p_test):.6f} < |sel-2024| {abs(p_sel - p_test):.6f}")

    keys = ["pitcher_id"]
    t_sel = te.build_te(d, keys, k=50, fit_mask=sel)
    t_dep = te.build_te(d, keys, k=50, fit_mask=dep)
    rate = [c for c in t_sel.columns if c.endswith("_rate")][0]
    n_col = [c for c in t_sel.columns if c.endswith("_n")][0]
    moved = float(np.nanmax(np.abs(t_sel[rate].to_numpy() - t_dep[rate].to_numpy())))
    check("the two artifacts produce different TE rates", moved > 1e-9,
          f"max |delta| {moved:.3e}")
    check("only the prior differs -- the expanding body is identical",
          np.allclose(t_sel[n_col].to_numpy(), t_dep[n_col].to_numpy(),
                      equal_nan=True))

    # --- the deployment artifact must not reach the test season -----------
    # It may see validation. It may never see the season it will score.
    t_leak = te.build_te(d, keys, k=50, fit_mask=np.ones(len(d), bool))
    check("a whole-frame prior really would differ from the deployment one",
          abs(float(d.control_success.mean()) - p_dep) > 1e-6,
          f"whole {d.control_success.mean():.6f} vs dep {p_dep:.6f}")
    check("whole-frame and deployment TE rates differ",
          float(np.nanmax(np.abs(t_leak[rate].to_numpy()
                                 - t_dep[rate].to_numpy()))) > 1e-9)

    # --- skill: explicit row_id ordering ----------------------------------
    # `_future` is a reverse cumulative mean inside a pitcher-season, so the
    # target depends on row order. It used to inherit whatever order the caller
    # passed; a shuffled frame silently produced different labels.
    sub = d[d.season <= 2020].copy()
    for c in sk.FEAT:
        if c not in sub.columns:
            sub[c] = 0.5
    rng = np.random.default_rng(0)
    shuffled = sub.iloc[rng.permutation(len(sub))].reset_index(drop=True)
    a = sk.build(sub)
    b = sk.build(shuffled)
    same = all(np.allclose(a["coef"][s], b["coef"][s])
               for s in a["coef"] if not isinstance(a["coef"][s], str))
    check("shuffling the input frame does not change skill coefficients", same)

    fn_a, fr_a = sk._future(sub)
    fn_b, fr_b = sk._future(shuffled)
    ja = pd.DataFrame({"row_id": sub.row_id, "fr": fr_a.reindex(sub.index)})
    jb = pd.DataFrame({"row_id": shuffled.row_id, "fr": fr_b.reindex(shuffled.index)})
    m = ja.merge(jb, on="row_id", suffixes=("_a", "_b"))
    check("per-row 'rest of season' target is order independent",
          bool(np.allclose(m.fr_a.to_numpy(), m.fr_b.to_numpy(), equal_nan=True)),
          f"{len(m):,} rows joined on row_id")

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

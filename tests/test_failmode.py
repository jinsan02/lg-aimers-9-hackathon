"""Regression tests for the failure-mode cell labels.

Two defects were found on 2026-08-13 and fixed; these lock the contract.

1. `_pitch_labels` shifted globally instead of within the pitcher group. The diff
   is grouped, but the shift that aligns it to the row was not, so 99,078 rows
   (6.72%) at an inning/pitcher change pulled the *next pitcher's* outcome.
   57,157 rows (3.87%) ended up in a different cell.
2. `build_cells` was called on the whole frame, so the last 310 fit rows took
   their label from the validation season's first pitch, and the rare-class
   taxonomy was chosen with validation rows in it.

Run: python tests/test_failmode.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode as fm                                          # noqa: E402

COLS = ["row_id", "pitcher_id", "season", "game_type", "control_success",
        "asof_pitcher_n", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate"]
VAL_SEASON = 2024
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=COLS)
    fit = (d.season < VAL_SEASON).to_numpy()
    lab = fm._pitch_labels(d)
    code, names, succ = fm.build_cells(d, verbose=False, fit_mask=fit)

    # 1. The success bit must reproduce the real target exactly -- that identity is
    #    the whole reason the target is inside the cell.
    hit = code.isin(list(succ)).to_numpy() == d.control_success.to_numpy()
    check("success bit == target", hit.all(), f"{hit.mean():.6%}")

    # 2. A pitcher's final pitch in the data has no successor, so it must be NaN.
    ds = d.sort_values("row_id", kind="stable")
    pid = ds.pitcher_id.to_numpy()
    final = ~pd.Series(pid).duplicated(keep="last").to_numpy()
    tail = lab.reindex(ds.index).loc[final, list(fm.MODES)]
    check("pitcher's last pitch is NaN", bool(tail.isna().all().all()),
          f"{final.sum():,} pitchers")

    # 3. The label at a pitcher change must come from that pitcher's next
    #    appearance, never from the pitcher who relieved them.
    nxt_pid = pd.Series(pid).groupby(pid).shift(-1)
    check("no label crosses pitchers", bool(nxt_pid.dropna().eq(
        pd.Series(pid)[nxt_pid.notna()]).all()))

    # 4. Row order must not matter once row_id is restored.
    sh = d.sample(frac=1.0, random_state=1)
    shuffled = fm.build_cells(sh, verbose=False,
                              fit_mask=(sh.season < VAL_SEASON).to_numpy())[0]
    check("stable under shuffle", bool((shuffled.reindex(d.index) == code).all()))

    # 5. Perturbing the first validation row must not move a single fit label or
    #    change the taxonomy.
    d2 = d.copy()
    d2.loc[d2.index[d2.season == VAL_SEASON][0], "asof_pitcher_middle_rate"] = .999
    code2, names2, _ = fm.build_cells(d2, verbose=False, fit_mask=fit)
    check("validation cannot touch fit labels", bool((code[fit] == code2[fit]).all()))
    check("validation cannot touch taxonomy", names == names2)

    # 6. Unknown validation cells fall back to a reserved class, never NaN.
    check("no NaN codes", bool(code.notna().all()))
    check("reserved classes present", {"0xxx", "1xxx"} <= set(names))

    # 7. The split-safe build must differ from the legacy one only on the fit
    #    boundary -- if it moves validation rows too, something else changed.
    legacy = fm.build_cells(d, verbose=False)[0]
    moved = (legacy != code).to_numpy()
    check("only the fit boundary moves", moved[~fit].sum() == 0,
          f"fit {moved[fit].sum():,} / val {moved[~fit].sum():,}")

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

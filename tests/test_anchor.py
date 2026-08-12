"""Audit 4.1 -- the season anchor must count the season's final pitch.

`asof_*` columns are pre-pitch, so a season's last row describes the state
*before* its own final pitch and the anchor built from it is one short.
Measured on the official train (2026-08-13): `next_first_n ==
previous_last_asof_n + 1` for 1,468/1,468 pitcher and 1,563/1,563 batter
transitions.

The audit is explicit that testing the raw-data property is worthless -- it
holds under the buggy code too. These tests check the **generated** anchor.

Run: python tests/test_anchor.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import season_std as ss                                         # noqa: E402

FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def first_n(d, idc, ncol):
    f = d.sort_values(ncol).groupby([idc, "season"], sort=False).head(1)
    return f[[idc, "season", ncol]].rename(columns={ncol: "first_n"})


def main() -> int:
    idc, ncol = "pitcher_id", "asof_pitcher_n"
    rate = "asof_pitcher_success_rate"
    cols = [idc, "batter_id", "season", ncol, "asof_batter_n", rate,
            "asof_pitcher_middle_rate", "control_success"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)

    legacy = ss.build_anchors(d)["pitcher"]
    fixed = ss.build_anchors(d, last_pitch=True)["pitcher"]

    # The anchor for season S must equal the first asof_n actually observed in
    # season S -- that is what "state at the start of S" means.
    obs = first_n(d, idc, ncol)
    for name, tab in (("legacy", legacy), ("fixed", fixed)):
        col = "n0_" + rate if "n0_" + rate in tab.columns else "n0"
        m = obs.merge(tab[[idc, "season", col]], on=[idc, "season"], how="inner")
        m = m.dropna(subset=[col])
        agree = float((m[col] == m["first_n"]).mean())
        if name == "legacy":
            check("legacy anchor.n0 does NOT match the season's first asof_n",
                  agree < 0.5, f"{agree * 100:.2f}% agree ({len(m):,} rows)")
        else:
            check("fixed anchor.n0 == the season's first asof_n",
                  agree > 0.999, f"{agree * 100:.2f}% agree ({len(m):,} rows)")

    check("the fix moves n0 by exactly one pitch, never more",
          bool(set(np.unique((fixed["n0_" + rate] - legacy["n0"]).dropna()))
               <= {0.0, 1.0}),
          f"deltas {sorted(set(np.unique((fixed['n0_' + rate] - legacy['n0']).dropna())))}")

    # S0 must gain the final pitch's own outcome, so it moves by 0 or 1 too.
    ds0 = (fixed["S0_" + rate] - legacy["S0_" + rate]).dropna()
    check("S0 gains at most the one pitch it was missing",
          bool(ds0.min() >= -1e-9 and ds0.max() <= 1.0 + 1e-9),
          f"range {ds0.min():+.3f}..{ds0.max():+.3f}")

    # Per-rate denominators: only rates whose last-pitch outcome is exactly
    # known may be corrected. A corrected count under an uncorrected numerator
    # is a worse error than the off-by-one.
    mid = "asof_pitcher_middle_rate"
    check("only the success rate gets a corrected denominator",
          bool((fixed[f"n0_{mid}"].dropna()
                == legacy["n0"].reindex(fixed.index).dropna()).all()),
          "middle keeps the short anchor")
    check("S0 for middle is untouched",
          bool(np.allclose(fixed["S0_" + mid].to_numpy(),
                           legacy["S0_" + mid].to_numpy(), equal_nan=True)))

    # fit_mask: an anchor derived from a row outside the fit partition must not
    # be corrected, because that correction reads the row's target.
    mask = (d.season <= 2022).to_numpy()
    masked = ss.build_anchors(d, last_pitch=True, fit_mask=mask)["pitcher"]
    # Only rows whose *source* season is outside the mask. A player who did not
    # pitch in 2023 forward-fills a corrected 2022 anchor into 2024, and that is
    # correct -- its source row is inside the fit partition.
    active23 = set(d.loc[d.season == 2023, idc].unique())
    late = masked[(masked.season == 2024) & masked[idc].isin(active23)].dropna(
        subset=["n0"])
    check("anchors sourced outside the fit partition stay uncorrected",
          bool((late["n0_" + rate] == late["n0"]).all()),
          f"{len(late):,} pitchers active in 2023")
    early = masked[(masked.season > 2019) & (masked.season <= 2022)]
    moved = (early["n0_" + rate] - early["n0"]).dropna()
    check("anchors inside the fit partition are still corrected",
          bool(len(moved) and moved.max() == 1.0),
          f"{int((moved == 1).sum()):,} of {len(moved):,} moved")

    print(f"\n{len(FAILED)} failed" if FAILED else "\nall passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

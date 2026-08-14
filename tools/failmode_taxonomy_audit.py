"""corrected12 vs legacy14 -- what actually differs, cell by cell.

The +3.97 that reverting the label fix is worth (SETTLED
`failmode-corrected-labels`) changed at least two things at once:

  * the cell *taxonomy* went 12 -> 14 classes
  * about 3.87% of cell assignments moved, because the legacy `shift(-1)` is
    global instead of per-pitcher and pulls the next pitcher's first diff across
    the boundary

"auxiliary label noise regularises" is one reading. "a 14-way auxiliary task is
a better task than a 12-way one" is another, and nothing measured so far
separates them. This tool answers the question the separation depends on:

    is the legacy 14-cell taxonomy definable WITHOUT the bug?

It is not obvious either way, because the taxonomy is not hand-designed. It is
whatever `{success}{middle}{ball}{reverse}` combinations clear `min_share`, so
the class count is an emergent property of the labels. If the two extra legacy
classes only exist because corrupted rows land in them, then "clean14" is not a
thing that can be built, and the experiment the plan wants cannot be run as
specified.

  python tools/failmode_taxonomy_audit.py
  python tools/failmode_taxonomy_audit.py --val-season 2023   # judging surface
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode as fm                                          # noqa: E402

# Only what _pitch_labels and build_cells touch. train.csv is 1.37M x 49 and
# the laptop does the analysis, so the frame stays narrow.
COLS = ["row_id", "season", "pitcher_id", "asof_pitcher_n", "control_success",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_reverse_rate"]


def describe(name):
    """Decode a cell code into words. 'xxx' is the reserved rare/unknown bin."""
    if name.endswith("xxx"):
        return f"success={name[0]}, failure modes not recovered (reserved)"
    s, mid, ball, rev = name[0], name[1], name[2], name[3]
    bits = [f"middle={mid}", f"ball={ball}", f"reverse={rev}"]
    return f"success={s}, " + ", ".join(bits)


def impossible(name):
    """Combinations that a correctly-labelled row cannot occupy.

    The pitch is one event. `middle` means it crossed the middle of the zone,
    `ball` means it missed the zone -- a pitch cannot do both, so middle=1 and
    ball=1 together is contradictory. And a pitch that went down the middle of
    the zone while the catcher asked for the other side is a control failure by
    definition, so success=1 with middle=1 is contradictory too.
    """
    if name.endswith("xxx"):
        return ""
    s, mid, ball, rev = name[0], name[1], name[2], name[3]
    if mid == "1" and ball == "1":
        return "middle and ball at once"
    if s == "1" and mid == "1":
        return "success while down the middle"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-season", type=int, default=2024,
                    help="2024 = submission surface (where +3.97 was measured)")
    ap.add_argument("--min-share", type=float, default=fm.MIN_SHARE)
    a = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                     encoding="utf-8-sig", usecols=COLS)
    # Same partition the real run uses: the taxonomy is frozen on the fit rows.
    fit = (tr["season"] != a.val_season).to_numpy()
    print(f"train {len(tr):,} rows | fit {fit.sum():,} | "
          f"val{a.val_season} {(~fit).sum():,} | min_share {a.min_share}")

    out = {}
    for label, legacy in (("corrected", False), ("legacy", True)):
        code, names, succ = fm.build_cells(
            tr, min_share=a.min_share, fit_mask=fit, legacy_shift=legacy,
            verbose=False)
        out[label] = (code, names, succ)
        print(f"\n{label}: {len(names)} cells, {len(succ)} carrying the success bit")

    ccode, cnames, csucc = out["corrected"]
    lcode, lnames, lsucc = out["legacy"]
    cser = pd.Series([cnames[i] for i in ccode], index=tr.index)
    lser = pd.Series([lnames[i] for i in lcode], index=tr.index)

    print("\n" + "=" * 78)
    print("CELL TABLE  (share over the whole frame)")
    print("=" * 78)
    print(f"{'cell':>6} {'corrected':>10} {'legacy':>10}  {'':2} meaning")
    cshare = cser.value_counts(normalize=True)
    lshare = lser.value_counts(normalize=True)
    for n in sorted(set(cnames) | set(lnames)):
        c = f"{cshare.get(n, 0) * 100:9.3f}%" if n in cnames else "        --"
        l = f"{lshare.get(n, 0) * 100:9.3f}%" if n in lnames else "        --"
        mark = "!!" if impossible(n) else ("+ " if n not in cnames else "  ")
        print(f"{n:>6} {c} {l}  {mark} {describe(n)}"
              + (f"   <- {impossible(n)}" if impossible(n) else ""))

    only_legacy = [n for n in lnames if n not in cnames]
    only_corr = [n for n in cnames if n not in lnames]
    print(f"\nonly in legacy : {only_legacy}")
    print(f"only in corrected: {only_corr}")

    print("\n" + "=" * 78)
    print("TRANSITIONS  corrected -> legacy, on rows where they disagree")
    print("=" * 78)
    moved = cser != lser
    print(f"{moved.sum():,} of {len(tr):,} rows move ({moved.mean() * 100:.4f}%)")
    x = pd.crosstab(cser[moved], lser[moved])
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(x)

    print("\ntop transitions")
    flat = x.stack().sort_values(ascending=False)
    for (frm, to), n in flat[flat > 0].head(12).items():
        tag = "  IMPOSSIBLE TARGET" if impossible(to) else ""
        print(f"  {frm} -> {to:>6}  {n:>8,}  ({n / moved.sum() * 100:5.2f}% of moves){tag}")

    # The decisive question. If every row in a legacy-only cell got there by
    # moving, the cell has no clean population and clean14 cannot be built.
    if only_legacy:
        print("\n" + "=" * 78)
        print("CAN THE EXTRA LEGACY CELLS SURVIVE WITHOUT THE BUG?")
        print("=" * 78)
        for n in only_legacy:
            rows = lser == n
            from_moved = (rows & moved).sum()
            print(f"  {n}: {rows.sum():,} rows, {from_moved:,} of them "
                  f"({from_moved / max(rows.sum(), 1) * 100:.1f}%) arrived by "
                  f"corruption. {impossible(n) or 'semantically possible'}")

    # What the success bit does -- the binary truth must never move.
    print("\nsuccess-bit integrity")
    cs = cser.str[0]
    ls = lser.str[0]
    y = tr["control_success"].astype(int).astype(str)
    print(f"  corrected cell success bit == control_success : {(cs == y).all()}")
    print(f"  legacy    cell success bit == control_success : {(ls == y).all()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

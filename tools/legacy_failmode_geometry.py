"""What the legacy failure-mode construction changed about the label geometry.

The legacy (pre-2026-08-13) `_pitch_labels` shifted globally instead of within
each pitcher, so 6.72% of rows took the next pitcher's first difference. Holding
the base family fixed, that construction scores **core +4.15, SE 1.11, t 3.76,
6/6 positive** against the corrected one -- statistically KEEP-grade, mechanism
unisolated, and `failmode-noise12-row-random` showed generic row-random
corruption at the same rate does not reproduce it.

This does not try to reproduce the bug, define `clean14`, or derive a
boundary-targeted variant -- all three are banned. The question is narrower and
general: **what did structured auxiliary corruption do to the supervision
geometry the cell arm learns from?** Class balance, entropy, effective number of
classes, and where the moved rows land.

Nothing here changes a model or proposes a feature.

  python tools/legacy_failmode_geometry.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode as fm                                            # noqa: E402

TARGET = "control_success"


def entropy(p):
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def describe(name, cell, names):
    v = cell.value_counts().sort_values(ascending=False)
    share = (v / v.sum()).to_numpy()
    h = entropy(share)
    print(f"\n{name}: {len(v)} occupied cells of {len(names)} in the taxonomy")
    print(f"  entropy {h:.5f} nats | effective classes exp(H) = {np.exp(h):.3f}"
          f" | dominant share {100 * share[0]:.3f}%"
          f" | imbalance max/min {share[0] / share[-1]:.1f}x")
    return v, share, h


def main():
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      encoding="utf-8-sig",
                      usecols=header + [TARGET, "pitcher_id"])
    # judging surface frame, the one the +4.15 was measured near
    raw = raw[~((raw.game_type == "F") & (raw.season <= 2022))]
    tr = raw[raw.season <= 2023].reset_index(drop=True)
    fit_mask = (tr.season <= 2022).to_numpy()
    print(f"frame {len(tr):,} rows | fit {int(fit_mask.sum()):,} "
          f"| val {int((~fit_mask).sum()):,}")

    out = {}
    for tag, legacy in (("corrected", False), ("legacy", True)):
        cell, names, succ = fm.build_cells(
            tr, modes=("middle", "ball", "reverse"), min_share=0.005,
            fit_mask=fit_mask, legacy_shift=legacy)
        # build_cells returns integer codes into `names`, and the two
        # taxonomies have different lengths -- so codes are not comparable
        # across them. Everything below compares cell *names*.
        cell = pd.Series(cell).map(lambda i: names[i]).rename("cell")
        out[tag] = (cell, names, succ)
        v, share, h = describe(tag, cell, names)
        print("  " + "  ".join(f"{k}:{100 * n / len(cell):.2f}%"
                               for k, n in v.head(8).items()))
        s = cell.isin([names[i] for i in succ])
        for blk, m in (("success block", s), ("failure block", ~s)):
            sub = cell[m].value_counts()
            p = (sub / sub.sum()).to_numpy()
            print(f"  {blk:14} {int(m.sum()):>9,} rows ({100 * m.mean():5.2f}%)"
                  f"  cells {len(sub):2d}  entropy {entropy(p):.5f}"
                  f"  eff {np.exp(entropy(p)):.3f}")

    c0, n0, s0 = out["corrected"]
    c1, n1, s1 = out["legacy"]
    print(f"\ntaxonomy: corrected {len(n0)} cells, legacy {len(n1)} cells")
    print(f"  only in legacy   : {sorted(set(n1) - set(n0))}")
    print(f"  only in corrected: {sorted(set(n0) - set(n1))}")

    moved = (c0.to_numpy() != c1.to_numpy())
    print(f"\nrows whose cell changes: {int(moved.sum()):,} "
          f"({100 * moved.mean():.3f}%)")
    succ0 = c0.isin([n0[i] for i in s0]).to_numpy()
    succ1 = c1.isin([n1[i] for i in s1]).to_numpy()
    print(f"  of those, success-bit flips: {int((succ0 != succ1).sum()):,} "
          f"(must be 0 -- the bit is read from the target)")
    print(f"  target rate on moved rows {tr[TARGET].to_numpy()[moved].mean():.6f}"
          f" vs {tr[TARGET].mean():.6f} overall")

    mv = pd.crosstab(c0[moved], c1[moved])
    print("\nwhere the moved rows go (corrected cell -> legacy cell, top 12):")
    flat = mv.stack().sort_values(ascending=False).head(12)
    for (a, b), n in flat.items():
        print(f"  {a} -> {b:8} {n:>8,}  ({100 * n / moved.sum():5.2f}% of moved)")

    # what a rarer taxonomy costs the learner: how much of the mass sits in
    # cells small enough that a tree sees them only occasionally
    for tag in ("corrected", "legacy"):
        cell = out[tag][0]
        v = cell.value_counts(normalize=True).sort_values()
        rare = v[v < 0.01]
        print(f"\n{tag}: cells under 1% of rows: {len(rare)} "
              f"holding {100 * rare.sum():.3f}% of the frame")


if __name__ == "__main__":
    main()

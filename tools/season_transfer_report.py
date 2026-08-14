"""Read the transfer grid and classify each family. Rules fixed before looking.

Absolute ΔBSS is not comparable across seasons — the same model scores 2401 on
2022 and 856 on 2024, so a family can shed two thirds of its ΔBSS while holding
exactly its slice of what the model explains. Retention is therefore read on
**share of the season's total positive ΔBSS**, with an absolute floor so that
noise around zero cannot produce a ratio.

  WEAK_BOTH        neither side reaches 2% of the slice
  SIGN_FLIP        permuting it *helps* on one side by more than 1 BSS
  NEXT_ONLY        share at least doubles into the unseen season, and >= 5%
  STABLE_POSITIVE  keeps >= 70% of its share, both sides >= 2%
  COLLAPSED        keeps < 50% of its share, from a base of >= 5%
  MIXED            everything else

`in_sample_val` is the model's own validation season, which the refit absorbed,
so it is memorisation-inflated. `in_sample_old` is a season the model trained on
but never validated against; a family that is strong on both and collapses only
on the unseen season has lost transfer, while one that climbs monotonically
toward the recent season was measuring recency.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

FLOOR, STABLE, COLLAPSE, NEXT_MULT = 2.0, 0.70, 0.50, 2.0


def classify(same, nxt, d_same, d_next):
    if min(d_same, d_next) < -1.0 and (d_same > 0) != (d_next > 0):
        return "SIGN_FLIP"
    if max(same, nxt) < FLOOR:
        return "WEAK_BOTH"
    if same > 0 and nxt >= NEXT_MULT * same and nxt >= 5.0:
        return "NEXT_ONLY"
    if same >= FLOOR and nxt >= FLOOR and nxt >= STABLE * same:
        return "STABLE_POSITIVE"
    if same >= 5.0 and nxt < COLLAPSE * same:
        return "COLLAPSED"
    return "MIXED"


def pivot(df, arm, league):
    d = df[(df.arm == arm) & (df.league == league)]
    if d.empty:
        return None
    # average the seeds within a (model-era, season, family)
    g = (d.groupby(["fit_era", "horizon", "season", "family"], as_index=False)
          .agg(dbss=("dbss", "mean"), share=("share", "mean"),
               seeds=("seed", "nunique") if "seed" in d else ("dbss", "size")))
    return g


def one_transition(g, era, order):
    """same-season vs next-season for one fitted model."""
    s = g[(g.fit_era == era) & (g.horizon == "in_sample_val")].set_index("family")
    o = g[(g.fit_era == era) & (g.horizon == "in_sample_old")].set_index("family")
    n = g[(g.fit_era == era) & (g.horizon == "next_season")].set_index("family")
    if s.empty or n.empty:
        return None
    rows = []
    for f in order:
        if f not in s.index or f not in n.index:
            continue
        rows.append(dict(
            family=f,
            old_share=o.share.get(f, np.nan), same_share=s.share[f],
            next_share=n.share[f],
            old_dbss=o.dbss.get(f, np.nan), same_dbss=s.dbss[f],
            next_dbss=n.dbss[f],
            retention=(n.share[f] / s.share[f] if s.share[f] > 0.05 else np.nan),
            klass=classify(s.share[f], n.share[f], s.dbss[f], n.dbss[f])))
    t = pd.DataFrame(rows)
    t["same_rank"] = t.same_share.rank(ascending=False).astype(int)
    t["next_rank"] = t.next_share.rank(ascending=False).astype(int)
    return t


def show(t, title):
    print(f"\n### {title}")
    print(f"{'family':14} {'old%':>7} {'same%':>7} {'next%':>7} "
          f"{'retain':>7} {'same_d':>9} {'next_d':>9} {'rank':>7}  class")
    for _, r in t.sort_values("same_share", ascending=False).iterrows():
        ret = "  n/a" if not np.isfinite(r.retention) else f"{r.retention:6.2f}"
        old = "    -" if not np.isfinite(r.old_share) else f"{r.old_share:6.2f}"
        print(f"{r.family:14} {old:>7} {r.same_share:6.2f} {r.next_share:6.2f} "
              f"{ret:>7} {r.same_dbss:+9.1f} {r.next_dbss:+9.1f} "
              f"{r.same_rank:2d}>{r.next_rank:<2d}   {r.klass}")
    rho = t[["same_share", "next_share"]].corr(method="spearman").iloc[0, 1]
    print(f"  rank stability (spearman same vs next) = {rho:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="out/season_transfer_map.csv")
    ap.add_argument("--core", default="out/season_transfer_core.csv")
    ap.add_argument("--league", default="R")
    args = ap.parse_args()

    parts = []
    if os.path.exists(args.map):
        parts.append(pd.read_csv(args.map))
    if os.path.exists(args.core):
        c = pd.read_csv(args.core)
        c["seed"] = -1
        parts.append(c)
    df = pd.concat(parts, ignore_index=True)
    order = df.family.drop_duplicates().tolist()

    tables = {}
    for arm in ("base", "cell", "core"):
        g = pivot(df, arm, args.league)
        if g is None:
            continue
        for era in sorted(g.fit_era.unique()):
            t = one_transition(g, era, order)
            if t is None:
                continue
            tables[(arm, era)] = t
            show(t, f"{arm.upper()} arm | fit<={era}, unseen {era + 1} "
                    f"[{args.league} league]")

    # agreement across the independent boundaries
    print("\n\n### transition agreement (next-season share), "
          f"{args.league} league")
    for arm in ("base", "cell", "core"):
        eras = sorted(e for a, e in tables if a == arm)
        if len(eras) < 2:
            continue
        w = pd.DataFrame({e: tables[(arm, e)].set_index("family").next_share
                          for e in eras})
        print(f"\n{arm.upper()}  columns = fit era")
        print(w.round(2).to_string())
        for i in range(len(eras) - 1):
            a, b = eras[i], eras[i + 1]
            rho = w[[a, b]].corr(method="spearman").iloc[0, 1]
            print(f"  spearman({a},{b}) = {rho:+.3f}")
        cls = pd.DataFrame({e: tables[(arm, e)].set_index("family").klass
                            for e in eras})
        agree = [f for f in cls.index if cls.loc[f].nunique() == 1]
        print(f"  same class on every boundary: {agree}")

    return tables


if __name__ == "__main__":
    main()

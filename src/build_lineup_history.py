"""Build leakage-safe historical batting-order role features from Trackman.

For each game half, plate appearances are ordered by pitch_no and assigned
cyclic slots 1..9.  For prediction season S, only games from seasons < S are
aggregated.  Inference is therefore a frozen batter lookup and row-independent.
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/lineup_history.csv"


def main():
    bmap = pd.read_csv(f"{DATA}/processed/batter_map2.csv")
    tm_to_batter = bmap.set_index("tm_batter_id")["batter_id"]
    cols = ["season", "trackman_game_id", "pitch_no", "inning", "top_bottom",
            "pitch_of_pa", "batter_trackman_id"]
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", usecols=cols)
    pa = tm[tm.pitch_of_pa == 1].copy()
    pa["batter_id"] = pa.batter_trackman_id.map(tm_to_batter)
    linked = pa.batter_id.notna()
    print(f"plate appearances {len(pa):,} | linked {linked.mean():.1%}")
    pa = pa[linked].copy()
    pa["batter_id"] = pa.batter_id.astype(np.int64)
    pa = pa.sort_values(["trackman_game_id", "top_bottom", "pitch_no"])
    grp = ["trackman_game_id", "top_bottom"]
    # Require a complete start of the batting cycle; otherwise modulo slots shift.
    starts_first = pa.groupby(grp).inning.transform("min").eq(1)
    pa = pa[starts_first].copy()
    pa["pa_i"] = pa.groupby(grp).cumcount()
    pa["slot"] = (pa.pa_i % 9 + 1).astype(np.int8)
    size = pa.groupby(grp).slot.transform("size")
    pa = pa[size >= 9].copy()

    slot_counts = pa.groupby(["batter_id", "season", "slot"]).size().rename("n") \
        .reset_index()
    seasons = range(int(pa.season.min()) + 1, int(pa.season.max()) + 2)
    rows = []
    for pred_season in seasons:
        hist = slot_counts[slot_counts.season < pred_season]
        if hist.empty:
            continue
        tab = hist.groupby(["batter_id", "slot"]).n.sum().unstack(fill_value=0)
        tab = tab.reindex(columns=range(1, 10), fill_value=0)
        n = tab.sum(axis=1)
        prob = tab.div(n, axis=0)
        mode = tab.idxmax(axis=1)
        entropy = -(prob.where(prob > 0) * np.log(prob.where(prob > 0))).sum(axis=1)
        rec = pd.DataFrame({
            "batter_id": tab.index,
            "season": pred_season,
            "lineup_n": n,
            "lineup_slot": mode.astype(np.int8),
            "lineup_stability": prob.max(axis=1),
            "lineup_entropy": entropy.fillna(0.0),
            "lineup_top_share": prob[[1, 2]].sum(axis=1),
            "lineup_core_share": prob[[3, 4, 5]].sum(axis=1),
            "lineup_lower_share": prob[[6, 7, 8, 9]].sum(axis=1),
        }).reset_index(drop=True)
        rows.append(rec)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"usable PA {len(pa):,} | gamesides {pa.groupby(grp).ngroups:,}")
    print(f"saved {OUT}: {len(out):,} rows, batter={out.batter_id.nunique():,}, "
          f"season={out.season.min()}..{out.season.max()}")
    y25 = out[out.season == 2025]
    print(f"2025 lookup batter={len(y25):,} | median n={y25.lineup_n.median():.0f} "
          f"stability={y25.lineup_stability.median():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

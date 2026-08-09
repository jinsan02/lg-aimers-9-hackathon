"""Year-to-year stability audit for defensive-team failure-mode calling style."""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


def main():
    cols = ["row_id", "season", "game_type", "pitcher_id", "pitcher_team_id",
            "balls_before", "strikes_before", "num_runners_on", "batter_hand",
            "asof_pitcher_n", "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate",
            "control_success"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    lab = failmode._pitch_labels(d, modes=("middle", "ball", "reverse"))
    d = pd.concat([d, lab], axis=1)
    d = d[(d.game_type == "R") & d.middle.notna() & d.ball.notna() & d.reverse.notna()].copy()
    d["count_family"] = np.select(
        [d.balls_before == 3,
         (d.strikes_before == 2) & (d.balls_before < 3),
         d.balls_before == d.strikes_before],
        ["must_strike", "chase", "neutral"], default="other")
    d["runner"] = (d.num_runners_on > 0).astype(np.int8)
    keys = ["season", "pitcher_team_id", "count_family", "runner", "batter_hand"]
    modes = ["middle", "ball", "reverse"]
    g = d.groupby(keys)[modes].agg(["sum", "size"])
    # Flatten duplicated size columns, which are identical within each group.
    n = g[("middle", "size")].to_numpy(float)
    tab = g.index.to_frame(index=False)
    for mode in modes:
        tab[mode] = g[(mode, "sum")].to_numpy(float)
    tab["n"] = n

    ctx = ["season", "count_family", "runner", "batter_hand"]
    pri = d.groupby(ctx)[modes].mean().reset_index()
    tab = tab.merge(pri, on=ctx, suffixes=("_sum", "_prior"))
    k = 300.0
    for mode in modes:
        rate = (tab[f"{mode}_sum"]+k*tab[f"{mode}_prior"])/(tab.n+k)
        tab[f"{mode}_dev"] = rate-tab[f"{mode}_prior"]

    rows = []
    for s in range(2019, 2024):
        a = tab[tab.season == s].copy()
        b = tab[tab.season == s+1].copy()
        join = ["pitcher_team_id", "count_family", "runner", "batter_hand"]
        m = a.merge(b, on=join, suffixes=("_a", "_b"))
        vals = []
        for mode in modes:
            c = corr(m[f"{mode}_dev_a"].to_numpy(), m[f"{mode}_dev_b"].to_numpy())
            vals.append(c)
            rows.append({"from": s, "to": s+1, "mode": mode,
                         "corr": c, "cells": len(m)})
        va = np.concatenate([m[f"{x}_dev_a"].to_numpy() for x in modes])
        vb = np.concatenate([m[f"{x}_dev_b"].to_numpy() for x in modes])
        rows.append({"from": s, "to": s+1, "mode": "combined",
                     "corr": corr(va, vb), "cells": len(m)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    c = out.loc[out["mode"] == "combined", "corr"].to_numpy(float)
    print(f"combined median={np.nanmedian(c):+.3f} min={np.nanmin(c):+.3f} "
          f"last={c[-1]:+.3f} gate={np.nanmedian(c) >= .25 and c[-1] > 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

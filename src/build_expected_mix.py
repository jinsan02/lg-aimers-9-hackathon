"""EXP-D -- expected pitch mix: personal rate x league context ratio.

The plan proposes multiplying each pitcher's own pitch-group rates by a
situational league ratio taken from Trackman, without ever joining player IDs.

Re-scoped after the EXP-000 audit. Both halves already exist in some form
(`src/tm_context.py` builds linkage-free context aggregates,
`src/build_tm_pitchmix.py` builds P(group | pitcher, count)) and the champion
already carries **twelve** pitch-mix columns, not three:

    asof_pitcher_{fastball,breaking,offspeed}_rate       (+ _shr)
    std_asof_pitcher_{fastball,breaking,offspeed}_rate   (+ _delta)

So the only new element is the context ratio multiplier

    context_ratio_g = P(g | context) / P(g)

and the question is whether it survives a correlation check against those
twelve, not against three. `--tm-feats` is already CLOSED at -6.04 and
`predicted-pitch-probability-features` at roughly +2 to +3; this sits between
them.

Temporal safety: a row in season S uses Trackman seasons **< S** only. 2019 has
no prior Trackman season, so its ratio is exactly 1.0 and the feature reduces
to the personal rate the model already has. 2025 inference may use <= 2024.

    python src/build_expected_mix.py
    -> data/processed/expected_mix.csv  (row_id, exp_fastball, exp_breaking, exp_offspeed)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA = "./data"
CTX = ["balls_before", "strikes_before", "pitcher_hand", "batter_hand"]
GROUPS = ["fastball", "breaking", "offspeed"]
K_SMOOTH = 50.0          # cells are large here; the sweep lives in the caller


def context_tables(tm):
    """P(g | context) and P(g), cumulative over seasons strictly before S."""
    seasons = sorted(tm.season.unique())
    out = {}
    for s in seasons + [max(seasons) + 1]:
        past = tm[tm.season < s]
        if not len(past):
            out[s] = None
            continue
        glob = past.groupby("g").size()
        glob = glob / glob.sum()
        cell = past.groupby(CTX + ["g"]).size().unstack("g").fillna(0.0)
        n = cell.sum(axis=1)
        # Shrink each cell toward the global mix so a thin situation cannot
        # invent a ratio. n is large here (hundreds of thousands), so this
        # mostly matters for the extreme counts.
        p = cell.div(n, axis=0)
        for g in GROUPS:
            if g not in p.columns:
                p[g] = 0.0
            p[g] = (n * p[g] + K_SMOOTH * glob.get(g, 0.0)) / (n + K_SMOOTH)
        p = p[GROUPS].div(p[GROUPS].sum(axis=1), axis=0)
        out[s] = (p, glob.reindex(GROUPS).fillna(0.0))
    return out


def main():
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitch_type_group"] + CTX)
    tm = tm.rename(columns={"pitch_type_group": "g"})
    tm = tm[tm.g.isin(GROUPS)]
    # Trackman spells hands "Right"/"Left"; train.csv codes them 2/1. Without
    # this the context reindex misses every row, the ratio comes back 1.0
    # everywhere, and the feature is a byte copy of the personal rate -- which
    # is exactly what the first run produced (correlation 1.000000, sd of the
    # difference 0.000000). Same mapping as src/tm_context.py:34.
    for c in ("pitcher_hand", "batter_hand"):
        tm[c] = tm[c].map({"Left": 1, "Right": 2})
    bad = int(tm[["pitcher_hand", "batter_hand"]].isna().any(axis=1).sum())
    if bad:
        print(f"  dropping {bad:,} trackman rows with an unmapped hand")
        tm = tm.dropna(subset=["pitcher_hand", "batter_hand"])
    tm[["pitcher_hand", "batter_hand"]] = tm[["pitcher_hand", "batter_hand"]].astype(int)
    print(f"trackman {len(tm):,} rows, seasons {sorted(tm.season.unique())}")

    tabs = context_tables(tm)
    tr = pd.read_csv(f"{DATA}/train.csv",
                     usecols=["row_id", "season"] + CTX
                     + [f"asof_pitcher_{g}_rate" for g in GROUPS])
    print(f"train {len(tr):,} rows")

    exp = np.full((len(tr), 3), np.nan)
    for s, idx in tr.groupby("season").groups.items():
        t = tabs.get(s)
        sub = tr.loc[idx]
        own = sub[[f"asof_pitcher_{g}_rate" for g in GROUPS]].to_numpy(float)
        own = np.where(np.isfinite(own), own, np.nan)
        if t is None:
            ratio = np.ones((len(sub), 3))          # no past Trackman season
        else:
            p, glob = t
            key = pd.MultiIndex.from_arrays([sub[c] for c in CTX])
            hit = p.reindex(key)
            ratio = hit[GROUPS].to_numpy(float) / glob.to_numpy(float)[None, :]
            miss = float(np.isnan(ratio[:, 0]).mean())
            if miss > 0.01:
                raise SystemExit(
                    f"season {s}: {miss * 100:.2f}% of rows missed the context "
                    f"table. A silent miss returns ratio 1.0 and the feature "
                    f"becomes a copy of the personal rate.")
            ratio = np.where(np.isfinite(ratio), ratio, 1.0)
        raw = own * ratio
        tot = np.nansum(raw, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            exp[tr.index.get_indexer(idx)] = np.where(tot > 0, raw / tot, np.nan)
        print(f"  season {s}: trackman<{s} "
              f"{'none' if t is None else 'yes'} | ratio mean "
              f"{np.nanmean(ratio, axis=0).round(4)}")

    out = pd.DataFrame({"row_id": tr.row_id.to_numpy()})
    for i, g in enumerate(GROUPS):
        out[f"exp_{g}_prob"] = exp[:, i]
    os.makedirs(f"{DATA}/processed", exist_ok=True)
    out.to_csv(f"{DATA}/processed/expected_mix.csv", index=False)
    print(f"\nsaved data/processed/expected_mix.csv  "
          f"missing {out.iloc[:, 1].isna().mean() * 100:.2f}%")


if __name__ == "__main__":
    main()

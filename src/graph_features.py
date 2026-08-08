"""Season-expanding pitcher-batter graph features.

Every row in season S is transformed from edges and labels in seasons < S.
The frozen tables stored in the artifact are used row-by-row at inference; test
rows are never connected to one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


PKEY, BKEY, TARGET = "pitcher_id", "batter_id", "control_success"
COLS = [
    "g_pitcher_degree", "g_batter_degree", "g_pair_n", "g_pair_seen",
    "g_pitcher_neigh_rate", "g_pitcher_neigh_sd",
    "g_batter_neigh_rate", "g_batter_neigh_sd",
]


def _weighted_stats(edges: pd.DataFrame, value: str, key: str) -> pd.DataFrame:
    x = edges.dropna(subset=[value]).copy()
    x["_w"] = np.sqrt(x["g_pair_n"].astype(float))
    x["_wx"] = x["_w"] * x[value]
    x["_wx2"] = x["_w"] * x[value] ** 2
    g = x.groupby(key, sort=False)[["_w", "_wx", "_wx2"]].sum()
    mean = g["_wx"] / g["_w"]
    var = (g["_wx2"] / g["_w"] - mean ** 2).clip(lower=0)
    return pd.DataFrame({"mean": mean, "sd": np.sqrt(var)})


def build_tables(history: pd.DataFrame) -> dict:
    """Build frozen lookup tables from completed historical seasons only."""
    if history.empty:
        return {"pair": pd.DataFrame(columns=[PKEY, BKEY, "g_pair_n"]),
                "pitcher": pd.DataFrame(columns=[PKEY]),
                "batter": pd.DataFrame(columns=[BKEY])}
    pair = (history.groupby([PKEY, BKEY], sort=False).size()
            .rename("g_pair_n").reset_index())
    pr = history.groupby(PKEY, sort=False)[TARGET].mean().rename("_p_rate")
    br = history.groupby(BKEY, sort=False)[TARGET].mean().rename("_b_rate")
    edge = pair.join(pr, on=PKEY).join(br, on=BKEY)
    pn = _weighted_stats(edge, "_b_rate", PKEY).rename(
        columns={"mean": "g_pitcher_neigh_rate", "sd": "g_pitcher_neigh_sd"})
    bn = _weighted_stats(edge, "_p_rate", BKEY).rename(
        columns={"mean": "g_batter_neigh_rate", "sd": "g_batter_neigh_sd"})
    pdeg = pair.groupby(PKEY, sort=False)[BKEY].nunique().rename("g_pitcher_degree")
    bdeg = pair.groupby(BKEY, sort=False)[PKEY].nunique().rename("g_batter_degree")
    pitcher = pd.concat([pdeg, pn], axis=1).reset_index()
    batter = pd.concat([bdeg, bn], axis=1).reset_index()
    return {"pair": pair, "pitcher": pitcher, "batter": batter}


def apply_tables(df: pd.DataFrame, tables: dict) -> tuple[pd.DataFrame, list[str]]:
    """Left-join frozen lookups; no statistic is computed from ``df``."""
    out = df.merge(tables["pitcher"], on=PKEY, how="left", sort=False)
    out = out.merge(tables["batter"], on=BKEY, how="left", sort=False)
    out = out.merge(tables["pair"], on=[PKEY, BKEY], how="left", sort=False)
    # merge preserves left order but not the original index.
    out.index = df.index
    for c in COLS:
        if c not in out:
            out[c] = np.nan
    out["g_pair_seen"] = out["g_pair_n"].notna().astype(np.int8)
    for c in ("g_pitcher_degree", "g_batter_degree", "g_pair_n"):
        out[c] = out[c].fillna(0).astype(np.float32)
    for c in COLS:
        if c not in ("g_pair_seen",):
            out[c] = out[c].astype(np.float32)
    return out, COLS.copy()


def add_expanding(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Attach honest features and return the final frozen inference tables."""
    parts = []
    for season in sorted(df["season"].unique()):
        cur = df[df["season"] == season]
        past = df[(df["season"] < season) & ~df.get(
            "_is_test", pd.Series(False, index=df.index)).astype(bool)]
        got, _ = apply_tables(cur, build_tables(past))
        parts.append(got)
    out = pd.concat(parts).sort_index()
    history = df[~df.get("_is_test", pd.Series(False, index=df.index)).astype(bool)]
    return out, COLS.copy(), build_tables(history)

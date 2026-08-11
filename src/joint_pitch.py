"""Training-only recovery of current pitch-type labels from provided Trackman.

The label is accepted only when the shared pre-pitch context plus the already
validated anonymous pitcher/batter mappings identify exactly one row on both
sides.  It must never be used as an inference feature: current pitch type is
known only after the pitch.  Its legal use is a masked auxiliary target whose
head is discarded at inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data")
TYPES = ["fastball", "breaking", "offspeed"]
CONTEXT = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_hand",
    "batter_hand",
]
KEY = CONTEXT + ["pitcher_id", "batter_id"]


def exact_pitch_labels() -> pd.DataFrame:
    """Return ``row_id, pitch_label`` for strict one-to-one joined rows."""
    main = pd.read_csv(DATA / "train.csv", usecols=["row_id"] + KEY)
    tm = pd.read_csv(
        DATA / "trackman_history.csv", encoding="utf-8-sig",
        usecols=CONTEXT + ["pitcher_trackman_id", "batter_trackman_id",
                           "pitch_type_group"],
    )
    pm = pd.read_csv(DATA / "processed/pitcher_map2.csv")
    bm = pd.read_csv(DATA / "processed/batter_map2.csv")
    pmap = pm.drop_duplicates("tm_id").set_index("tm_id")["pitcher_id"]
    bmap = bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id")["batter_id"]
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(pmap)
    tm["batter_id"] = tm.batter_trackman_id.map(bmap)
    tm["top_bottom"] = tm.top_bottom.map({"Top": "T", "Bottom": "B"})
    for col in ["pitcher_hand", "batter_hand"]:
        tm[col] = tm[col].map({"Left": 1, "Right": 2})
    tm = tm[tm.pitcher_id.notna() & tm.batter_id.notna()].copy()
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    tm["batter_id"] = tm.batter_id.astype(np.int64)

    main_count = main.groupby(KEY, dropna=False).size().rename("main_n")
    tm_group = tm.groupby(KEY, dropna=False).agg(
        tm_n=("pitch_type_group", "size"),
        pitch_type_group=("pitch_type_group", "first"),
    )
    safe = (main_count[main_count == 1].rename_axis(KEY).reset_index()
            .drop(columns="main_n")
            .merge(tm_group[(tm_group.tm_n == 1)
                            & tm_group.pitch_type_group.isin(TYPES)]
                   .reset_index().drop(columns="tm_n"), on=KEY, how="inner"))
    out = main[["row_id"] + KEY].merge(safe, on=KEY, how="inner")
    out["pitch_label"] = out.pitch_type_group.map(
        {name: i for i, name in enumerate(TYPES)}).astype(np.int8)
    return out[["row_id", "pitch_label"]]


def labels_for_rows(row_ids: pd.Series) -> np.ndarray:
    tab = exact_pitch_labels().set_index("row_id").pitch_label
    labels = row_ids.map(tab).fillna(-1).to_numpy(np.int8)
    return labels


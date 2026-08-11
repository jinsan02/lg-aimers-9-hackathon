"""Audit whether train control labels can be joined to Trackman pitch labels.

The contest table has ``control_success`` but no current-pitch type, while the
provided Trackman table has pitch type but no contest target.  This script uses
only shared pre-pitch context plus the existing high-confidence anonymous
player mappings, then reports ambiguity at the *group* level.  It does not
write a joined training set: even equal-size groups cannot be ordered safely
without a shared game or pitch identifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA = Path("data")
OUT = Path("out")
CONTEXT = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom",
    "balls_before", "strikes_before", "outs_before", "pitcher_hand",
    "batter_hand",
]
KEY = CONTEXT + ["pitcher_id", "batter_id"]
TYPES = {"fastball", "breaking", "offspeed"}


def pct(mask: pd.Series, weight: pd.Series) -> float:
    return float(weight[mask].sum() / weight.sum())


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv", usecols=KEY)
    tm = pd.read_csv(
        DATA / "trackman_history.csv", encoding="utf-8-sig",
        usecols=CONTEXT + ["pitcher_trackman_id", "batter_trackman_id",
                           "pitch_type_group"],
    )
    pm = pd.read_csv(DATA / "processed/pitcher_map2.csv")
    bm = pd.read_csv(DATA / "processed/batter_map2.csv")
    pmap = pm.drop_duplicates("tm_id").set_index("tm_id")["pitcher_id"]
    bmap = bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id")["batter_id"]

    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(pmap)
    tm["batter_id"] = tm["batter_trackman_id"].map(bmap)
    tm["top_bottom"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    for col in ["pitcher_hand", "batter_hand"]:
        tm[col] = tm[col].map({"Left": 1, "Right": 2})

    player_mapped = tm.pitcher_id.notna() & tm.batter_id.notna()
    tm = tm.loc[player_mapped].copy()
    tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
    tm["batter_id"] = tm.batter_id.astype(np.int64)
    tm["pitch_label_ok"] = tm.pitch_type_group.isin(TYPES)

    trg = tr.groupby(KEY, dropna=False).size().rename("tr_n").reset_index()
    tmg = tm.groupby(KEY, dropna=False).agg(
        tm_n=("pitch_type_group", "size"),
        tm_labeled_n=("pitch_label_ok", "sum"),
        tm_type_nunique=("pitch_type_group", lambda x: x.dropna().nunique()),
        tm_first_type=("pitch_type_group", "first"),
    ).reset_index()
    g = trg.merge(tmg, on=KEY, how="left")
    for col in ["tm_n", "tm_labeled_n", "tm_type_nunique"]:
        g[col] = g[col].fillna(0).astype(np.int64)

    has = g.tm_n > 0
    one_to_one = (g.tr_n == 1) & (g.tm_n == 1) & (g.tm_labeled_n == 1)
    equal_size = has & (g.tr_n == g.tm_n)
    # Safe only as a common group label, not as an ordered row-to-row match.
    pure_labeled = has & (g.tm_labeled_n == g.tm_n) & \
        (g.tm_type_nunique == 1) & g.tm_first_type.isin(TYPES)
    equal_pure = equal_size & pure_labeled

    report = {
        "train_rows": int(g.tr_n.sum()),
        "trackman_rows_total": int(len(player_mapped)),
        "trackman_rows_both_players_mapped": int(player_mapped.sum()),
        "trackman_both_players_mapped_rate": float(player_mapped.mean()),
        "train_row_coverage_any_candidate": pct(has, g.tr_n),
        "train_row_coverage_one_tm_candidate": pct(g.tm_n == 1, g.tr_n),
        "train_row_coverage_one_to_one_labeled": pct(one_to_one, g.tr_n),
        "train_row_coverage_equal_group_size": pct(equal_size, g.tr_n),
        "train_row_coverage_type_pure_group": pct(pure_labeled, g.tr_n),
        "train_row_coverage_equal_size_and_pure": pct(equal_pure, g.tr_n),
        "candidate_multiplicity": {
            "0": pct(g.tm_n == 0, g.tr_n),
            "1": pct(g.tm_n == 1, g.tr_n),
            "2_to_3": pct(g.tm_n.between(2, 3), g.tr_n),
            "4_to_10": pct(g.tm_n.between(4, 10), g.tr_n),
            "11_plus": pct(g.tm_n >= 11, g.tr_n),
        },
    }
    OUT.mkdir(exist_ok=True)
    path = OUT / "joint_pitch_label_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

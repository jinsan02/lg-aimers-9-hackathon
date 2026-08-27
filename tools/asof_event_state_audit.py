"""CPU-only unique-signal gate for explicit ASOF season event mass."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import entity_unique_audit as unique  # noqa: E402
import research_frames as rf  # noqa: E402
import season_std  # noqa: E402

RATES = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
         "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate"]
NAMES = ["season_success_events", "season_middle_events",
         "season_ball_events", "season_reverse_events"]


def event_rows(data):
    anchors = season_std.build_anchors(data, last_pitch=True)["pitcher"]
    use = ["pitcher_id", "season", "n0"] + [c for c in anchors if c.startswith("S0_")
                                                     or c.startswith("n0_")]
    d = data.merge(anchors[use], on=["pitcher_id", "season"], how="left")
    n = d.asof_pitcher_n.to_numpy(float)
    vals, sanity = {}, {}
    for rate, name in zip(RATES, NAMES):
        n0c = f"n0_{rate}"
        den0 = d[n0c].fillna(0).to_numpy(float) if n0c in d else \
            d.n0.fillna(0).to_numpy(float)
        s0 = d[f"S0_{rate}"].fillna(0).to_numpy(float)
        raw = n * d[rate].fillna(0).to_numpy(float) - s0
        vals[name] = np.maximum(raw, 0.0)
        sanity[name] = {
            "negative_before_clamp": int(np.sum(raw < -1e-8)),
            "fractional_distance_median": float(np.nanmedian(np.abs(raw-np.round(raw)))),
            "max": float(np.nanmax(vals[name])),
        }
        if rate == RATES[0]:
            vals["season_n"] = np.maximum(n - den0, 0.0)
    out = d[["row_id", "season", "pitcher_id"]].copy()
    for k, v in vals.items(): out[k] = v
    return out, sanity


def profiles(rows, events):
    z = rows[["row_id", "pitcher_id"]].merge(events, on=["row_id", "pitcher_id"],
                                                how="left")
    cols = NAMES + ["season_n"]
    return z.groupby("pitcher_id")[cols].mean()


def main():
    data = rf.load_train()
    events, sanity = event_rows(data)
    existing = set()
    transfers = []
    for spec in rf.BOUNDARIES:
        b = rf.boundary(data, *spec)
        existing.update(b["numeric_features"])
        cp_s, cp_t = rf.champion_profiles(b, "pitcher_id")
        sp, tp = profiles(b["s"], events), profiles(b["t"], events)
        ans = unique.audit(b["s"], b["t"], sp, tp, cp_s, cp_t,
                           b["rs"], b["rt"], "pitcher_id")
        ans.update({"source": b["source"], "target": b["target"]})
        transfers.append(ans)
    explicit = sorted(set(NAMES + ["season_n"]) & existing)
    result = {
        "contract": {"columns": NAMES + ["season_n"], "ridge_alpha": unique.ALPHA,
                     "null_reps": unique.REPS, "aggregation_unit": "pitcher-season mean"},
        "duplicate_audit": {"explicit_columns_in_champion": explicit,
                            "status": "NEW_REPRESENTATION" if not explicit else "DUPLICATE"},
        "numerical_sanity": sanity,
        "transfers": transfers,
        "independence": {"pass": True,
            "reason": "existing frozen prior-season anchors plus current row ASOF values only"},
    }
    out = ROOT / "out/asof_event_state_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

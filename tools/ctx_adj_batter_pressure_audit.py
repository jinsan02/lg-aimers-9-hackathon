"""CPU-only unique-signal gate for context-adjusted batter pressure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import entity_unique_audit as unique  # noqa: E402
import research_frames as rf  # noqa: E402

K = 80.0
CAT = ["season", "game_dayofweek", "top_bottom", "game_type", "base_state",
       "pitcher_hand", "batter_hand", "pitcher_id"]
NUM = ["game_month", "inning", "balls_before", "strikes_before", "outs_before",
       "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
       "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
       "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li",
       "asof_pitcher_n", "asof_pitcher_success_rate",
       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
       "asof_pitcher_reverse_rate", "asof_pitcher_strike_rate",
       "asof_pitcher_prev1_game_success_rate",
       "asof_pitcher_prev3_game_success_rate",
       "asof_pitcher_prev5_game_success_rate"]


def context_model():
    prep = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT),
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), NUM),
    ])
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                        max_iter=30, tol=1e-4, random_state=20260827,
                        average=True)
    return make_pipeline(prep, clf)


def build_lookup(data, cutoff):
    hist = data[data.season < cutoff].copy()
    model = context_model().fit(hist[CAT + NUM], hist.control_success)
    q = model.predict_proba(hist[CAT + NUM])[:, 1]
    r = hist.control_success.to_numpy(float) - q
    g = (pd.DataFrame({"batter_id": hist.batter_id.to_numpy(), "r": r})
         .groupby("batter_id").r.agg(["sum", "size"]))
    skill = g["sum"] / (g["size"] + K)
    return skill, {"cutoff": cutoff,
                   "fit_seasons": sorted(map(int, hist.season.unique())),
                   "n_rows": int(len(hist)), "n_batters": int(len(skill)),
                   "q_mean": float(q.mean()), "residual_mean": float(r.mean())}


def profile(rows, lookup):
    ids = pd.Index(rows.batter_id.unique(), name="batter_id")
    return pd.DataFrame({"ctx_adj_batter_pressure":
                         ids.map(pd.Series(lookup)).fillna(0).to_numpy(float)},
                        index=ids)


def main():
    data = rf.load_train()
    lookups, meta = {}, []
    for cutoff in (2022, 2023, 2024):
        lookups[cutoff], m = build_lookup(data, cutoff)
        meta.append(m)
    transfers = []
    sanity = None
    for spec in rf.BOUNDARIES:
        b = rf.boundary(data, *spec)
        cp_s, cp_t = rf.champion_profiles(b, "batter_id")
        sp = profile(b["s"], lookups[b["source"]])
        tp = profile(b["t"], lookups[b["target"]])
        ans = unique.audit(b["s"], b["t"], sp, tp, cp_s, cp_t,
                           b["rs"], b["rt"], "batter_id")
        ans.update({"source": b["source"], "target": b["target"]})
        transfers.append(ans)
        if b["target"] == 2024:
            v = b["t"].batter_id.map(lookups[2024]).fillna(0).to_numpy(float)
            sanity = {
                "variance": float(np.var(v)),
                "coverage": float(b["t"].batter_id.isin(lookups[2024].index).mean()),
                "corr_raw_batter_success": unique.safe_corr(
                    v, b["tx"]["asof_batter_success_rate"]),
                "corr_std_batter_success": unique.safe_corr(
                    v, b["tx"]["std_asof_batter_success_rate"]),
                "corr_te_batter": unique.safe_corr(
                    v, b["tx"].get("te_batter_ratio", np.full(len(v), np.nan))),
            }
    result = {"contract": {"entity": "batter_id", "k": K,
                            "context_model": "SGD logistic",
                            "batter_history_inputs_present": False,
                            "null_reps": unique.REPS},
              "fit_meta": meta, "sanity": sanity, "transfers": transfers,
              "independence": {"pass": True,
                  "reason": "train-only frozen batter lookup; row-local batter_id join"}}
    out = ROOT / "out/ctx_adj_batter_pressure_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

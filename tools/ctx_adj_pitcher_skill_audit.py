"""CPU-only gate for context-adjusted historical pitcher skill.

One fixed nuisance model estimates P(success | context), explicitly excluding
pitcher identity and every pitcher-history feature.  Historical residuals are
then k=80 shrunk by pitcher.  The lookup for season S uses only seasons < S.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import fpipe  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402

K = 80.0
ALPHA = 100.0
CAT = ["season", "game_dayofweek", "top_bottom", "game_type", "base_state",
       "pitcher_hand", "batter_hand"]
NUM = ["game_month", "inning", "balls_before", "strikes_before", "outs_before",
       "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
       "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
       "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li",
       "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"]


def context_model():
    prep = ColumnTransformer([
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CAT),
        ("num", make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler()), NUM),
    ])
    # Fixed, deterministic, single-pass-friendly nuisance model; no sweep.
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                        max_iter=30, tol=1e-4, random_state=20260816,
                        average=True)
    return make_pipeline(prep, clf)


def skill_lookup(data: pd.DataFrame, cutoff: int) -> tuple[pd.Series, dict]:
    hist = data[data.season < cutoff].copy()
    model = context_model().fit(hist[CAT + NUM], hist.control_success)
    q = model.predict_proba(hist[CAT + NUM])[:, 1]
    r = hist.control_success.to_numpy(float) - q
    tab = pd.DataFrame({"pitcher_id": hist.pitcher_id.to_numpy(), "r": r})
    g = tab.groupby("pitcher_id").r.agg(["sum", "size"])
    skill = g["sum"] / (g["size"] + K)
    meta = {"cutoff": cutoff, "fit_seasons": sorted(map(int, hist.season.unique())),
            "n_rows": int(len(hist)), "n_pitchers": int(len(skill)),
            "q_mean": float(q.mean()), "resid_mean": float(r.mean())}
    return skill, meta


def champion(tag_base: str, tag_cell: str, kind: str):
    return core.core_predictions(tag_base, tag_cell, kind)


def transfer(data, source, target, base, cell, skills):
    ps, ys = champion(base, cell, "val")
    pt, yt = champion(base, cell, "test")
    s = data[data.season == source].reset_index(drop=True)
    t = data[data.season == target].reset_index(drop=True)
    if len(s) != len(ys) or len(t) != len(yt):
        raise RuntimeError("prediction/frame mismatch")
    xs = s.pitcher_id.map(skills[source]).fillna(0).to_numpy(float)[:, None]
    xt = t.pitcher_id.map(skills[target]).fillna(0).to_numpy(float)[:, None]
    rs, rt = ys - ps, yt - pt
    fit, add = core.ridge_transfer(xs, xt, rs)
    rho = core._safe_corr(add, rt)
    null = core.matched_null(s, t, xs, xt, rs, rt, reps=400)
    masks = {"R": t.game_type.eq("R"), "F": t.game_type.eq("F"),
             "early": t.game_month.le(6), "late": t.game_month.gt(6)}
    return {"source": source, "target": target,
            "rho_source": core._safe_corr(fit, rs), "rho": rho,
            "null_median": float(np.nanmedian(null)),
            "null_p95": float(np.nanquantile(null, .95)),
            "null_p99": float(np.nanquantile(null, .99)),
            "null_percentile": float(np.nanmean(null <= abs(rho)) * 100),
            "segments": {k: core._safe_corr(add[m], rt[m]) for k, m in masks.items()}}


def novelty_and_sanity(data, skills):
    blocks = []
    for season, tag in ((2023, "BND22"), (2024, "B1J6")):
        pack = joblib.load(ROOT / f"model/cat_{tag}_base_s3.pkl")
        raw = data[data.season == season].reset_index(drop=True)
        transformed = fpipe.transform(raw.drop(columns="control_success").copy(), pack["fpipe"])
        nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
        agg = transformed[nums].assign(pitcher_id=raw.pitcher_id.to_numpy()).groupby("pitcher_id").mean()
        z = pd.DataFrame({"ctx_skill": skills[season]}).join(agg, how="inner")
        z["season"] = season
        blocks.append(z.reset_index())
    z = pd.concat(blocks, ignore_index=True)
    nums = [c for c in z if c not in ("pitcher_id", "ctx_skill", "season")]
    x = z[nums].to_numpy(float)
    med = np.nanmedian(x, axis=0); med = np.where(np.isfinite(med), med, 0.)
    x = np.where(np.isfinite(x), x, med)
    cv = GroupKFold(5)
    pred = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)),
                             x, z.ctx_skill, cv=cv, groups=z.pitcher_id)
    r2 = float(r2_score(z.ctx_skill, pred))

    pack = joblib.load(ROOT / "model/cat_B1J6_base_s3.pkl")
    raw = data[data.season == 2024].reset_index(drop=True)
    tr = fpipe.transform(raw.drop(columns="control_success").copy(), pack["fpipe"])
    # The shipped champion's cell member additionally carries general
    # ``skill_hat``; use it only for the requested sanity correlation, not to
    # change the pre-specified 121-feature reconstructibility gate above.
    gpack = joblib.load(ROOT / "model/cat_GSKDEP_cell_s3.pkl")
    gtr = fpipe.transform(raw.drop(columns="control_success").copy(), gpack["fpipe"])
    v = raw.pitcher_id.map(skills[2024]).to_numpy(float)
    def corr(col):
        if col not in tr: return None
        return core._safe_corr(v, tr[col].to_numpy(float))
    return {"champion_reconstructibility_r2": r2,
            "variance": float(np.nanvar(v)),
            "corr_raw_pitcher_success": corr("asof_pitcher_success_rate"),
            "corr_std_pitcher_success": corr("std_asof_pitcher_success_rate"),
            "corr_skill_pc_hat": corr("skill_pc_hat"),
            "corr_skill_hat": (core._safe_corr(v, gtr["skill_hat"].to_numpy(float))
                                if "skill_hat" in gtr else None),
            "n_pitcher_seasons": int(len(z))}


def main():
    # Keep the official inference schema available for the champion transform;
    # the nuisance model itself still selects only CAT+NUM and cannot see any
    # pitcher-history column.
    header = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    cols = list(dict.fromkeys(header + ["control_success"]))
    data = pd.read_csv(ROOT / "data/train.csv", usecols=cols)
    skills, meta = {}, []
    for cutoff in (2022, 2023, 2024):
        skills[cutoff], m = skill_lookup(data, cutoff); meta.append(m)
    novelty = novelty_and_sanity(data, skills)
    transfers = [
        transfer(data, 2022, 2023, "BND22_base", "BND22_cell", skills),
        transfer(data, 2023, 2024, "B1J6_base", "B1J6_cell", skills),
    ]
    result = {"contract": {"context_model": "SGD logistic", "k": K,
                            "ridge_alpha": ALPHA, "null_reps": 400,
                            "pitcher_inputs_present": False},
              "fit_meta": meta, "novelty": novelty, "transfer": transfers,
              "independence": {"pass": True,
                  "reason": "train-only frozen pitcher lookup; inference is a row-local pitcher_id join"}}
    out = ROOT / "out/ctx_adj_pitcher_skill_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

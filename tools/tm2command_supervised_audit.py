"""Pre-registered CPU gate for supervised Trackman -> future command.

The unit is pitcher-season.  A fixed 100-column Trackman distribution summary
from season t predicts the four official command outcomes in season t+1 with a
single standardized multivariate Ridge (alpha=100).  The resulting four scores
are then tested against champion residuals with a fit-on-source/apply-on-next
contract and a matched permutation null.

Nothing from evaluation rows is aggregated.  A deployable version would freeze
the 2019-2024 Trackman lookup and use row-local pitcher_id joins only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import failmode  # noqa: E402
import fpipe  # noqa: E402
import pbmf_transfer_audit as core  # noqa: E402


TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]
Q_NUM = TM_NUM[:7]
PTYPES = ["fastball", "breaking", "offspeed"]
OUTCOMES = ["success", "middle", "ball", "reverse"]
ALPHA = 100.0
SEEDS = (3, 4, 5, 6, 8, 13)


def _safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def build_tm100(cache: Path) -> pd.DataFrame:
    if cache.exists():
        z = pd.read_pickle(cache)
        if len([c for c in z if c not in ("pitcher_id", "season")]) != 100:
            raise RuntimeError("cached TM summary is not 100-dimensional")
        return z
    use = ["season", "pitcher_trackman_id", "pitch_type_group"] + TM_NUM
    tm = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=use)
    pmap = pd.read_csv(ROOT / "data/processed/pitcher_map2.csv")
    tm = tm.merge(pmap[["tm_id", "pitcher_id"]],
                  left_on="pitcher_trackman_id", right_on="tm_id", how="inner")
    tm["pitch_type_group"] = tm.pitch_type_group.fillna("other").astype(str)
    keys = ["pitcher_id", "season"]
    size = tm.groupby(keys).size().rename("tm_n")
    mix = (pd.crosstab([tm.pitcher_id, tm.season], tm.pitch_type_group)
           .reindex(columns=PTYPES + ["other"], fill_value=0))
    mix = mix.div(mix.sum(axis=1), axis=0)
    mix.columns = [f"usage_{c}" for c in mix.columns]
    entropy = -(mix.clip(lower=1e-12) * np.log(mix.clip(lower=1e-12))).sum(1)
    parts = [mix, entropy.rename("arsenal_entropy"),
             np.exp(entropy).rename("effective_modes")]
    for pt in PTYPES:
        sub = tm[tm.pitch_type_group == pt]
        q = sub.groupby(keys)[Q_NUM].quantile([.1, .5, .9]).unstack(-1)
        q.columns = [f"{pt}_{c}_q{int(v*100):02d}" for c, v in q.columns]
        for c in Q_NUM:
            q[f"{pt}_{c}_iqr"] = q[f"{pt}_{c}_q90"] - q[f"{pt}_{c}_q10"]
        parts.append(q)
    # Five whole-cloud covariance geometry summaries.
    cov_rows = []
    sep_rows = []
    for key, d in tm.groupby(keys, sort=False):
        x = d[TM_NUM].to_numpy(float)
        med = np.nanmedian(x, axis=0)
        x = np.where(np.isfinite(x), x, med)
        x = x[:, np.isfinite(x).all(0)]
        if x.shape[0] >= 3 and x.shape[1] >= 2:
            c = np.cov(x, rowvar=False)
            ev = np.clip(np.linalg.eigvalsh(c), 0, None)
            trace = float(ev.sum())
            logdet = float(np.log(ev + 1e-8).sum())
            top = float(ev[-1] / trace) if trace > 0 else 0.
        else:
            trace = logdet = top = np.nan
        cov_rows.append((*key,
                         d.rel_speed.cov(d.induced_vert_break),
                         d.horz_break.cov(d.rel_side), trace, logdet, top))
        means = d.groupby("pitch_type_group")[Q_NUM].mean()
        scale = d[Q_NUM].std().replace(0, np.nan)
        vals = []
        for a, b in (("fastball", "breaking"), ("fastball", "offspeed"),
                     ("breaking", "offspeed")):
            if a in means.index and b in means.index:
                vals.append(float(np.sqrt(np.nanmean(((means.loc[a]-means.loc[b])/scale) ** 2))))
            else:
                vals.append(np.nan)
        sep_rows.append((*key, *vals))
    cov = pd.DataFrame(cov_rows, columns=keys + ["cov_speed_ivb", "cov_hb_side",
                                                "cov_trace", "cov_logdet", "cov_top_share"])
    sep = pd.DataFrame(sep_rows, columns=keys + ["sep_fb_br", "sep_fb_off", "sep_br_off"])
    out = pd.concat(parts, axis=1).reset_index().merge(cov, on=keys).merge(sep, on=keys)
    out["tm_n"] = out.set_index(keys).index.map(size).astype(float)
    out["tm_log_n"] = np.log1p(out.tm_n)
    feat = [c for c in out if c not in keys]
    if len(feat) != 100:
        raise RuntimeError(f"TM summary dimension {len(feat)} != 100")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    return out


def command_profiles() -> pd.DataFrame:
    cols = ["row_id", "season", "pitcher_id", "control_success",
            "asof_pitcher_n", "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate"]
    d = pd.read_csv(ROOT / "data/train.csv", usecols=cols)
    labs = []
    for _, s in d.groupby("season", sort=True):
        labs.append(failmode._pitch_labels(s, ("middle", "ball", "reverse")))
    lab = pd.concat(labs).reindex(d.index)
    for c in ("middle", "ball", "reverse"):
        d[c] = lab[c]
    d = d.rename(columns={"control_success": "success"})
    g = d.groupby(["pitcher_id", "season"])
    out = g[OUTCOMES].mean().reset_index()
    out["command_n"] = g.size().to_numpy()
    # A pitcher-season with only an unrecoverable boundary pitch has no honest
    # failure-mode target.  Do not impute supervised outcomes.
    return out.dropna(subset=OUTCOMES).reset_index(drop=True)


def fit_predict_tm(train_pairs, target_x):
    feats = [c for c in train_pairs if c.startswith("tm_") or c.startswith("usage_")
             or c.startswith("arsenal_") or c.startswith("effective_")
             or c.startswith(tuple(f"{p}_" for p in PTYPES))
             or c.startswith("cov_") or c.startswith("sep_")]
    x = train_pairs[feats].to_numpy(float)
    xt = target_x[feats].to_numpy(float)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    x, xt = np.where(np.isfinite(x), x, med), np.where(np.isfinite(xt), xt, med)
    sc = StandardScaler().fit(x)
    m = Ridge(alpha=ALPHA).fit(sc.transform(x), train_pairs[OUTCOMES])
    return m.predict(sc.transform(xt))


def rolling_tm_outputs(tm, cmd):
    future = cmd.rename(columns={"season": "target_season"}).copy()
    future["season"] = future.target_season - 1
    pairs = tm.merge(future, on=["pitcher_id", "season"], how="inner")
    targets = cmd.rename(columns={"season": "target_season"})
    rows, quality = [], []
    for season in range(2021, 2025):
        tr = pairs[pairs.target_season < season]
        tx = (tm[tm.season == season-1]
              .merge(targets[targets.target_season == season][["pitcher_id", "target_season"]],
                     on="pitcher_id", how="inner"))
        truth = targets[targets.target_season == season]
        if len(tr) < 50 or tx.empty:
            continue
        pred = fit_predict_tm(tr, tx)
        z = tx[["pitcher_id"]].copy(); z["season"] = season
        for j, c in enumerate(OUTCOMES): z[f"tm_cmd_{c}"] = pred[:, j]
        z = z.merge(truth[["pitcher_id", "target_season"] + OUTCOMES],
                    left_on=["pitcher_id", "season"],
                    right_on=["pitcher_id", "target_season"], how="inner")
        q = {"season": season, "n": len(z)}
        for c in OUTCOMES:
            q[f"{c}_corr"] = _safe_corr(z[c], z[f"tm_cmd_{c}"])
            q[f"{c}_r2"] = float(r2_score(z[c], z[f"tm_cmd_{c}"]))
        quality.append(q)
        rows.append(z[["pitcher_id", "season"] + [f"tm_cmd_{c}" for c in OUTCOMES]])
    return pd.concat(rows, ignore_index=True), pd.DataFrame(quality)


def ensemble(tag, kind):
    return core.ensemble(tag, kind)[:2]


def core_predictions(tag_base, tag_cell, kind):
    b, y = ensemble(tag_base, kind); c, yc = ensemble(tag_cell, kind)
    if not np.array_equal(y, yc): raise RuntimeError("base/cell target mismatch")
    return core.post((1-core.W_CELL)*b + core.W_CELL*c), y


def ridge_transfer(src_x, tgt_x, resid):
    sc = StandardScaler().fit(src_x)
    m = Ridge(alpha=ALPHA).fit(sc.transform(src_x), resid)
    a, b = m.predict(sc.transform(src_x)), m.predict(sc.transform(tgt_x))
    return a-a.mean(), b-a.mean()


def matched_null(src_rows, tgt_rows, src_x, tgt_x, resid, target_resid, reps=400):
    pids = np.union1d(src_rows.pitcher_id.unique(), tgt_rows.pitcher_id.unique())
    src_tab = pd.DataFrame(src_x, columns=range(src_x.shape[1])).assign(
        pitcher_id=src_rows.pitcher_id.to_numpy()).groupby("pitcher_id").mean()
    tgt_tab = pd.DataFrame(tgt_x, columns=range(tgt_x.shape[1])).assign(
        pitcher_id=tgt_rows.pitcher_id.to_numpy()).groupby("pitcher_id").mean()
    vals = []
    for k in range(reps):
        rng = np.random.default_rng(20260816 + k)
        perm = pd.Series(rng.permutation(pids), index=pids)
        xs = src_rows.pitcher_id.map(perm).to_frame("p").join(src_tab, on="p").iloc[:, 1:].to_numpy(float)
        xt = tgt_rows.pitcher_id.map(perm).to_frame("p").join(tgt_tab, on="p").iloc[:, 1:].to_numpy(float)
        xs = np.nan_to_num(xs); xt = np.nan_to_num(xt)
        _, add = ridge_transfer(xs, xt, resid)
        vals.append(abs(_safe_corr(add, target_resid)))
    return np.asarray(vals)


def duplicate_r2(outputs, season, pack_path):
    pack = joblib.load(pack_path)
    cols = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    d = pd.read_csv(ROOT / "data/train.csv", usecols=cols)
    d = d[d.season == season].reset_index(drop=True)
    x = fpipe.transform(d.copy(), pack["fpipe"])
    nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
    agg = x.assign(pitcher_id=d.pitcher_id).groupby("pitcher_id")[nums].mean()
    score_cols = [f"tm_cmd_{c}" for c in OUTCOMES]
    z = (outputs[outputs.season == season].set_index("pitcher_id")[score_cols]
         .join(agg, how="inner"))
    kf = KFold(5, shuffle=True, random_state=20260816)
    ans = {}
    xx = np.nan_to_num(z[nums].to_numpy(float))
    for c in OUTCOMES:
        yy = z[f"tm_cmd_{c}"].to_numpy(float)
        yp = cross_val_predict(make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)),
                               xx, yy, cv=kf)
        ans[c] = float(r2_score(yy, yp))
    return len(z), ans


def transfer_one(outputs, source, target, base, cell):
    ps, ys = core_predictions(base, cell, "val")
    pt, yt = core_predictions(base, cell, "test")
    use = ["season", "game_month", "game_type", "pitcher_id"]
    d = pd.read_csv(ROOT / "data/train.csv", usecols=use)
    s = d[d.season == source].reset_index(drop=True)
    t = d[d.season == target].reset_index(drop=True)
    if len(s) != len(ys) or len(t) != len(yt): raise RuntimeError("prediction/frame mismatch")
    cols = [f"tm_cmd_{c}" for c in OUTCOMES]
    so = outputs[outputs.season == source].set_index("pitcher_id")
    to = outputs[outputs.season == target].set_index("pitcher_id")
    prior = outputs[outputs.season < source][cols].mean().to_numpy(float)
    xs = s[["pitcher_id"]].join(so[cols], on="pitcher_id")[cols].to_numpy(float)
    xt = t[["pitcher_id"]].join(to[cols], on="pitcher_id")[cols].to_numpy(float)
    xs = np.where(np.isfinite(xs), xs, prior); xt = np.where(np.isfinite(xt), xt, prior)
    rs, rt = ys-ps, yt-pt
    a, b = ridge_transfer(xs, xt, rs)
    rho = _safe_corr(b, rt)
    null = matched_null(s, t, xs, xt, rs, rt)
    seg = {}
    masks = {"R": t.game_type.eq("R"), "F": t.game_type.eq("F"),
             "early": t.game_month.le(6), "late": t.game_month.gt(6),
             "known": t.pitcher_id.isin(to.index), "cold": ~t.pitcher_id.isin(to.index)}
    for k, m in masks.items(): seg[k] = _safe_corr(b[m], rt[m])
    return {"source": source, "target": target, "rho_source": _safe_corr(a, rs),
            "rho_frozen": rho, "ceiling": 1e5*rho*rho,
            "null_median": float(np.nanmedian(null)), "null_p99": float(np.nanquantile(null, .99)),
            "null_percentile": float(np.nanmean(null <= abs(rho))*100), "segments": seg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="out/tm2command_supervised_audit.json")
    ap.add_argument("--cache", default="out/tm100_pitcher_season.pkl")
    args = ap.parse_args()
    tm = build_tm100(ROOT / args.cache)
    cmd = command_profiles()
    outputs, quality = rolling_tm_outputs(tm, cmd)
    outputs.to_pickle(ROOT / "out/tm2command_outputs.pkl")
    if not {2022, 2023, 2024}.issubset(set(outputs.season)):
        raise RuntimeError("required rolling outputs missing")
    dups = {}
    for year, tag in ((2023, "BND22"), (2024, "B1J6")):
        n, r2 = duplicate_r2(outputs, year, ROOT / f"model/cat_{tag}_base_s3.pkl")
        dups[str(year)] = {"n_pitchers": n, "cv_r2": r2}
    transfers = [
        transfer_one(outputs, 2022, 2023, "BND22_base", "BND22_cell"),
        transfer_one(outputs, 2023, 2024, "B1J6_base", "B1J6_cell"),
    ]
    result = {"contract": {"tm_features": 100, "outcomes": OUTCOMES,
                            "ridge_alpha": ALPHA, "null_reps": 400},
              "future_command_quality": quality.to_dict("records"),
              "duplicate_audit": dups, "residual_transfer": transfers}
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

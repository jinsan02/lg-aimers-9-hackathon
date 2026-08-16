"""Confirmatory CPU audit of the champion-orthogonal part of CTX_ADJ skill."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import ctx_adj_pitcher_skill_audit as prior  # noqa: E402
import fpipe  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402

ALPHA = 100.0
REPS = 400


def recon_pipe():
    return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))


def transform_pitchers(data, season, pack):
    raw = data[data.season == season].reset_index(drop=True)
    x = fpipe.transform(raw.drop(columns="control_success").copy(), pack["fpipe"])
    nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
    agg = x[nums].assign(pitcher_id=raw.pitcher_id.to_numpy()).groupby("pitcher_id").mean()
    return raw, x, agg, nums


def residualise_pair(xs, xt, zs, zt):
    """Source OOF residual and target residual from one source-frozen model."""
    groups = xs.index.to_numpy()
    cv = GroupKFold(5)
    oof = cross_val_predict(recon_pipe(), xs, zs, cv=cv, groups=groups)
    model = recon_pipe().fit(xs, zs)
    return zs - oof, zt - model.predict(xt)


def one_boundary(data, skills, source, target, base, cell, seed=20260816):
    pack = joblib.load(ROOT / f"model/cat_{base}_s3.pkl")
    sr, sxrow, sx, nums = transform_pitchers(data, source, pack)
    tr, txrow, tx, nums_t = transform_pitchers(data, target, pack)
    if nums != nums_t:
        raise RuntimeError("source/target feature order mismatch")
    common = (sx.index.intersection(tx.index)
              .intersection(skills[source].index).intersection(skills[target].index))
    sx, tx = sx.loc[common, nums], tx.loc[common, nums]
    med = sx.median().fillna(0)
    sx = sx.fillna(med); tx = tx.fillna(med)
    zs = skills[source].loc[common].to_numpy(float)
    zt = skills[target].loc[common].to_numpy(float)
    us, ut = residualise_pair(sx, tx, zs, zt)
    us_map, ut_map = pd.Series(us, index=common), pd.Series(ut, index=common)

    ps, ys = prior.champion(base, cell, "val")
    pt, yt = prior.champion(base, cell, "test")
    if len(sr) != len(ys) or len(tr) != len(yt):
        raise RuntimeError("prediction/frame mismatch")
    xsr = sr.pitcher_id.map(us_map).fillna(0).to_numpy(float)[:, None]
    xtr = tr.pitcher_id.map(ut_map).fillna(0).to_numpy(float)[:, None]
    rs, rt = ys - ps, yt - pt
    fit, add = core.ridge_transfer(xsr, xtr, rs)
    rho = core._safe_corr(add, rt)

    null = []
    rng = np.random.default_rng(seed)
    for _ in range(REPS):
        p = rng.permutation(len(common))
        nus, nut = residualise_pair(sx, tx, zs[p], zt[p])
        ns = sr.pitcher_id.map(pd.Series(nus, index=common)).fillna(0).to_numpy(float)[:, None]
        nt = tr.pitcher_id.map(pd.Series(nut, index=common)).fillna(0).to_numpy(float)[:, None]
        _, nadd = core.ridge_transfer(ns, nt, rs)
        null.append(abs(core._safe_corr(nadd, rt)))
    null = np.asarray(null)
    masks = {"R": tr.game_type.eq("R"), "F": tr.game_type.eq("F"),
             "early": tr.game_month.le(6), "late": tr.game_month.gt(6)}
    report = {"source": source, "target": target, "n_pitchers": int(len(common)),
              "rho": rho, "rho_source": core._safe_corr(fit, rs),
              "null_percentile": float(np.mean(null <= abs(rho)) * 100),
              "null_p95": float(np.quantile(null, .95)),
              "null_p99": float(np.quantile(null, .99)),
              "segments": {k: core._safe_corr(add[m], rt[m]) for k, m in masks.items()}}
    diag = pd.DataFrame({"pitcher_id": common, "eval_season": target,
                         "z": zt, "z_unique": ut}).set_index("pitcher_id")
    diag = diag.join(tx, how="left")
    return report, diag, nums


def sanity(data, diags, nums):
    z = pd.concat(diags).reset_index()
    def corr(a, b): return core._safe_corr(z[a].to_numpy(float), z[b].to_numpy(float))
    x = z[nums].to_numpy(float)
    med = np.nanmedian(x, axis=0); med = np.where(np.isfinite(med), med, 0.)
    x = np.where(np.isfinite(x), x, med)
    cv = GroupKFold(5)
    yp = cross_val_predict(recon_pipe(), x, z.z_unique, cv=cv, groups=z.pitcher_id)

    # ``skill_hat`` is a shipped 123-feature cell column, not part of the fixed
    # 121-feature reconstruction. Attach it only for the requested correlation.
    gpack = joblib.load(ROOT / "model/cat_GSKDEP_cell_s3.pkl")
    raw24 = data[data.season == 2024].reset_index(drop=True)
    gx = fpipe.transform(raw24.drop(columns="control_success").copy(), gpack["fpipe"])
    gskill = gx.assign(pitcher_id=raw24.pitcher_id).groupby("pitcher_id").skill_hat.mean()
    z["skill_hat_external"] = np.where(z.eval_season.eq(2024), z.pitcher_id.map(gskill), np.nan)
    return {"var_z": float(np.var(z.z)), "var_z_unique": float(np.var(z.z_unique)),
            "variance_fraction": float(np.var(z.z_unique) / np.var(z.z)),
            "corr_unique_z": corr("z_unique", "z"),
            "corr_unique_raw": corr("z_unique", "asof_pitcher_success_rate"),
            "corr_unique_std": corr("z_unique", "std_asof_pitcher_success_rate"),
            "corr_unique_skill_pc": corr("z_unique", "skill_pc_hat"),
            "corr_unique_skill_hat": core._safe_corr(
                z.z_unique.to_numpy(float), z.skill_hat_external.to_numpy(float)),
            "unique_oof_reconstructibility_r2": float(r2_score(z.z_unique, yp)),
            "n_pitcher_seasons": int(len(z))}


def main():
    header = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    data = pd.read_csv(ROOT / "data/train.csv", usecols=header + ["control_success"])
    skills = {s: prior.skill_lookup(data, s)[0] for s in (2022, 2023, 2024)}
    a, da, cols_a = one_boundary(data, skills, 2022, 2023,
                                 "BND22_base", "BND22_cell")
    b, db, cols_b = one_boundary(data, skills, 2023, 2024,
                                 "B1J6_base", "B1J6_cell", seed=20260817)
    cols = [c for c in cols_a if c in cols_b]
    result = {"contract": {"ridge_alpha": ALPHA, "null_reps": REPS,
                            "unit": "pitcher-season", "source_oof": True,
                            "target_frozen": True},
              "sanity": sanity(data, [da[["eval_season", "z", "z_unique"] + cols],
                                       db[["eval_season", "z", "z_unique"] + cols]], cols),
              "transfer": [a, b],
              "independence": {"pass": True,
                  "reason": "source-only reconstruction and train-frozen pitcher lookup"}}
    out = ROOT / "out/ctx_adj_unique_signal_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

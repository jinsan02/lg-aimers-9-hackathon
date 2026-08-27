"""One-off CPU audit for the strict temporal-consensus target.

Temporary research harness.  The only persistent product is the Agent-C report.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fpipe  # noqa: E402
import research_frames as rf  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402
from teacher import A as TeacherArgs  # noqa: E402
from train_gbdt2 import CAT_COLS  # noqa: E402

SEED = 20260828
ALPHA = 100.0
FOLDS = 5
REPS = 400


class A(TeacherArgs):
    # Match the current 121-feature base contract rather than the old teacher's
    # feat-k=50.  No current-pitch/future-row feature is introduced.
    feat_k = 200.0


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def strict_teacher(data: pd.DataFrame, cutoff: int, predict_years: list[int]):
    """Fit one fixed CPU teacher on seasons <= cutoff and predict later years."""
    header = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    features = [c for c in header if c not in ("row_id", "pitcher_id", "batter_id")]
    src = data[data.season.le(cutoff)].copy().reset_index(drop=True)
    fit_mask = pd.Series(True, index=src.index)
    src, new_cols, new_cats, art = fpipe.fit(src, A, fit_mask, None, verbose=False)
    features += [c for c in new_cols if c not in features]
    cats = [c for c in list(CAT_COLS) + list(new_cats) if c in features]
    cats = list(dict.fromkeys(cats))
    nums = [c for c in features if c not in cats]

    ni = SimpleImputer(strategy="median")
    ns = StandardScaler()
    xn = ns.fit_transform(ni.fit_transform(src[nums])).astype(np.float32)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    xc = enc.fit_transform(src[cats].astype(str))
    x = sparse.hstack([sparse.csr_matrix(xn), xc], format="csr")
    y = src.control_success.to_numpy(np.int8)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                          max_iter=30, tol=1e-4, average=True,
                          random_state=SEED)
    model.fit(x, y)
    ans = {}
    for year in predict_years:
        tgt = data[data.season.eq(year)].copy().reset_index(drop=True)
        tx = fpipe.transform(tgt.drop(columns="control_success").copy(), art)
        tn = ns.transform(ni.transform(tx[nums])).astype(np.float32)
        tc = enc.transform(tx[cats].astype(str))
        z = sparse.hstack([sparse.csr_matrix(tn), tc], format="csr")
        ans[year] = model.predict_proba(z)[:, 1].astype(np.float64)
    return ans, {"rows": len(src), "features": len(features),
                 "numeric": len(nums), "categorical": len(cats),
                 "matrix_columns": int(x.shape[1])}


def consensus_diagnostics(qs, y, p):
    mat = np.column_stack(qs)
    pair = [corr(mat[:, i], mat[:, j]) for i in range(3) for j in range(i+1, 3)]
    z = mat.mean(axis=1)
    return z, {
        "coverage": float(np.isfinite(mat).all(axis=1).mean()),
        "variance": float(np.var(z)),
        "sd": float(np.std(z)),
        "teacher_disagreement_mean_sd": float(np.mean(np.std(mat, axis=1))),
        "teacher_disagreement_p95_sd": float(np.quantile(np.std(mat, axis=1), .95)),
        "teacher_pair_corr_min": float(min(pair)),
        "teacher_pair_corr_max": float(max(pair)),
        "corr_y": corr(z, y),
        "corr_champion": corr(z, p),
        "teacher_cutoff_means": [float(v.mean()) for v in qs],
        "teacher_cutoff_sds": [float(v.std()) for v in qs],
    }


def prep_x(xs, xt):
    xs, xt = np.asarray(xs, np.float64), np.asarray(xt, np.float64)
    med = np.nanmedian(xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xs = np.where(np.isfinite(xs), xs, med)
    xt = np.where(np.isfinite(xt), xt, med)
    sc = StandardScaler().fit(xs)
    return sc.transform(xs), sc.transform(xt)


def reconstruct(xs, xt, zs, zt):
    xs, xt = prep_x(xs, xt)
    oof = np.empty(len(zs), float)
    cv = KFold(FOLDS, shuffle=True, random_state=SEED)
    for tr, va in cv.split(xs):
        oof[va] = Ridge(alpha=ALPHA).fit(xs[tr], zs[tr]).predict(xs[va])
    m = Ridge(alpha=ALPHA).fit(xs, zs)
    pred_t = m.predict(xt)
    return oof, pred_t


def permute_strata(values, rows, rng):
    out = np.asarray(values).copy()
    strata = (rows.game_type.astype(str) + "_" +
              np.where(rows.game_month.le(6), "early", "late"))
    for _, idx in strata.groupby(strata).groups.items():
        idx = np.asarray(list(idx), int)
        out[idx] = values[rng.permutation(idx)]
    return out


def boundary_audit(b, zs, zt):
    xs = b["sx"][b["numeric_features"]].to_numpy(float)
    xt = b["tx"][b["numeric_features"]].to_numpy(float)
    hs, ht = reconstruct(xs, xt, zs, zt)
    us, ut = zs-hs, zt-ht
    fit, add = core.ridge_transfer(us[:, None], ut[:, None], b["rs"])
    rho = corr(add, b["rt"])
    raw_fit, raw_add = core.ridge_transfer(zs[:, None], zt[:, None], b["rs"])
    raw_rho = corr(raw_add, b["rt"])

    null = np.empty(REPS, float)
    for k in range(REPS):
        rng = np.random.default_rng(SEED + 1000*k + b["target"])
        ns = permute_strata(us, b["s"], rng)
        nt = permute_strata(ut, b["t"], rng)
        _, na = core.ridge_transfer(ns[:, None], nt[:, None], b["rs"])
        null[k] = abs(corr(na, b["rt"]))
    masks = {"R": b["t"].game_type.eq("R").to_numpy(),
             "F": b["t"].game_type.eq("F").to_numpy(),
             "early": b["t"].game_month.le(6).to_numpy(),
             "late": b["t"].game_month.gt(6).to_numpy()}
    return {
        "source": b["source"], "target": b["target"],
        "champion_numeric_features": len(b["numeric_features"]),
        "source_reconstructibility_r2": float(r2_score(zs, hs)),
        "target_frozen_reconstructibility_r2": float(r2_score(zt, ht)),
        "source_unique_variance_share": float(np.var(us)/np.var(zs)),
        "target_unique_variance_share": float(np.var(ut)/np.var(zt)),
        "raw_source_rho": corr(raw_fit, b["rs"]),
        "raw_target_rho": raw_rho,
        "unique_source_rho": corr(fit, b["rs"]),
        "unique_target_rho": rho,
        "null_percentile": float(np.mean(null <= abs(rho))*100),
        "null_p95": float(np.quantile(null, .95)),
        "null_p99": float(np.quantile(null, .99)),
        "segments": {k: corr(add[m], b["rt"][m]) for k, m in masks.items()},
    }


def main():
    t0 = time.time()
    data = rf.load_train()
    # Five nested fits are enough for the two fixed three-teacher consensuses:
    # 2022 <- cutoffs 2019/20/21, 2023 <- 2020/21/22,
    # 2024 <- 2021/22/23.
    need = {2019: [2022], 2020: [2022, 2023],
            2021: [2022, 2023, 2024], 2022: [2023, 2024], 2023: [2024]}
    preds, provenance = {}, {}
    for cutoff, years in need.items():
        print(f"teacher <= {cutoff} -> {years}", flush=True)
        got, meta = strict_teacher(data, cutoff, years)
        provenance[str(cutoff)] = meta
        for year, p in got.items(): preds[(cutoff, year)] = p

    boundaries = [rf.boundary(data, *spec) for spec in rf.BOUNDARIES]
    champs = {}
    for b in boundaries:
        champs[b["source"]] = (b["ys"], b["ps"])
        champs[b["target"]] = (b["yt"], b["pt"])
    consensus, diag = {}, {}
    for year in (2022, 2023, 2024):
        cuts = [year-3, year-2, year-1]
        y, p = champs[year]
        z, d = consensus_diagnostics([preds[(c, year)] for c in cuts], y, p)
        consensus[year] = z
        d["cutoffs"] = cuts
        d["rows"] = len(z)
        diag[str(year)] = d

    transfers = [boundary_audit(b, consensus[b["source"]], consensus[b["target"]])
                 for b in boundaries]
    result = {
        "contract": {
            "teacher": "fixed SGD log-loss alpha=1e-5 max_iter=30 average=True",
            "teacher_inputs": "strict-cutoff 121-feature base pipeline; no privileged/current-pitch data",
            "consensus": "equal mean of cutoffs S-3,S-2,S-1",
            "student_soft_weight_if_licensed": 0.25,
            "reconstruction": "5-fold OOF/source-frozen Ridge alpha=100 on 112 champion numeric features",
            "matched_null": "400 independent within-(R/F x early/late) permutations of unique source/target directions; same scalar Ridge transfer",
        },
        "provenance": provenance,
        "season_diagnostics": diag,
        "transfers": transfers,
        "elapsed_sec": time.time()-t0,
    }
    print("RESULT_JSON")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

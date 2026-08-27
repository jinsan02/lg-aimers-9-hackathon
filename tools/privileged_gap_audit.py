"""CPU-only audit of a legally transferable current-pitch Trackman gap.

The current-pitch Trackman block is privileged training information.  It is
used only to define ``q_priv - q_legal`` on official historical train rows.
The deployable object under audit is a Ridge student of that gap from the
legal champion feature frame.  Nothing in this file reads evaluation data or
aggregates test rows.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import fpipe  # noqa: E402
import invalidated  # noqa: E402
import research_frames as rf  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402
from joint_pitch import CONTEXT, KEY  # noqa: E402


SEED = 20260827
FOLDS = 5
ALPHA = 100.0
TM_TYPES = ["fastball", "breaking", "offspeed", "other"]
TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]
BOUNDARIES = (
    (2022, 2023, "BND22_base", "BND22_cell", "BND22"),
    (2023, 2024, "B1J6_base", "B1J6_cell", "B1J6"),
)


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def bss(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    base = y.mean() * (1-y.mean())
    return float(1e5 * (1 - np.mean((p-y)**2) / base))


def exact_linked(data, cache: Path):
    """Strict 1-main-row : 1-Trackman-row linkage, retaining the fixed Z."""
    if cache.exists():
        out = pd.read_pickle(cache)
        expected = {"row_id", "pitch_type_group", "asof_pitcher_n", *TM_NUM}
        if expected.issubset(out.columns) and out.row_id.is_unique:
            return out
        raise RuntimeError("privileged linkage cache has the wrong schema")

    tm_cols = CONTEXT + ["pitcher_trackman_id", "batter_trackman_id",
                         "pitch_type_group", *TM_NUM]
    tm = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=tm_cols,
                     encoding="utf-8-sig")
    pm = pd.read_csv(ROOT / "data/processed/pitcher_map2.csv")
    bm = pd.read_csv(ROOT / "data/processed/batter_map2.csv")
    pmap = pm.drop_duplicates("tm_id").set_index("tm_id").pitcher_id
    bmap = bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id").batter_id
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(pmap)
    tm["batter_id"] = tm.batter_trackman_id.map(bmap)
    tm["top_bottom"] = tm.top_bottom.map({"Top": "T", "Bottom": "B"})
    for c in ("pitcher_hand", "batter_hand"):
        tm[c] = tm[c].map({"Left": 1, "Right": 2})
    tm = tm[tm.pitcher_id.notna() & tm.batter_id.notna()].copy()
    tm[["pitcher_id", "batter_id"]] = tm[["pitcher_id", "batter_id"]].astype(np.int64)
    tm["pitch_type_group"] = tm.pitch_type_group.where(
        tm.pitch_type_group.isin(TM_TYPES[:-1]), "other")

    main_keys = data[["row_id", *KEY]].copy()
    mc = main_keys.groupby(KEY, dropna=False).size().rename("main_n")
    agg = {"tm_n": ("pitch_type_group", "size"),
           "pitch_type_group": ("pitch_type_group", "first")}
    agg.update({c: (c, "first") for c in TM_NUM})
    tg = tm.groupby(KEY, dropna=False).agg(**agg)
    safe = (mc[mc.eq(1)].reset_index().drop(columns="main_n")
            .merge(tg[tg.tm_n.eq(1)].reset_index().drop(columns="tm_n"),
                   on=KEY, how="inner", validate="one_to_one"))
    # ``main`` contains duplicate keys that are absent from ``safe``.  Pandas'
    # one-to-one validator inspects those non-matching keys too, so validate the
    # retained rows explicitly after a many-to-one merge.
    out = data.merge(safe, on=KEY, how="inner", validate="many_to_one")
    if not out.row_id.is_unique:
        raise RuntimeError("exact linkage produced duplicate row_id")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(cache)
    return out


def z_matrix(rows, med=None, scaler=None):
    num = rows[TM_NUM].to_numpy(np.float32)
    if med is None:
        med = np.nanmedian(num, axis=0)
        med = np.where(np.isfinite(med), med, 0).astype(np.float32)
    num = np.where(np.isfinite(num), num, med)
    if scaler is None:
        scaler = StandardScaler().fit(num)
    num = scaler.transform(num).astype(np.float32)
    typ = rows.pitch_type_group.fillna("other").astype(str)
    onehot = np.column_stack([(typ == c).to_numpy(np.float32) for c in TM_TYPES])
    return np.column_stack([onehot, num]).astype(np.float32), med, scaler


def legal_matrix(rows, pack, med=None, scaler=None):
    raw = rows.drop(columns=["control_success", "pitch_type_group", *TM_NUM],
                    errors="ignore").copy()
    x = fpipe.transform(raw, pack["fpipe"])
    nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
    a = x[nums].to_numpy(np.float32)
    if med is None:
        med = np.nanmedian(a, axis=0)
        med = np.where(np.isfinite(med), med, 0).astype(np.float32)
    a = np.where(np.isfinite(a), a, med)
    if scaler is None:
        scaler = StandardScaler().fit(a)
    a = scaler.transform(a).astype(np.float32)
    return a, nums, med, scaler


def classifier():
    # Exact fixed CPU nuisance convention used by PB_MULTIYEAR_EXACT_POOL.
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                         max_iter=30, tol=1e-4, average=True,
                         random_state=SEED)


def q_predictions(xs, zs, ys, xt, zt):
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    ql = np.empty(len(ys), np.float32)
    qp = np.empty(len(ys), np.float32)
    xzs = np.column_stack([xs, zs]).astype(np.float32)
    xzt = np.column_stack([xt, zt]).astype(np.float32)
    for k, (tr, va) in enumerate(cv.split(xs, ys)):
        ml = classifier().fit(xs[tr], ys[tr])
        mp = classifier().fit(xzs[tr], ys[tr])
        ql[va] = ml.predict_proba(xs[va])[:, 1]
        qp[va] = mp.predict_proba(xzs[va])[:, 1]
        print(f"  teacher fold {k+1}/{FOLDS}", flush=True)
    ml = classifier().fit(xs, ys)
    mp = classifier().fit(xzs, ys)
    tl = ml.predict_proba(xt)[:, 1]
    tp = mp.predict_proba(xzt)[:, 1]
    return ql, qp, tl, tp


def ridge_gap(xs, xt, gap):
    cv = KFold(FOLDS, shuffle=True, random_state=SEED)
    oof = np.empty(len(gap), np.float32)
    for tr, va in cv.split(xs):
        oof[va] = Ridge(alpha=ALPHA).fit(xs[tr], gap[tr]).predict(xs[va])
    model = Ridge(alpha=ALPHA).fit(xs, gap)
    return oof, model.predict(xt).astype(np.float32), model


def core_residuals(data, source, target, base, cell):
    ps, ys = core.core_predictions(base, cell, "val")
    pt, yt = core.core_predictions(base, cell, "test")
    sr = data[data.season.eq(source)].reset_index(drop=True)
    tr = data[data.season.eq(target)].reset_index(drop=True)
    if len(sr) != len(ys) or len(tr) != len(yt):
        raise RuntimeError("champion residual frame length mismatch")
    if not np.array_equal(sr.control_success.to_numpy(), ys):
        raise RuntimeError("source champion target mismatch")
    if not np.array_equal(tr.control_success.to_numpy(), yt):
        raise RuntimeError("target champion target mismatch")
    return (pd.Series(ys-ps, index=sr.row_id.astype(str)),
            pd.Series(yt-pt, index=tr.row_id.astype(str)))


def gap_stats(g):
    q = np.quantile(g, [.01, .10, .50, .90, .99])
    return {"mean": float(np.mean(g)), "sd": float(np.std(g)),
            "variance": float(np.var(g)), "p01": float(q[0]),
            "p10": float(q[1]), "p50": float(q[2]),
            "p90": float(q[3]), "p99": float(q[4])}


def linkage_audit(data, linked):
    got = set(linked.row_id.astype(str))
    rows = []
    for season, d in data.groupby("season", sort=True):
        m = d.row_id.astype(str).isin(got).to_numpy()
        freq_p = d.pitcher_id.map(d.pitcher_id.value_counts()).to_numpy(float)
        freq_b = d.batter_id.map(d.batter_id.value_counts()).to_numpy(float)
        metrics = {
            "target": d.control_success.to_numpy(float),
            "R": d.game_type.eq("R").to_numpy(float),
            "F": d.game_type.eq("F").to_numpy(float),
            "balls": d.balls_before.to_numpy(float),
            "strikes": d.strikes_before.to_numpy(float),
            "outs": d.outs_before.to_numpy(float),
            "pitcher_asof_n": d.asof_pitcher_n.to_numpy(float),
            "batter_asof_n": d.asof_batter_n.to_numpy(float),
            "pitcher_frequency": freq_p, "batter_frequency": freq_b,
        }
        diffs = {}
        for name, v in metrics.items():
            a, b = v[m], v[~m]
            pooled = np.sqrt((np.nanvar(a)+np.nanvar(b))/2)
            diffs[name] = {"linked": float(np.nanmean(a)),
                           "unmatched": float(np.nanmean(b)),
                           "std_diff": float((np.nanmean(a)-np.nanmean(b))/pooled)
                           if pooled > 0 else 0.0}
        rows.append({"season": int(season), "rows": int(len(d)),
                     "linked": int(m.sum()), "coverage": float(m.mean()),
                     "differences": diffs})
    return rows


def one_boundary(data, linked, source, target, base, cell, tag, cache_dir):
    print(f"\n=== {source}->{target} ===", flush=True)
    pack = joblib.load(ROOT / f"model/cat_{tag}_base_s3.pkl")
    src = linked[linked.season.le(source)].reset_index(drop=True)
    tgt = linked[linked.season.eq(target)].reset_index(drop=True)
    if src.empty or tgt.empty:
        raise RuntimeError("linked source/target is empty")
    xs, names, xmed, xsc = legal_matrix(src, pack)
    xt, names2, _, _ = legal_matrix(tgt, pack, xmed, xsc)
    if names != names2:
        raise RuntimeError("legal feature order mismatch")
    zs, zmed, zsc = z_matrix(src)
    zt, _, _ = z_matrix(tgt, zmed, zsc)
    ys = src.control_success.to_numpy(np.int8)
    yt = tgt.control_success.to_numpy(np.int8)

    t0 = time.time()
    ql, qp, tl, tp = q_predictions(xs, zs, ys, xt, zt)
    g, gt = qp-ql, tp-tl
    gh, ght, gap_model = ridge_gap(xs, xt, g)
    elapsed = time.time()-t0

    rs_map, rt_map = core_residuals(data, source, target, base, cell)
    src_ids = src.row_id.astype(str)
    tgt_ids = tgt.row_id.astype(str)
    sm = src.season.eq(source).to_numpy()
    rs = rs_map.reindex(src_ids[sm]).to_numpy(float)
    rt = rt_map.reindex(tgt_ids).to_numpy(float)
    if not np.isfinite(rs).all() or not np.isfinite(rt).all():
        raise RuntimeError("row_id residual join left holes")

    source_corr = safe_corr(gh[sm], g[sm])
    target_corr = safe_corr(ght, gt)
    source_r2 = float(r2_score(g, gh))
    # The downstream direction is frozen from source-season champion residual.
    direction = Ridge(alpha=ALPHA).fit(gh[sm, None], rs)
    add = direction.predict(ght[:, None])
    rho = safe_corr(add, rt)
    masks = {"R": tgt.game_type.eq("R").to_numpy(),
             "F": tgt.game_type.eq("F").to_numpy(),
             "early": tgt.game_month.le(6).to_numpy(),
             "late": tgt.game_month.gt(6).to_numpy()}

    # Leakage-safe champion reconstruction.  Since h(X) is a linear Ridge on
    # this exact X, this intentionally tests the preregistered tautology rather
    # than silently calling the student novel.
    recon_oof, recon_t, _ = ridge_gap(xs, xt, gh)
    recon_r2 = float(r2_score(gh, recon_oof))
    unique_s, unique_t = gh-recon_oof, ght-recon_t
    retained = float(np.var(unique_t)/np.var(ght)) if np.var(ght) else 0.0
    udir = Ridge(alpha=ALPHA).fit(unique_s[sm, None], rs)
    uadd = udir.predict(unique_t[:, None])

    corrs = {"champion_prediction": safe_corr(ght, yt-rt),
             "skill_hat": safe_corr(ght, xt[:, names.index("skill_hat")])
             if "skill_hat" in names else None,
             "skill_pc_hat": safe_corr(ght, xt[:, names.index("skill_pc_hat")])
             if "skill_pc_hat" in names else None,
             "std_pitcher_success": safe_corr(
                 ght, xt[:, names.index("std_asof_pitcher_success_rate")])
             if "std_asof_pitcher_success_rate" in names else None,
             "raw_pitcher_success": safe_corr(
                 ght, xt[:, names.index("asof_pitcher_success_rate")])
             if "asof_pitcher_success_rate" in names else None}

    artifact = cache_dir / f"privileged_gap_{target}.npz"
    np.savez_compressed(artifact, source_row_id=src.row_id.to_numpy(),
                        target_row_id=tgt.row_id.to_numpy(), source_gap=g,
                        target_gap=gt, source_gap_hat=gh,
                        target_gap_hat=ght, source_unique=unique_s,
                        target_unique=unique_t)
    del xs, xt, zs, zt
    gc.collect()
    return {
        "source": source, "target": target, "source_rows": int(len(src)),
        "target_rows": int(len(tgt)), "legal_features": len(names),
        "privileged_features": len(TM_TYPES)+len(TM_NUM),
        "elapsed_seconds": elapsed,
        "stage1": {"legal_bss": bss(yt, tl), "privileged_bss": bss(yt, tp),
                   "delta": bss(yt, tp)-bss(yt, tl),
                   "rms_gap": float(np.sqrt(np.mean(gt**2))),
                   "pearson_arms": safe_corr(tl, tp),
                   "gap_variance": float(np.var(gt))},
        "stage2": {"source_gap": gap_stats(g), "target_gap": gap_stats(gt),
                   "segments_target_gap": {k: gap_stats(gt[m]) for k, m in masks.items()}},
        "stage3": {"source_oof_r2": source_r2,
                   "source_corr_gap_hat_gap": source_corr,
                   "target_corr_gap_hat_gap": target_corr,
                   "target_spearman": float(pd.Series(ght).corr(pd.Series(gt), method="spearman")),
                   "target_gap_hat_variance": float(np.var(ght))},
        "stage4": {"reconstructibility_cv_r2": recon_r2,
                   "correlations": corrs, "unique_variance_retained": retained},
        "stage5_pre_null": {"rho": rho,
                            "segments": {k: safe_corr(add[m], rt[m]) for k, m in masks.items()},
                            "unique_rho": safe_corr(uadd, rt),
                            "unique_segments": {k: safe_corr(uadd[m], rt[m]) for k, m in masks.items()}},
        "artifact": str(artifact.relative_to(ROOT)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="out/privileged_gap_audit.json")
    ap.add_argument("--link-cache", default="out/privileged_current_pitch.pkl")
    args = ap.parse_args()
    invalidated.guard(["BND22_base", "BND22_cell", "B1J6_base", "B1J6_cell"])
    data = rf.load_train()
    linked = exact_linked(data, ROOT / args.link_cache)
    if not linked.row_id.is_unique:
        raise RuntimeError("row_id integrity failure")
    result = {
        "contract": {"teacher": "SGDClassifier log_loss alpha=1e-5 max_iter=30 average",
                     "teacher_folds": FOLDS, "student": f"Ridge alpha={ALPHA:g}",
                     "legal_x": "boundary-pack champion numeric frame",
                     "privileged_z": ["pitch_type_group one-hot", *TM_NUM],
                     "null_reps": 400},
        "duplicate_audit": {"status": "NEW",
            "reason": "no prior axis completed q_legal/q_priv gap, legal student, frozen transfer and matched null"},
        "linkage": {"rows": int(len(linked)),
                    "coverage": float(len(linked)/len(data)),
                    "pitcher_coverage": float(linked.pitcher_id.nunique()/data.pitcher_id.nunique()),
                    "batter_coverage": float(linked.batter_id.nunique()/data.batter_id.nunique()),
                    "selection": linkage_audit(data, linked)},
        "boundaries": [],
        "independence": {"pass": True,
            "reason": "official train-only exact linkage; frozen source fit; row_id joins; no evaluation frame"},
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    for spec in BOUNDARIES:
        result["boundaries"].append(one_boundary(data, linked, *spec, out.parent))
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

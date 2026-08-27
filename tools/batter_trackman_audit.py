"""CPU-only audit of a frozen historical batter Trackman profile.

The profile contains pitches historically seen by a batter.  For a row in
season S every lookup is built strictly from Trackman seasons < S.  The fixed
20-dimensional raw profile is compressed to PCA-8 with source-fitted scaling.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import entity_unique_audit as eu  # noqa: E402
import invalidated  # noqa: E402
import research_frames as rf  # noqa: E402


FAMILIES = ["fastball", "breaking", "offspeed", "other"]
TM_NUM = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
          "extension", "rel_height", "rel_side", "zone_speed"]
K = 80.0
PCA_DIM = 8
REPS = 400
SEED = 20260827


def safe_corr(a, b):
    return eu.safe_corr(a, b)


def load_mapped_trackman():
    use = ["season", "batter_trackman_id", "pitch_type_group", *TM_NUM]
    tm = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=use,
                     encoding="utf-8-sig")
    bm = pd.read_csv(ROOT / "data/processed/batter_map2.csv")
    if bm.batter_id.duplicated().any() or bm.tm_batter_id.duplicated().any():
        raise RuntimeError("accepted batter map is not one-to-one")
    mapper = bm.set_index("tm_batter_id").batter_id
    tm["batter_id"] = tm.batter_trackman_id.map(mapper)
    tm["pitch_type_group"] = tm.pitch_type_group.where(
        tm.pitch_type_group.isin(FAMILIES[:-1]), "other")
    return tm, bm


def profile(history):
    """Fixed k=80 empirical-distribution profile (4 mix + 8 mean + 8 sd)."""
    if history.empty:
        raise RuntimeError("empty Trackman history")
    h = history[history.batter_id.notna()].copy()
    h["batter_id"] = h.batter_id.astype(np.int64)
    idx = pd.Index(sorted(h.batter_id.unique()), name="batter_id")
    n = h.groupby("batter_id").size().reindex(idx).astype(float)

    ct = pd.crosstab(h.batter_id, h.pitch_type_group).reindex(
        index=idx, columns=FAMILIES, fill_value=0).astype(float)
    prior_mix = h.pitch_type_group.value_counts(normalize=True).reindex(
        FAMILIES, fill_value=0).to_numpy(float)
    mix = (ct.to_numpy(float) + K*prior_mix) / (n.to_numpy()[:, None] + K)
    if not np.allclose(mix.sum(1), 1, atol=1e-10):
        raise RuntimeError("shrunk pitch mix does not sum to one")

    means = h.groupby("batter_id")[TM_NUM].mean().reindex(idx)
    counts = h.groupby("batter_id")[TM_NUM].count().reindex(idx).astype(float)
    squares = h.assign(**{f"__sq_{c}": h[c].astype(float)**2 for c in TM_NUM})
    m2 = squares.groupby("batter_id")[[f"__sq_{c}" for c in TM_NUM]].mean().reindex(idx)
    gmean = h[TM_NUM].mean().to_numpy(float)
    gm2 = (h[TM_NUM].astype(float)**2).mean().to_numpy(float)
    cnt = counts.to_numpy(float)
    mu = (np.nan_to_num(means.to_numpy(float))*cnt + K*gmean) / (cnt+K)
    sec = (np.nan_to_num(m2.to_numpy(float))*cnt + K*gm2) / (cnt+K)
    sd = np.sqrt(np.maximum(sec-mu**2, 0))
    vals = np.column_stack([mix, mu, sd])
    names = ([f"mix_{c}" for c in FAMILIES] +
             [f"mean_{c}" for c in TM_NUM] +
             [f"sd_{c}" for c in TM_NUM])
    out = pd.DataFrame(vals, index=idx, columns=names)
    out["support"] = n
    return out


def frozen_pca(source_raw, target_raw):
    cols = [c for c in source_raw if c != "support"]
    sc = StandardScaler().fit(source_raw[cols])
    pca = PCA(PCA_DIM, random_state=SEED).fit(sc.transform(source_raw[cols]))
    names = [f"batter_tm_pc{i}" for i in range(PCA_DIM)]
    s = pd.DataFrame(pca.transform(sc.transform(source_raw[cols])),
                     index=source_raw.index, columns=names)
    t = pd.DataFrame(pca.transform(sc.transform(target_raw[cols])),
                     index=target_raw.index, columns=names)
    return s, t, sc, pca


def persistence(annual):
    out = {}
    for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
        pa, pb, _, _ = frozen_pca(annual[a], annual[b])
        ids = pa.index.intersection(pb.index)
        xa, xb = pa.loc[ids].to_numpy(float), pb.loc[ids].to_numpy(float)
        cca = CCA(n_components=4, max_iter=2000).fit(xa, xb)
        ca, cb = cca.transform(xa, xb)
        cans = [safe_corr(ca[:, j], cb[:, j]) for j in range(4)]
        pear = [safe_corr(xa[:, j], xb[:, j]) for j in range(PCA_DIM)]
        spear = [float(pd.Series(xa[:, j]).corr(pd.Series(xb[:, j]), method="spearman"))
                 for j in range(PCA_DIM)]
        out[f"{a}->{b}"] = {"common_batters": int(len(ids)),
                            "canonical_correlations": cans,
                            "component_pearson": pear,
                            "component_spearman": spear,
                            "median_component_pearson": float(np.median(pear)),
                            "median_component_spearman": float(np.median(spear))}
    return out


def map_profiles(rows, ids, values):
    tab = pd.DataFrame(values, index=pd.Index(ids, name="batter_id"))
    x = rows[["batter_id"]].join(tab, on="batter_id").iloc[:, 1:].to_numpy(float)
    return np.nan_to_num(x)


def null_summary(null, rho):
    return {"repetitions": int(len(null)),
            "median_abs_rho": float(np.median(null)),
            "p90": float(np.quantile(null, .90)),
            "p95": float(np.quantile(null, .95)),
            "p99": float(np.quantile(null, .99)),
            "percentile": float(np.mean(null <= abs(rho))*100)}


def segments(rows, signal, resid):
    masks = {"R": rows.game_type.eq("R").to_numpy(),
             "F": rows.game_type.eq("F").to_numpy(),
             "early": rows.game_month.le(6).to_numpy(),
             "late": rows.game_month.gt(6).to_numpy()}
    return {k: safe_corr(signal[m], resid[m]) for k, m in masks.items()}


def raw_transfer(b, sp, tp, cp_s, cp_t):
    ids, zs, zt, _, _ = eu._aligned_profiles(sp, tp, cp_s, cp_t)
    xs = map_profiles(b["s"], ids, zs)
    xt = map_profiles(b["t"], ids, zt)
    fit, add = eu._direction(xs, xt, b["rs"])
    rho = safe_corr(add, b["rt"])
    null = np.empty(REPS, float)
    for k in range(REPS):
        p = np.random.default_rng(SEED+k).permutation(len(ids))
        ns = map_profiles(b["s"], ids, zs[p])
        nt = map_profiles(b["t"], ids, zt[p])
        _, na = eu._direction(ns, nt, b["rs"])
        null[k] = abs(safe_corr(na, b["rt"]))
    return {"rho_source": safe_corr(fit, b["rs"]), "rho": rho,
            "null": null_summary(null, rho),
            "segments": segments(b["t"], add, b["rt"])}


def unique_transfer(b, sp, tp, cp_s, cp_t):
    ids, zs, zt, cxs, cxt = eu._aligned_profiles(sp, tp, cp_s, cp_t)
    h_oof, h_target = eu.reconstruction_operators(cxs, cxt)
    us, ut, recon = eu.apply_reconstruction(h_oof, h_target, zs, zt)
    xs = map_profiles(b["s"], ids, us)
    xt = map_profiles(b["t"], ids, ut)
    fit, add = eu._direction(xs, xt, b["rs"])
    rho = safe_corr(add, b["rt"])
    null = np.empty(REPS, float)
    for k in range(REPS):
        p = np.random.default_rng(SEED+k).permutation(len(ids))
        nus, nut, _ = eu.apply_reconstruction(h_oof, h_target, zs[p], zt[p])
        ns = map_profiles(b["s"], ids, nus)
        nt = map_profiles(b["t"], ids, nut)
        _, na = eu._direction(ns, nt, b["rs"])
        null[k] = abs(safe_corr(na, b["rt"]))
    comp_r2 = [float(r2_score(zs[:, j], recon[:, j])) for j in range(PCA_DIM)]
    second = h_oof @ us
    var = float(np.var(zs)); uvar = float(np.var(us))
    return {"common_batters": int(len(ids)),
            "reconstructibility_multivariate_r2": float(r2_score(zs, recon)),
            "component_r2": comp_r2, "max_component_r2": float(max(comp_r2)),
            "median_component_r2": float(np.median(comp_r2)),
            "total_variance": var, "unique_variance": uvar,
            "unique_variance_retained": uvar/var if var else 0.0,
            "second_oof_reconstruction_r2": float(r2_score(us, second)),
            "rho_source": safe_corr(fit, b["rs"]), "rho": rho,
            "null": null_summary(null, rho),
            "segments": segments(b["t"], add, b["rt"]),
            "profiles": (ids, zs, zt, us, ut)}


def feature_correlations(b, ids, zt):
    prof = pd.DataFrame(zt, index=pd.Index(ids, name="batter_id"))
    tx = b["tx"].copy().assign(batter_id=b["t"].batter_id.to_numpy())
    candidates = ["asof_batter_success_rate", "asof_batter_middle_rate",
                  "std_asof_batter_success_rate", "std_asof_batter_middle_rate",
                  "asof_batter_n", "std_batter_n", "te_batter_ratio"]
    agg = tx.groupby("batter_id").mean(numeric_only=True)
    ans = {}
    for c in candidates:
        if c not in agg:
            ans[c] = None
            continue
        common = prof.index.intersection(agg.index)
        vals = [safe_corr(prof.loc[common, j], agg.loc[common, c])
                for j in prof.columns]
        ans[c] = {"max_abs": float(np.nanmax(np.abs(vals))),
                  "component_correlations": vals}
    # Hand is row-local categorical; use the batter-level modal numeric code.
    hand = b["t"].groupby("batter_id").batter_hand.first()
    common = prof.index.intersection(hand.index)
    vals = [safe_corr(prof.loc[common, j], hand.loc[common]) for j in prof.columns]
    ans["batter_hand"] = {"max_abs": float(np.nanmax(np.abs(vals))),
                          "component_correlations": vals}
    return ans


def support_summary(rows, raw):
    support = raw.support.to_numpy(float)
    q = np.quantile(support, [.25, .50, .75])
    return {"batters": int(len(raw)), "p25": float(q[0]),
            "p50": float(q[1]), "p75": float(q[2]),
            "fraction_lt50": float(np.mean(support < 50)),
            "fraction_lt100": float(np.mean(support < 100)),
            "fraction_lt300": float(np.mean(support < 300)),
            "target_row_coverage": float(rows.batter_id.isin(raw.index).mean())}


def main():
    invalidated.guard(["BND22_base", "BND22_cell", "B1J6_base", "B1J6_cell"])
    data = rf.load_train()
    tm, bm = load_mapped_trackman()
    mapped = tm[tm.batter_id.notna()].copy()
    annual = {y: profile(mapped[mapped.season.eq(y)]) for y in range(2021, 2025)}
    result = {
        "contract": {"raw_dimensions": 20, "pca_dimensions": PCA_DIM,
                     "shrink_k": K, "ridge_alpha": eu.ALPHA,
                     "null_repetitions": REPS, "families": FAMILIES,
                     "numeric_columns": TM_NUM},
        "duplicate_audit": {"status": "PARTIAL_OVERLAP_NOT_DUPLICATE",
            "closest": "BATTER_ARSENAL_FAMILIARITY scalar JSD",
            "difference": "full 20-d batter-only exposure distribution, PCA-8 persistence, champion-unique frozen transfer"},
        "mapping": {
            "train_batter_ids": int(data.batter_id.nunique()),
            "trackman_batter_ids": int(tm.batter_trackman_id.nunique()),
            "mapped_ids": int(len(bm)),
            "unmapped_train_ids": int(data.batter_id.nunique()-len(bm)),
            "ambiguous_accepted": int(bm.batter_id.duplicated().sum()+bm.tm_batter_id.duplicated().sum()),
            "train_row_coverage": float(data.batter_id.isin(bm.batter_id).mean()),
            "trackman_row_coverage": float(tm.batter_trackman_id.isin(bm.tm_batter_id).mean()),
            "settled_755_reproduced": bool(len(bm) == 755),
            "reproducibility_note": ("current committed src/link_batters.py deterministically "
                                     "produces 699 accepted pairs; the older SETTLED claim of "
                                     "755 has no matching committed code or artifact"),
            "train_season_coverage": {str(int(k)): float(v) for k, v in
                data.assign(_hit=data.batter_id.isin(bm.batter_id)).groupby("season")._hit.mean().items()},
            "trackman_season_coverage": {str(int(k)): float(v) for k, v in
                tm.assign(_hit=tm.batter_trackman_id.isin(bm.tm_batter_id)).groupby("season")._hit.mean().items()},
            "confidence_rule": "src/link_batters.py: unique tm id, margin>=1.5, ratio>=.50, score>=20"},
        "persistence": persistence(annual), "boundaries": [],
        "independence": {"pass": True,
            "reason": "season-S row joins only batter_id to a Trackman< S frozen lookup; no target/test aggregation"},
    }
    raw_cache = {}
    for spec in rf.BOUNDARIES:
        b = rf.boundary(data, *spec)
        sr = profile(mapped[mapped.season.lt(b["source"])])
        tr = profile(mapped[mapped.season.lt(b["target"])])
        sp, tp, _, pca = frozen_pca(sr, tr)
        cp_s, cp_t = rf.champion_profiles(b, "batter_id")
        raw = raw_transfer(b, sp, tp, cp_s, cp_t)
        uniq = unique_transfer(b, sp, tp, cp_s, cp_t)
        ids, zs, zt, us, ut = uniq.pop("profiles")
        corr = feature_correlations(b, ids, zt)
        miss = {c: float(sr[c].isna().mean()) for c in sr if c != "support"}
        result["boundaries"].append({
            "source": b["source"], "target": b["target"],
            "source_history_seasons": sorted(map(int, mapped.loc[mapped.season.lt(b["source"]), "season"].unique())),
            "target_history_seasons": sorted(map(int, mapped.loc[mapped.season.lt(b["target"]), "season"].unique())),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "dimension_summary": {c: {
                "source_mean": float(sr[c].mean()), "source_sd": float(sr[c].std()),
                "target_mean": float(tr[c].mean()), "target_sd": float(tr[c].std())}
                for c in sr if c != "support"},
            "profile_missing_rate": miss,
            "support": support_summary(b["t"], tr),
            "raw_transfer": raw, "unique": uniq,
            "correlations": corr})
        raw_cache[b["target"]] = (ids, zs, zt, us, ut)
    out = ROOT / "out/batter_trackman_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

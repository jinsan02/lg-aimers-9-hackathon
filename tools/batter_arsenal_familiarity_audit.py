"""CPU-only unique-signal gate for batter arsenal familiarity.

The only candidate is Jensen-Shannon divergence between a frozen, k=80 shrunk
batter historical pitch-family exposure and the current row's official pitcher
ASOF mix.  Trackman linkage is exact and train-only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import entity_unique_audit as unique  # noqa: E402
import research_frames as rf  # noqa: E402

FAMILIES = ["fastball", "breaking", "offspeed"]
MIX_COLS = [f"asof_pitcher_{x}_rate" for x in FAMILIES]
K = 80.0


def normalise(x):
    x = np.clip(np.asarray(x, float), 1e-9, None)
    return x / x.sum(axis=1, keepdims=True)


def jsd(p, q):
    p, q = normalise(p), normalise(q)
    m = .5 * (p + q)
    return .5 * np.sum(p*np.log(p/m), axis=1) + .5 * np.sum(q*np.log(q/m), axis=1)


def exposure_tables(data):
    linked = pd.read_pickle(ROOT / "out/tm_type_command_linked.pkl")
    linked = linked.merge(data[["row_id", "batter_id"]], on="row_id", how="left")
    linked = linked[linked.pitch_type_group.isin(FAMILIES) & linked.batter_id.notna()]
    out, meta = {}, []
    for cutoff in (2022, 2023, 2024):
        h = linked[linked.season < cutoff]
        ct = pd.crosstab(h.batter_id, h.pitch_type_group).reindex(columns=FAMILIES,
                                                                  fill_value=0)
        glob = ct.sum(0).to_numpy(float); glob /= glob.sum()
        n = ct.sum(1).to_numpy(float)
        p = (ct.to_numpy(float) + K*glob) / (n[:, None] + K)
        out[cutoff] = pd.DataFrame(p, index=ct.index, columns=FAMILIES)
        meta.append({"cutoff": cutoff, "fit_seasons": sorted(map(int, h.season.unique())),
                     "linked_rows": int(len(h)), "batters": int(len(ct)),
                     "global_mix": dict(zip(FAMILIES, map(float, glob)))})
    return out, meta


def map_matrix(rows, ids, values):
    tab = pd.DataFrame(values, index=pd.Index(ids, name="batter_id"), columns=FAMILIES)
    x = rows[["batter_id"]].join(tab, on="batter_id")[FAMILIES].to_numpy(float)
    prior = np.nanmean(values, axis=0)
    return np.where(np.isfinite(x), x, prior)


def direction(xs, xt, rs):
    return unique._direction(xs[:, None], xt[:, None], rs)


def one_boundary(b, exposures):
    cp_s, cp_t = rf.champion_profiles(b, "batter_id")
    sp, tp = exposures[b["source"]], exposures[b["target"]]
    ids, zs, zt, cxs, cxt = unique._aligned_profiles(sp, tp, cp_s, cp_t)
    h_oof,h_target=unique.reconstruction_operators(cxs,cxt)
    us,ut,recon_s=unique.apply_reconstruction(h_oof,h_target,zs,zt)
    # Unique familiarity is the change in JSD caused by the champion-unexplained
    # part of the batter exposure, not JSD of a signed residual vector.
    full_s = map_matrix(b["s"], ids, zs)
    full_t = map_matrix(b["t"], ids, zt)
    pred_s = map_matrix(b["s"], ids, zs-us)
    pred_t = map_matrix(b["t"], ids, zt-ut)
    mix_s = b["s"][MIX_COLS].to_numpy(float)
    mix_t = b["t"][MIX_COLS].to_numpy(float)
    prior_mix = np.nanmean(mix_s, axis=0)
    mix_s = np.where(np.isfinite(mix_s), mix_s, prior_mix)
    mix_t = np.where(np.isfinite(mix_t), mix_t, prior_mix)
    sig_s = jsd(full_s, mix_s) - jsd(pred_s, mix_s)
    sig_t = jsd(full_t, mix_t) - jsd(pred_t, mix_t)
    fit, add = direction(sig_s, sig_t, b["rs"])
    rho = unique.safe_corr(add, b["rt"])
    null = np.empty(unique.REPS)
    for k in range(unique.REPS):
        p = np.random.default_rng(20260827+k).permutation(len(ids))
        nus,nut,_=unique.apply_reconstruction(h_oof,h_target,zs[p],zt[p])
        nfs, nft = map_matrix(b["s"], ids, zs[p]), map_matrix(b["t"], ids, zt[p])
        nps = map_matrix(b["s"], ids, zs[p]-nus)
        npt = map_matrix(b["t"], ids, zt[p]-nut)
        ns = jsd(nfs, mix_s)-jsd(nps, mix_s)
        nt = jsd(nft, mix_t)-jsd(npt, mix_t)
        _, na = direction(ns, nt, b["rs"])
        null[k] = abs(unique.safe_corr(na, b["rt"]))
    masks = {"R": b["t"].game_type.eq("R"), "F": b["t"].game_type.eq("F"),
             "early": b["t"].game_month.le(6), "late": b["t"].game_month.gt(6)}
    return {"source": b["source"], "target": b["target"],
            "n_common_batters": int(len(ids)),
            "exposure_reconstructibility_r2": float(1-np.sum(us*us)/np.sum((zs-zs.mean(0))**2)),
            "unique_signal_variance": float(np.var(sig_s)),
            "coverage_source": float(b["s"].batter_id.isin(sp.index).mean()),
            "coverage_target": float(b["t"].batter_id.isin(tp.index).mean()),
            "rho_source": unique.safe_corr(fit, b["rs"]), "rho": rho,
            "null_percentile": float(np.mean(null <= abs(rho))*100),
            "null_p95": float(np.quantile(null,.95)),
            "null_p99": float(np.quantile(null,.99)),
            "segments": {k: unique.safe_corr(add[m], b["rt"][m]) for k,m in masks.items()}}


def main():
    data = rf.load_train()
    exposures, meta = exposure_tables(data)
    transfers = [one_boundary(rf.boundary(data, *spec), exposures)
                 for spec in rf.BOUNDARIES]
    result = {"contract": {"families": FAMILIES, "k": K,
                            "feature": "Jensen-Shannon divergence", "null_reps": 400},
              "fit_meta": meta, "transfers": transfers,
              "independence": {"pass": True,
                 "reason": "frozen historical batter exposure plus current row official ASOF mix"}}
    out = ROOT / "out/batter_arsenal_familiarity_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

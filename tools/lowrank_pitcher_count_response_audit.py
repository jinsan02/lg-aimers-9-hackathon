"""CPU-only rank-2 pitcher x count-response unique-signal gate."""

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

K = 80.0
RANK = 2
CELLS = [(b, s) for b in range(4) for s in range(3)]


def matrices(data, cutoff, global_rate):
    h = data[data.season < cutoff]
    g = h.groupby(["pitcher_id", "balls_before", "strikes_before"]).control_success.agg(["sum","size"])
    pitchers = np.sort(h.pitcher_id.unique())
    rates = np.empty((len(pitchers), len(CELLS)), float)
    counts = np.zeros_like(rates)
    for j, cell in enumerate(CELLS):
        try: q = g.xs(cell, level=["balls_before","strikes_before"])
        except KeyError: q = pd.DataFrame(columns=["sum","size"])
        su = pd.Series(q.get("sum", []), index=q.index if len(q) else None)
        nn = pd.Series(q.get("size", []), index=q.index if len(q) else None)
        s = pd.Series(pitchers).map(su).fillna(0).to_numpy(float)
        n = pd.Series(pitchers).map(nn).fillna(0).to_numpy(float)
        rates[:,j] = (s + K*global_rate[j])/(n+K)
        counts[:,j] = n
    return pitchers, rates, counts


def fit_basis(data, source):
    h = data[data.season < source]
    gg = h.groupby(["balls_before","strikes_before"]).control_success.mean()
    glob = np.array([gg.get(c, h.control_success.mean()) for c in CELLS], float)
    ids, rates, counts = matrices(data, source, glob)
    level = rates.mean(1, keepdims=True)
    response = rates-level
    centre = response.mean(0)
    _, _, vt = np.linalg.svd(response-centre, full_matrices=False)
    basis = vt[:RANK].copy()
    for j in range(RANK):
        k = np.argmax(np.abs(basis[j]))
        if basis[j,k] < 0: basis[j] *= -1
    return glob, centre, basis, ids, rates, counts


def projected_profile(data, cutoff, glob, centre, basis):
    ids, rates, counts = matrices(data, cutoff, glob)
    level = rates.mean(1, keepdims=True)
    score = (rates-level-centre) @ basis.T
    recon = level + centre + score @ basis
    cols = [f"count_{b}{s}" for b,s in CELLS] + ["pc1_amplitude"]
    return pd.DataFrame(np.column_stack([recon, score[:,0]]), index=ids, columns=cols), counts


def map_unique(rows, ids, u):
    tab = pd.DataFrame(u, index=pd.Index(ids,name="pitcher_id"))
    allv = rows[["pitcher_id"]].join(tab,on="pitcher_id").iloc[:,1:].to_numpy(float)
    cell = (rows.balls_before.to_numpy(int)*3 + rows.strikes_before.to_numpy(int))
    return np.column_stack([allv[np.arange(len(rows)),cell], allv[:,-1]])


def one_boundary(data, spec):
    b = rf.boundary(data,*spec)
    glob, centre, basis, _, _, _ = fit_basis(data,b["source"])
    sp, scount = projected_profile(data,b["source"],glob,centre,basis)
    tp, tcount = projected_profile(data,b["target"],glob,centre,basis)
    cp_s, cp_t = rf.champion_profiles(b,"pitcher_id")
    ids,zs,zt,cxs,cxt = unique._aligned_profiles(sp,tp,cp_s,cp_t)
    h_oof,h_target=unique.reconstruction_operators(cxs,cxt)
    us,ut,recon=unique.apply_reconstruction(h_oof,h_target,zs,zt)
    xs=np.nan_to_num(map_unique(b["s"],ids,us)); xt=np.nan_to_num(map_unique(b["t"],ids,ut))
    fit,add=unique._direction(xs,xt,b["rs"]); rho=unique.safe_corr(add,b["rt"])
    null=np.empty(unique.REPS)
    for k in range(unique.REPS):
        p=np.random.default_rng(20260827+k).permutation(len(ids))
        nus,nut,_=unique.apply_reconstruction(h_oof,h_target,zs[p],zt[p])
        nxs=np.nan_to_num(map_unique(b["s"],ids,nus)); nxt=np.nan_to_num(map_unique(b["t"],ids,nut))
        _,na=unique._direction(nxs,nxt,b["rs"]); null[k]=abs(unique.safe_corr(na,b["rt"]))
    masks={"R":b["t"].game_type.eq("R"),"F":b["t"].game_type.eq("F"),
           "early":b["t"].game_month.le(6),"late":b["t"].game_month.gt(6)}
    return {"source":b["source"],"target":b["target"],"n_common_pitchers":int(len(ids)),
            "reconstructibility_r2":float(1-np.sum(us*us)/np.sum((zs-zs.mean(0))**2)),
            "unique_variance_share":float(np.var(us)/np.var(zs)),
            "median_cell_support_source_history":float(np.median(scount)),
            "median_cell_support_target_history":float(np.median(tcount)),
            "basis":basis.tolist(),"rho_source":unique.safe_corr(fit,b["rs"]),"rho":rho,
            "null_percentile":float(np.mean(null<=abs(rho))*100),
            "null_p95":float(np.quantile(null,.95)),"null_p99":float(np.quantile(null,.99)),
            "segments":{k:unique.safe_corr(add[m],b["rt"][m]) for k,m in masks.items()}}


def main():
    data=rf.load_train()
    transfers=[one_boundary(data,s) for s in rf.BOUNDARIES]
    result={"contract":{"rank":RANK,"k":K,"cells":CELLS,"null_reps":unique.REPS,
                        "outputs":["current_count_reconstruction","pc1_amplitude"]},
            "transfers":transfers,
            "independence":{"pass":True,"reason":"frozen train-only pitcher profile and row-local count"}}
    out=ROOT/"out/lowrank_pitcher_count_response_audit.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

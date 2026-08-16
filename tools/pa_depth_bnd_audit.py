"""Re-evaluate the fixed prior-PA-depth arm on clean BND22/B1J6 arrays.

The historical arm called `batter_full_current_gate` actually gated the
historical full-count propensity by current *deep count* (balls+strikes >= 4),
not strictly 3-2.  That naming defect is preserved here so this is a replay,
not a silently changed candidate.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import pbmf_transfer_audit as core  # noqa:E402


def raw_bss(y,p):
    r=float(np.mean(y)); return 1e5*(1-float(np.mean((np.asarray(p)-y)**2))/(r*(1-r)))


def correction(resid,cs,ct,n_groups,k=500.):
    n=np.bincount(cs,minlength=n_groups).astype(float)
    q=np.bincount(cs,weights=resid,minlength=n_groups)/(n+k)
    q-=np.average(q[cs]); return q[ct]


def prior_table(all_rows, cutoff, k=500.):
    h=all_rows[all_rows.season<cutoff].copy()
    h["full"]=(h.balls_before.eq(3)&h.strikes_before.eq(2)).astype(float)
    g=h.groupby("batter_id").full.agg(["sum","size"])
    p=float(h.full.mean()); return (g["sum"]+k*p)/(g["size"]+k)


def ensemble(tag,kind): return core.ensemble(tag,kind)[:2]


def one(source,target,base,cell):
    bv,ys=ensemble(base,"val"); cv,yc=ensemble(cell,"val")
    bt,yt=ensemble(base,"test"); ct,ytc=ensemble(cell,"test")
    if not(np.array_equal(ys,yc) and np.array_equal(yt,ytc)): raise RuntimeError("target mismatch")
    ps=core.post((1-core.W_CELL)*bv+core.W_CELL*cv)
    pt=core.post((1-core.W_CELL)*bt+core.W_CELL*ct)
    use=["season","game_month","game_type","batter_id","balls_before","strikes_before"]
    d=pd.read_csv(ROOT/"data/train.csv",usecols=use)
    s=d[d.season==source].reset_index(drop=True); t=d[d.season==target].reset_index(drop=True)
    tab_s=prior_table(d,source); tab_t=prior_table(d,target)
    fs=s.batter_id.map(tab_s).fillna(float(tab_s.mean())).to_numpy(float)
    ft=t.batter_id.map(tab_t).fillna(float(tab_t.mean())).to_numpy(float)
    edges=np.unique(np.quantile(fs,np.linspace(0,1,11)[1:-1]))
    cs=np.searchsorted(edges,fs,side="right"); ctg=np.searchsorted(edges,ft,side="right")
    add=correction(ys-ps,cs,ctg,len(edges)+1,k=500.)
    # Preserve the original implementation: "full" used the deep-count gate.
    gate=((t.balls_before+t.strikes_before)>=4).to_numpy(float); add*=gate
    masks={"all":np.ones(len(t),bool),"R":t.game_type.eq("R").to_numpy(),
           "F":t.game_type.eq("F").to_numpy(),"early":t.game_month.le(6).to_numpy(),
           "late":t.game_month.gt(6).to_numpy()}
    out={}
    for k,m in masks.items(): out[k]=raw_bss(yt[m],np.clip(pt[m]+add[m],.01,.99))-raw_bss(yt[m],pt[m])
    print(f"{source}->{target} batter_full_current_gate(actual deep gate) "+
          " ".join(f"{k}={v:+.3f}" for k,v in out.items()))


def main():
    one(2022,2023,"BND22_base","BND22_cell")
    one(2023,2024,"B1J6_base","B1J6_cell")
    return 0


if __name__=="__main__": raise SystemExit(main())

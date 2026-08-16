"""CPU-only audit defined in docs/CONDITIONAL_TM_PREREGISTRATION_20260816.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tm2command_supervised_audit as tm2

ROOT=Path(__file__).resolve().parents[1]
PHYS=["rel_speed","spin_rate","induced_vert_break","horz_break",
      "extension","rel_height","rel_side"]
CONDS=["hand_L","hand_R","count_3b","count_2s","count_neutral"]


def build(cache):
    if cache.exists(): return pd.read_pickle(cache)
    use=["season","pitcher_trackman_id","batter_hand","balls_before","strikes_before"]+PHYS
    d=pd.read_csv(ROOT/"data/trackman_history.csv",usecols=use)
    pm=pd.read_csv(ROOT/"data/processed/pitcher_map2.csv")[["tm_id","pitcher_id"]]
    d=d.merge(pm,left_on="pitcher_trackman_id",right_on="tm_id",how="inner")
    d["hand_L"]=d.batter_hand.astype(str).str.lower().str.startswith("l")
    d["hand_R"]=d.batter_hand.astype(str).str.lower().str.startswith("r")
    d["count_3b"]=d.balls_before.eq(3)
    d["count_2s"]=d.strikes_before.eq(2)
    d["count_neutral"]=~(d.count_3b|d.count_2s)
    keys=["pitcher_id","season"]
    glob=d.groupby(keys)[PHYS].mean()
    rows=[]
    for cond in CONDS:
        q=d[d[cond]].groupby(keys)[PHYS].agg(["mean","size"])
        z=pd.DataFrame(index=q.index)
        for c in PHYS:
            n=q[(c,"size")].to_numpy(float)
            z[f"{cond}_{c}"]=(q[(c,"mean")]-glob.loc[q.index,c]).to_numpy()*n/(n+50.)
        rows.append(z)
    out=pd.concat(rows,axis=1).reset_index()
    if len([c for c in out if c not in keys])!=35: raise RuntimeError("expected 35 columns")
    cache.parent.mkdir(parents=True,exist_ok=True); out.to_pickle(cache); return out


def projector(raw, source):
    cols=[c for c in raw if c not in ("pitcher_id","season")]
    hist=raw[raw.season<source][cols].to_numpy(float)
    med=np.nanmedian(hist,axis=0); med=np.where(np.isfinite(med),med,0.)
    sc=StandardScaler().fit(np.where(np.isfinite(hist),hist,med))
    p=PCA(4,random_state=20260816).fit(sc.transform(np.where(np.isfinite(hist),hist,med)))
    def tr(year):
        q=raw[raw.season==year].copy(); x=q[cols].to_numpy(float)
        x=np.where(np.isfinite(x),x,med)
        v=p.transform(sc.transform(x))
        return pd.DataFrame(v,columns=[f"ctm{i}" for i in range(4)]).assign(
            pitcher_id=q.pitcher_id.to_numpy(),season=year)
    return tr


def duplicate(global100, rep):
    g=global100[global100.season==int(rep.season.iloc[0])].set_index("pitcher_id")
    r=rep.set_index("pitcher_id").join(g.drop(columns="season"),how="inner",rsuffix="_g")
    xcols=[c for c in g if c!="season"]
    x=np.nan_to_num(r[xcols].to_numpy(float)); cv=KFold(5,shuffle=True,random_state=20260816)
    return {c:float(r2_score(r[c],cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=100)),x,r[c],cv=cv)))
            for c in [f"ctm{i}" for i in range(4)]}


def one(raw,g100,source,target,base,cell):
    tr=projector(raw,source); sr=tr(source-1); tt=tr(target-1)
    a=sr.set_index("pitcher_id"); b=tt.set_index("pitcher_id")
    common=a.index.intersection(b.index)
    stability={c:tm2._safe_corr(a.loc[common,c],b.loc[common,c]) for c in [f"ctm{i}" for i in range(4)]}
    use=["season","game_month","game_type","pitcher_id"]
    d=pd.read_csv(ROOT/"data/train.csv",usecols=use)
    s=d[d.season==source].reset_index(drop=True); t=d[d.season==target].reset_index(drop=True)
    ps,ys=tm2.core_predictions(base,cell,"val"); pt,yt=tm2.core_predictions(base,cell,"test")
    cols=[f"ctm{i}" for i in range(4)]
    xs=s[["pitcher_id"]].join(a[cols],on="pitcher_id")[cols].to_numpy(float)
    xt=t[["pitcher_id"]].join(b[cols],on="pitcher_id")[cols].to_numpy(float)
    xs[~np.isfinite(xs).all(1)]=0.; xt[~np.isfinite(xt).all(1)]=0.
    rs,rt=ys-ps,yt-pt; fit,add=tm2.ridge_transfer(xs,xt,rs); rho=tm2._safe_corr(add,rt)
    null=tm2.matched_null(s,t,xs,xt,rs,rt)
    masks={"R":t.game_type.eq("R"),"F":t.game_type.eq("F"),
           "early":t.game_month.le(6),"late":t.game_month.gt(6),
           "known":np.any(xt!=0,axis=1),"cold":np.all(xt==0,axis=1)}
    return {"source":source,"target":target,"same_pitcher_n":len(common),"stability":stability,
            "duplicate_cv_r2":duplicate(g100,tt),"rho_source":tm2._safe_corr(fit,rs),
            "rho_frozen":rho,"ceiling":1e5*rho*rho,
            "null_median":float(np.nanmedian(null)),"null_p99":float(np.nanquantile(null,.99)),
            "null_percentile":float(np.mean(null<=abs(rho))*100),
            "segments":{k:tm2._safe_corr(add[m],rt[m]) for k,m in masks.items()}}


def main():
    raw=build(ROOT/"out/conditional_tm35.pkl")
    g100=tm2.build_tm100(ROOT/"out/tm100_pitcher_season.pkl")
    result={"contract":{"raw_dim":35,"pca_dim":4,"shrink_k":50,"null_reps":400},
            "transfers":[one(raw,g100,2022,2023,"BND22_base","BND22_cell"),
                         one(raw,g100,2023,2024,"B1J6_base","B1J6_cell")]}
    p=ROOT/"out/conditional_tm_geometry_audit.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

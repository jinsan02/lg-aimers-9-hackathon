"""CPU surrogate bridge for all-history exact pitcher-batter pooling.

This does not pretend to be the missing champion-generation rolling OOF archive.
One fixed legal-X nuisance model generation supplies comparable OOF residuals for
every historical source row, allowing CURRENT versus MULTIYEAR persistence to be
screened before any GPU rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import research_frames as rf  # noqa:E402
import pbmf_transfer_audit as pb  # noqa:E402

K=500.0
CAT=["season","game_dayofweek","top_bottom","game_type","base_state",
     "pitcher_hand","batter_hand","pitcher_team_id","batter_team_id"]
DROP={"row_id","control_success","pitcher_id","batter_id"}


def model(num):
    prep=ColumnTransformer([
      ("cat",make_pipeline(SimpleImputer(strategy="most_frequent"),
                            OneHotEncoder(handle_unknown="ignore")),CAT),
      ("num",make_pipeline(SimpleImputer(strategy="median"),StandardScaler()),num)])
    clf=SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-5,max_iter=30,
                      tol=1e-4,average=True,random_state=20260827)
    return make_pipeline(prep,clf)


def pair_table(frame,resid):
    z=pd.DataFrame({"pitcher_id":frame.pitcher_id.to_numpy(),
                    "batter_id":frame.batter_id.to_numpy(),"r":resid})
    g=z.groupby(["pitcher_id","batter_id"]).r.agg(["sum","size"])
    g["offset"]=g["sum"]/(g["size"]+K)
    return g


def apply(frame,tab):
    idx=pd.MultiIndex.from_frame(frame[["pitcher_id","batter_id"]])
    v=tab.offset.reindex(idx).to_numpy(float); known=np.isfinite(v)
    v=np.nan_to_num(v)
    # Same source-applied centring convention as the shipped generator.
    return v,known


def support(tab):
    q=tab["size"].quantile([.25,.5,.75]).to_numpy(float)
    return {"unique_pairs":int(len(tab)),"p25":float(q[0]),"p50":float(q[1]),"p75":float(q[2])}


def bss(y,p): return pb.bss(y,np.clip(p,0,1))


def one(data,target,spec):
    hist=data[data.season<target].reset_index(drop=True)
    num=[c for c in hist.columns if c not in DROP|set(CAT) and
         pd.api.types.is_numeric_dtype(hist[c])]
    cv=StratifiedKFold(5,shuffle=True,random_state=20260827)
    q=cross_val_predict(model(num),hist[CAT+num],hist.control_success,cv=cv,
                        method="predict_proba",n_jobs=1)[:,1]
    resid=hist.control_success.to_numpy(float)-q
    curmask=hist.season.eq(target-1).to_numpy()
    current=pair_table(hist[curmask],resid[curmask]); multi=pair_table(hist,resid)
    b=rf.boundary(data,*spec); t=b["t"]; y=b["yt"]; base=b["pt"]
    vc,kc=apply(t,current); vm,km=apply(t,multi)
    # Remove each map's mean on its own source evidence, exactly once.
    sc,_=apply(hist[curmask],current); sm,_=apply(hist,multi)
    vc-=sc.mean(); vm-=sm.mean()
    pc=np.clip(base+vc,0,1); pm=np.clip(base+vm,0,1)
    masks={"all":np.ones(len(t),bool),"same_coverage":kc&km,
           "coverage_expansion":(~kc)&km,"R":t.game_type.eq("R").to_numpy(),
           "F":t.game_type.eq("F").to_numpy(),"early":t.game_month.le(6).to_numpy(),
           "late":t.game_month.gt(6).to_numpy()}
    delta={k:(bss(y[m],pm[m])-bss(y[m],pc[m]) if m.sum()>20 else None)
           for k,m in masks.items()}
    return {"target":target,"source_rows":int(len(hist)),"oof_bss":bss(hist.control_success,q),
            "current":{"coverage":float(kc.mean()),**support(current)},
            "multiyear":{"coverage":float(km.mean()),**support(multi)},
            "delta_multiyear_minus_current":delta}


def main():
    data=rf.load_train()
    specs={2023:rf.BOUNDARIES[0],2024:rf.BOUNDARIES[1]}
    transfers=[one(data,t,specs[t]) for t in (2023,2024)]
    result={"contract":{"role":"surrogate persistence bridge","model":"SGD logistic",
                        "oof_folds":5,"pair_k":K,"ids_excluded_from_nuisance":True},
            "transfers":transfers,
            "promotion_limit":"can license champion-generation rolling OOF only",
            "independence":{"pass":True,"reason":"official train-only exact-pair tables"}}
    out=ROOT/"out/pb_multiyear_bridge_audit.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

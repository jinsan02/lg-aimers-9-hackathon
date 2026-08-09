"""Freeze career-middle q8/k200 plus PB k500 constants from 2024 OOF residuals."""

from glob import glob
import json
import os
import numpy as np
import pandas as pd

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SEEDS={"42","7","13","3","4","5"}
W,SLOPE,SHIFT=.55,1.0416,.0052

def ens(tag,seeds=None):
    fs=sorted(glob(os.path.join(ROOT,"out",f"cat_{tag}_s*_val_preds.npz")))
    if seeds is not None:
        fs=[f for f in fs if f.rsplit("_s",1)[1].split("_",1)[0] in seeds]
    z=[np.load(f) for f in fs]
    return np.mean([q["pred"].astype(float) for q in z],0),z[0]["y"].astype(float)

def post(p):
    p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT,0,1)

def fit_pair(d,r,k=500.):
    tab=d[["pitcher_id","batter_id"]].copy(); tab["r"]=r
    tab=tab.groupby(["pitcher_id","batter_id"]).r.agg(["sum","size"]).reset_index()
    tab["offset"]=tab["sum"]/(tab["size"]+k)
    src=d.merge(tab[["pitcher_id","batter_id","offset"]],
                on=["pitcher_id","batter_id"],how="left").offset.fillna(0).to_numpy()
    tab["offset"]-=float(src.mean())
    return tab

def main():
    b,y=ens("VB2_base"); c,yc=ens("ZD5",CELL_SEEDS)
    if not np.array_equal(y,yc): raise ValueError("target mismatch")
    col="asof_pitcher_middle_rate"
    d=pd.read_csv(os.path.join(ROOT,"data","train.csv"),
                  usecols=["season",col,"pitcher_id","batter_id"])
    d=d[d.season==2024].reset_index(drop=True)
    p=post((1-W)*b+W*c)
    x=d[col].to_numpy(float); good=x[np.isfinite(x)]
    edges=np.unique(np.quantile(good,np.linspace(0,1,9))); edges[0],edges[-1]=-np.inf,np.inf
    bn=np.searchsorted(edges[1:-1],x,side="right"); bn[~np.isfinite(x)]=-1
    r=y-p; r-=r.mean()
    mt=pd.DataFrame({"bin":bn,"r":r}).groupby("bin").r.agg(["sum","size"])
    mt["offset"]=mt["sum"]/(mt["size"]+200.)
    mt["offset"]-=float(np.average(mt.offset,weights=mt["size"]))
    mid=pd.Series(bn).map(mt.offset).fillna(0).to_numpy(float)
    r2=y-(p+mid); r2-=r2.mean()
    pb=fit_pair(d,r2,500.)
    out=os.path.join(ROOT,"out","final_constants_2024.npz")
    np.savez_compressed(out,
        thresholds=edges[1:-1].astype(np.float64),
        offsets=np.asarray([mt.loc[i,"offset"] for i in range(8)],np.float64),
        nan_offset=np.asarray([mt.loc[-1,"offset"] if -1 in mt.index else 0.],np.float64),
        pb_pitcher=pb.pitcher_id.to_numpy(np.int64),
        pb_batter=pb.batter_id.to_numpy(np.int64),
        pb_offset=pb.offset.to_numpy(np.float64))
    meta={"middle_col":col,"q":8,"middle_k":200,"pb_k":500,
          "thresholds":edges[1:-1].tolist(),"groups":len(pb),
          "middle_adj_sd":float(mid.std())}
    with open(os.path.join(ROOT,"out","final_constants_2024.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print(json.dumps(meta,indent=2)); print(out,os.path.getsize(out))

if __name__=="__main__": main()

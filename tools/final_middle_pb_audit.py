"""Compare current prev5-middle+PB against career-middle+PB on 2023->2024."""

from glob import glob
import os
import numpy as np
import pandas as pd

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W,SLOPE,SHIFT=.55,1.0416,.0052

def ens(tag,kind):
    fs=sorted(glob(os.path.join(ROOT,"out",f"cat_{tag}_s*_{kind}_preds.npz")))
    z=[np.load(f) for f in fs]
    return np.mean([q["pred"].astype(float) for q in z],0),z[0]["y"].astype(float)

def post(p):
    p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT,0,1)

def score(y,p):
    r=y.mean(); return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(r*(1-r)))

def bins(xs,xt,q):
    e=np.unique(np.quantile(xs[np.isfinite(xs)],np.linspace(0,1,q+1)))
    e[0],e[-1]=-np.inf,np.inf
    a=np.searchsorted(e[1:-1],xs,side="right"); b=np.searchsorted(e[1:-1],xt,side="right")
    a[~np.isfinite(xs)]=-1; b[~np.isfinite(xt)]=-1
    return a,b

def fit_key(src,tgt,ks,kt,r,k):
    tab=pd.DataFrame({"key":ks,"r":r}).groupby("key").r.agg(["sum","size"])
    mp=(tab["sum"]/(tab["size"]+k)).to_dict()
    a=pd.Series(ks).map(mp).fillna(0).to_numpy(float)
    b=pd.Series(kt).map(mp).fillna(0).to_numpy(float)
    m=float(a.mean()); return a-m,b-m

def pairkey(d):
    return (d.pitcher_id.astype(str)+"|"+d.batter_id.astype(str)).to_numpy()

def route(s,t,y3,y4,p3,p4,col,q,k):
    bs,bt=bins(s[col].to_numpy(float),t[col].to_numpy(float),q)
    r=y3-p3; r-=r.mean()
    m3,m4=fit_key(s,t,bs,bt,r,k)
    r2=y3-(p3+m3); r2-=r2.mean()
    pb3,pb4=fit_key(s,t,pairkey(s),pairkey(t),r2,500.)
    return p3+m3+pb3,p4+m4+pb4,m4,pb4

def main():
    b3,y3=ens("AB_base","val"); c3,yc3=ens("DW_cell","val")
    b4,y4=ens("AB_base","test"); c4,yc4=ens("DW_cell","test")
    p3=post((1-W)*b3+W*c3); p4=post((1-W)*b4+W*c4)
    use=["season","game_month","pitcher_id","batter_id","asof_pitcher_middle_rate",
         "asof_pitcher_prev5_game_middle_rate"]
    d=pd.read_csv(os.path.join(ROOT,"data","train.csv"),usecols=use)
    s=d[d.season==2023].reset_index(drop=True); t=d[d.season==2024].reset_index(drop=True)
    cur3,cur4,_,_=route(s,t,y3,y4,p3,p4,"asof_pitcher_prev5_game_middle_rate",8,500.)
    new3,new4,mid4,pb4=route(s,t,y3,y4,p3,p4,"asof_pitcher_middle_rate",8,200.)
    early=t.game_month.to_numpy()<=6
    print(f"current prev5+PB raw gain={score(y4,cur4)-score(y4,p4):+.3f}")
    print(f"career+PB raw gain={score(y4,new4)-score(y4,p4):+.3f}")
    print(f"candidate-current all={score(y4,new4)-score(y4,cur4):+.3f} "
          f"early={score(y4[early],new4[early])-score(y4[early],cur4[early]):+.3f} "
          f"late={score(y4[~early],new4[~early])-score(y4[~early],cur4[~early]):+.3f}")
    print(f"candidate components mid_sd={mid4.std():.6f} pb_sd={pb4.std():.6f}")

if __name__=="__main__": main()

"""Find cross-year signal orthogonal to the fixed prev5-middle correction."""

from glob import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, SLOPE, SHIFT = .55, 1.0416, .0052


def ens(tag, kind):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")))
    z = [np.load(f) for f in fs]
    y = z[0]["y"].astype(float)
    return np.mean([q["pred"].astype(float) for q in z], axis=0), y


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    z = np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT, 0, 1)


def bss(y, p, center=True):
    p = np.asarray(p, float).copy()
    if center:
        p += y.mean()-p.mean()
    r = y.mean()
    return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(r*(1-r)))


def qbins(x, z, q=8):
    x, z = np.asarray(x,float), np.asarray(z,float)
    good=x[np.isfinite(x)]
    e=np.unique(np.quantile(good,np.linspace(0,1,q+1)))
    if len(e)<3:
        return None
    e[0],e[-1]=-np.inf,np.inf
    a=np.searchsorted(e[1:-1],x,side="right").astype(str)
    b=np.searchsorted(e[1:-1],z,side="right").astype(str)
    a[~np.isfinite(x)],b[~np.isfinite(z)]="NA","NA"
    return a,b


def combine(a,b):
    return np.char.add(np.char.add(np.asarray(a).astype(str),"|"),np.asarray(b).astype(str))


def fit_apply(ks,kt,r,k):
    tab=pd.DataFrame({"k":ks,"r":r}).groupby("k").r.agg(["sum","size"])
    mp=(tab["sum"]/(tab["size"]+k)).to_dict()
    return pd.Series(kt).map(mp).fillna(0.).to_numpy(),len(tab)


def main():
    b3,y3=ens("AB_base","val"); c3,yc3=ens("DW_cell","val")
    b4,y4=ens("AB_base","test"); c4,yc4=ens("DW_cell","test")
    if not(np.array_equal(y3,yc3) and np.array_equal(y4,yc4)):
        raise ValueError("target mismatch")
    p3=post((1-W)*b3+W*c3); p4=post((1-W)*b4+W*c4)
    use=["season","game_month","game_type","balls_before","strikes_before",
         "pitcher_hand","batter_hand","asof_pitcher_n","asof_pitcher_middle_rate",
         "asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate",
         "asof_pitcher_prev5_game_middle_rate"]
    d=pd.read_csv(os.path.join(ROOT,"data","train.csv"),usecols=use)
    s=d[d.season==2023].reset_index(drop=True); t=d[d.season==2024].reset_index(drop=True)
    s["y"],s["pred"]=y3,p3; t["y"],t["pred"]=y4,p4

    p5="asof_pitcher_prev5_game_middle_rate"
    primary=qbins(s[p5],t[p5],8)
    r3=y3-p3; r3-=r3.mean()
    a3,_=fit_apply(primary[0],primary[0],r3,500.)
    a4,_=fit_apply(primary[0],primary[1],r3,500.)
    base3=p3+a3; base4=p4+a4
    print(f"primary target centered={bss(y4,base4)-bss(y4,p4):+.3f} "
          f"raw={bss(y4,base4,False)-bss(y4,p4,False):+.3f}")

    values={
      "career_middle":s.asof_pitcher_middle_rate,
      "prev1_middle":s.asof_pitcher_prev1_game_middle_rate,
      "prev3_middle":s.asof_pitcher_prev3_game_middle_rate,
      "delta5_career":s[p5]-s.asof_pitcher_middle_rate,
      "delta1_5":s.asof_pitcher_prev1_game_middle_rate-s[p5],
      "delta3_5":s.asof_pitcher_prev3_game_middle_rate-s[p5],
      "pitcher_n":s.asof_pitcher_n,
    }
    target_values={
      "career_middle":t.asof_pitcher_middle_rate,
      "prev1_middle":t.asof_pitcher_prev1_game_middle_rate,
      "prev3_middle":t.asof_pitcher_prev3_game_middle_rate,
      "delta5_career":t[p5]-t.asof_pitcher_middle_rate,
      "delta1_5":t.asof_pitcher_prev1_game_middle_rate-t[p5],
      "delta3_5":t.asof_pitcher_prev3_game_middle_rate-t[p5],
      "pitcher_n":t.asof_pitcher_n,
    }
    cand={}
    for n,x in values.items():
        q=qbins(x,target_values[n],8)
        if q: cand[n]=q
    count3=(s.balls_before.astype(str)+"-"+s.strikes_before.astype(str)).to_numpy()
    count4=(t.balls_before.astype(str)+"-"+t.strikes_before.astype(str)).to_numpy()
    cats={
      "count":(count3,count4),
      "game_type":(s.game_type.astype(str).to_numpy(),t.game_type.astype(str).to_numpy()),
      "pitcher_hand":(s.pitcher_hand.astype(str).to_numpy(),t.pitcher_hand.astype(str).to_numpy()),
      "same_hand":((s.pitcher_hand==s.batter_hand).astype(str).to_numpy(),
                   (t.pitcher_hand==t.batter_hand).astype(str).to_numpy()),
    }
    cand.update(cats)
    for n in ("career_middle","prev1_middle","prev3_middle","delta5_career",
              "delta1_5","pitcher_n"):
        cand[f"prev5*{n}"]=(combine(primary[0],cand[n][0]),combine(primary[1],cand[n][1]))
    for n,(x,z) in cats.items():
        cand[f"prev5*{n}"]=(combine(primary[0],x),combine(primary[1],z))

    first=np.arange(len(s))<len(s)//2
    rows=[]
    for name,(ks,kt) in cand.items():
      for k in (200.,500.,1000.,2000.):
        rr=y3-base3; rr-=rr.mean()
        add4,ng=fit_apply(ks,kt,rr,k)
        early=t.game_month.to_numpy()<=6
        cv=[]
        for fit,val in ((first,~first),(~first,first)):
            rp=y3[fit]-p3[fit]; rp-=rp.mean()
            prim_fit,_=fit_apply(np.asarray(primary[0])[fit],np.asarray(primary[0])[fit],rp,500.)
            prim_val,_=fit_apply(np.asarray(primary[0])[fit],np.asarray(primary[0])[val],rp,500.)
            rrfit=y3[fit]-(p3[fit]+prim_fit); rrfit-=rrfit.mean()
            sec_val,_=fit_apply(np.asarray(ks)[fit],np.asarray(ks)[val],rrfit,k)
            cv.append(bss(y3[val],p3[val]+prim_val+sec_val)-bss(y3[val],p3[val]+prim_val))
        rows.append(dict(name=name,k=k,groups=ng,source_cv=np.mean(cv),
            target=bss(y4,base4+add4)-bss(y4,base4),
            raw=bss(y4,base4+add4,False)-bss(y4,base4,False),
            early=bss(y4[early],base4[early]+add4[early])-bss(y4[early],base4[early]),
            late=bss(y4[~early],base4[~early]+add4[~early])-bss(y4[~early],base4[~early]),
            adj_sd=add4.std()))
    out=pd.DataFrame(rows).sort_values("target",ascending=False)
    print(out.head(60).to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    stable=out[(out.source_cv>0)&(out.early>0)&(out.late>0)&(out.target>=2)]
    print("\nPROMOTE")
    print(stable.to_string(index=False,float_format=lambda x:f"{x:+.3f}") if len(stable) else "none")
    out.to_csv(os.path.join(ROOT,"out","middle_orthogonal_audit.csv"),index=False)
    return 0

if __name__=="__main__":
    raise SystemExit(main())

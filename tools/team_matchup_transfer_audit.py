"""Audit team/opponent strategy effects after the fixed middle correction."""

from glob import glob
import os

import numpy as np
import pandas as pd

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W,SLOPE,SHIFT=.55,1.0416,.0052

def ens(tag,kind):
    z=[np.load(f) for f in sorted(glob(os.path.join(ROOT,"out",f"cat_{tag}_s*_{kind}_preds.npz")))]
    return np.mean([q["pred"].astype(float) for q in z],0),z[0]["y"].astype(float)

def post(p):
    p=np.clip(p,1e-6,1-1e-6); z=np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT,0,1)

def bss(y,p,center=True):
    p=np.asarray(p,float).copy()
    if center:
        p+=y.mean()-p.mean()
    r=y.mean()
    return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(r*(1-r)))

def qbins(x,z,q=8):
    x,z=np.asarray(x,float),np.asarray(z,float); g=x[np.isfinite(x)]
    e=np.unique(np.quantile(g,np.linspace(0,1,q+1))); e[0],e[-1]=-np.inf,np.inf
    a=np.searchsorted(e[1:-1],x,side="right").astype(str)
    b=np.searchsorted(e[1:-1],z,side="right").astype(str)
    a[~np.isfinite(x)],b[~np.isfinite(z)]="NA","NA"; return a,b

def key(d,cols):
    return d[list(cols)].fillna("NA").astype(str).agg("|".join,axis=1).to_numpy()

def fit_apply(ks,kt,r,k):
    tab=pd.DataFrame({"key":ks,"r":r}).groupby("key").r.agg(["sum","size"])
    mp=(tab["sum"]/(tab["size"]+k)).to_dict()
    z=pd.Series(kt).map(mp)
    return z.fillna(0.).to_numpy(),len(tab),z.notna().mean()

def main():
    b3,y3=ens("AB_base","val"); c3,yc3=ens("DW_cell","val")
    b4,y4=ens("AB_base","test"); c4,yc4=ens("DW_cell","test")
    p3=post((1-W)*b3+W*c3); p4=post((1-W)*b4+W*c4)
    cols=["season","game_month","game_type","pitcher_id","batter_id",
          "pitcher_team_id","batter_team_id","pitcher_hand","batter_hand",
          "balls_before","strikes_before","inning",
          "asof_pitcher_prev5_game_middle_rate"]
    d=pd.read_csv(os.path.join(ROOT,"data","train.csv"),usecols=cols)
    s=d[d.season==2023].reset_index(drop=True); t=d[d.season==2024].reset_index(drop=True)
    mid3,mid4=qbins(s.asof_pitcher_prev5_game_middle_rate,t.asof_pitcher_prev5_game_middle_rate)
    r=y3-p3; r-=r.mean(); a3,_,_=fit_apply(mid3,mid3,r,500.); a4,_,_=fit_apply(mid3,mid4,r,500.)
    base3,base4=p3+a3,p4+a4
    print(f"middle baseline gain={bss(y4,base4)-bss(y4,p4):+.3f}")

    specs={
      "pitcher_team":["pitcher_team_id"],
      "batter_team":["batter_team_id"],
      "team_matchup":["pitcher_team_id","batter_team_id"],
      "pitcher_opponent":["pitcher_id","batter_team_id"],
      "batter_opponent":["batter_id","pitcher_team_id"],
      "pitcher_batter_pair":["pitcher_id","batter_id"],
      "pteam_bhand":["pitcher_team_id","batter_hand"],
      "bteam_phand":["batter_team_id","pitcher_hand"],
      "team_matchup_count":["pitcher_team_id","batter_team_id","balls_before","strikes_before"],
      "pteam_count":["pitcher_team_id","balls_before","strikes_before"],
      "bteam_count":["batter_team_id","balls_before","strikes_before"],
      "pteam_inning":["pitcher_team_id","inning"],
    }
    first=np.arange(len(s))<len(s)//2; early=t.game_month.to_numpy()<=6
    rows=[]
    for name,cs in specs.items():
      ks,kt=key(s,cs),key(t,cs)
      for k in (200.,500.,1000.,3000.):
        rr=y3-base3; rr-=rr.mean(); add,ng,cov=fit_apply(ks,kt,rr,k)
        cv=[]
        for fit,val in ((first,~first),(~first,first)):
          rp=y3[fit]-p3[fit]; rp-=rp.mean()
          pf,_,_=fit_apply(mid3[fit],mid3[fit],rp,500.); pv,_,_=fit_apply(mid3[fit],mid3[val],rp,500.)
          rf=y3[fit]-(p3[fit]+pf); rf-=rf.mean()
          sv,_,_=fit_apply(ks[fit],ks[val],rf,k)
          cv.append(bss(y3[val],p3[val]+pv+sv)-bss(y3[val],p3[val]+pv))
        rows.append(dict(name=name,k=k,groups=ng,coverage=cov,source_cv=np.mean(cv),
          target=bss(y4,base4+add)-bss(y4,base4),
          early=bss(y4[early],base4[early]+add[early])-bss(y4[early],base4[early]),
          late=bss(y4[~early],base4[~early]+add[~early])-bss(y4[~early],base4[~early]),
          adj_sd=add.std()))
    out=pd.DataFrame(rows).sort_values("target",ascending=False)
    print(out.to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    stable=out[(out.source_cv>0)&(out.early>0)&(out.late>0)&(out.target>=2)]
    print("\nPROMOTE")
    print(stable.to_string(index=False,float_format=lambda x:f"{x:+.3f}") if len(stable) else "none")

    # The two most plausible effects overlap substantially: repeated pitcher-
    # batter meetings are concentrated within the same opponent team.  Measure
    # their incremental value in both orders instead of adding their standalone
    # gains.
    print("\nSEQUENTIAL")
    seq=[]
    for n1,c1,k1,n2,c2,k2 in (
        ("batter_opponent",specs["batter_opponent"],3000.,
         "pitcher_batter_pair",specs["pitcher_batter_pair"],500.),
        ("pitcher_batter_pair",specs["pitcher_batter_pair"],500.,
         "batter_opponent",specs["batter_opponent"],3000.),
    ):
      k1s,k1t=key(s,c1),key(t,c1); k2s,k2t=key(s,c2),key(t,c2)
      rr=y3-base3; rr-=rr.mean()
      a1,_,_=fit_apply(k1s,k1t,rr,k1)
      f1,_,_=fit_apply(k1s,k1s,rr,k1)
      # Freeze a conditional map without silently changing the global SHIFT.
      m1=float(f1.mean()); a1-=m1; f1-=m1
      rr2=y3-(base3+f1); rr2-=rr2.mean()
      a2,_,_=fit_apply(k2s,k2t,rr2,k2)
      f2,_,_=fit_apply(k2s,k2s,rr2,k2)
      m2=float(f2.mean()); a2-=m2
      both=bss(y4,base4+a1+a2)-bss(y4,base4)
      inc=bss(y4,base4+a1+a2)-bss(y4,base4+a1)
      both_raw=bss(y4,base4+a1+a2,False)-bss(y4,base4,False)
      inc_raw=(bss(y4,base4+a1+a2,False)-
               bss(y4,base4+a1,False))
      inc_e=(bss(y4[early],base4[early]+a1[early]+a2[early])-
             bss(y4[early],base4[early]+a1[early]))
      inc_l=(bss(y4[~early],base4[~early]+a1[~early]+a2[~early])-
             bss(y4[~early],base4[~early]+a1[~early]))
      seq.append(dict(first=n1,second=n2,both=both,both_raw=both_raw,
                      second_inc=inc,second_raw=inc_raw,
                      second_early=inc_e,second_late=inc_l))
    print(pd.DataFrame(seq).to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    out.to_csv(os.path.join(ROOT,"out","team_matchup_transfer_audit.csv"),index=False)
    return 0

if __name__=="__main__": raise SystemExit(main())

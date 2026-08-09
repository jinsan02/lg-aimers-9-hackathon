"""Second-transition audit: fit 2022 residual maps, apply unchanged to 2023."""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, SLOPE, SHIFT = .55, 1.0416, .0052


def load(tag, kind):
    z = np.load(os.path.join(ROOT, "out", f"cat_{tag}_{kind}_preds.npz"))
    return z["pred"].astype(float), z["y"].astype(float)


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    z = np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT, 0, 1)


def score(y, p, center=False):
    p = np.asarray(p, float).copy()
    if center:
        p += y.mean()-p.mean()
    r = y.mean()
    return 1e5*(1-np.mean((np.clip(p, 0, 1)-y)**2)/(r*(1-r)))


def qbin(xs, xt):
    good = xs[np.isfinite(xs)]
    e = np.unique(np.quantile(good, np.linspace(0, 1, 9)))
    e[0], e[-1] = -np.inf, np.inf
    a = np.searchsorted(e[1:-1], xs, side="right")
    b = np.searchsorted(e[1:-1], xt, side="right")
    a[~np.isfinite(xs)] = -1; b[~np.isfinite(xt)] = -1
    return a, b


def fit_apply(src, tgt, cols, resid, k):
    tab = src[list(cols)].copy(); tab["r"] = resid
    tab = tab.groupby(list(cols), dropna=False).r.agg(["sum", "size"]).reset_index()
    tab["off"] = tab["sum"]/(tab["size"]+k)
    sf = src.merge(tab[list(cols)+["off"]], on=list(cols), how="left").off.fillna(0).to_numpy()
    tf = tgt.merge(tab[list(cols)+["off"]], on=list(cols), how="left").off
    cov = float(tf.notna().mean())
    tf = tf.fillna(0).to_numpy()
    mean = float(sf.mean())
    return sf-mean, tf-mean, cov


def report(name, y, before, after, month):
    early = month <= 6
    print(f"{name:<12} raw={score(y,after)-score(y,before):+.3f} "
          f"center={score(y,after,True)-score(y,before,True):+.3f} "
          f"early={score(y[early],after[early])-score(y[early],before[early]):+.3f} "
          f"late={score(y[~early],after[~early])-score(y[~early],before[~early]):+.3f}")


def main():
    b2,y2=load("HB22_base","val"); c2,yc2=load("HC22_cell","val")
    b3,y3=load("HB22_base","test"); c3,yc3=load("HC22_cell","test")
    if not (np.array_equal(y2,yc2) and np.array_equal(y3,yc3)):
        raise ValueError("target mismatch")
    cols=["season","game_month","game_type","pitcher_id","batter_id",
          "pitcher_team_id","batter_team_id","balls_before","strikes_before",
          "asof_pitcher_middle_rate","asof_pitcher_prev5_game_middle_rate"]
    d=pd.read_csv(os.path.join(ROOT,"data","train.csv"),usecols=cols)
    d=d[~((d.game_type=="F") & (d.season<=2022))]
    s=d[d.season==2022].reset_index(drop=True); t=d[d.season==2023].reset_index(drop=True)
    if len(s)!=len(y2) or len(t)!=len(y3):
        raise ValueError(f"row mismatch data={len(s)},{len(t)} pred={len(y2)},{len(y3)}")
    p2=post((1-W)*b2+W*c2); p3=post((1-W)*b3+W*c3)

    q2,q3=qbin(s.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
               t.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    ss=s.assign(_bin=q2); tt=t.assign(_bin=q3)
    r=y2-p2; r-=r.mean()
    m2,m3,mcov=fit_apply(ss,tt,["_bin"],r,500.)
    base2,base3=p2+m2,p3+m3
    report("middle",y3,p3,base3,t.game_month.to_numpy())

    r=y2-base2; r-=r.mean()
    bo2,bo3,bcov=fit_apply(s,t,["batter_id","pitcher_team_id"],r,3000.)
    report("BO",y3,base3,base3+bo3,t.game_month.to_numpy())

    pb0_2,pb0_3,pb0cov=fit_apply(s,t,["pitcher_id","batter_id"],r,500.)
    report("PB_alone",y3,base3,base3+pb0_3,t.game_month.to_numpy())

    # Direct replacement candidate: career-middle q8/k200, then refit PB residual.
    cq2,cq3=qbin(s.asof_pitcher_middle_rate.to_numpy(float),
                 t.asof_pitcher_middle_rate.to_numpy(float))
    cs=s.assign(_cbin=cq2); ct=t.assign(_cbin=cq3)
    cr=y2-p2; cr-=cr.mean()
    cm2,cm3,_=fit_apply(cs,ct,["_cbin"],cr,200.)
    cr2=y2-(p2+cm2); cr2-=cr2.mean()
    cp2,cp3,_=fit_apply(s,t,["pitcher_id","batter_id"],cr2,500.)
    current=base3+pb0_3; candidate=p3+cm3+cp3
    report("career+PB",y3,p3,candidate,t.game_month.to_numpy())
    report("career-current",y3,current,candidate,t.game_month.to_numpy())

    print("\nOTHER STANDALONE AFTER MIDDLE")
    specs = [
        ("pitcher_opp", ["pitcher_id","batter_team_id"], 3000.),
        ("team_match", ["pitcher_team_id","batter_team_id"], 3000.),
        ("bteam_count", ["batter_team_id","balls_before","strikes_before"], 3000.),
        ("match_count", ["pitcher_team_id","batter_team_id",
                         "balls_before","strikes_before"], 3000.),
    ]
    for name, keys, k in specs:
        _, a, cov = fit_apply(s,t,keys,r,k)
        report(name,y3,base3,base3+a,t.game_month.to_numpy())
        print(f"  coverage={cov:.3f}")

    r=y2-(base2+bo2); r-=r.mean()
    pb2,pb3,pcov=fit_apply(s,t,["pitcher_id","batter_id"],r,500.)
    report("PB_after_BO",y3,base3+bo3,base3+bo3+pb3,t.game_month.to_numpy())
    report("BO+PB",y3,base3,base3+bo3+pb3,t.game_month.to_numpy())
    print(f"coverage middle={mcov:.3f} BO={bcov:.3f} "
          f"PB_alone={pb0cov:.3f} PB_after_BO={pcov:.3f}")


if __name__ == "__main__":
    main()

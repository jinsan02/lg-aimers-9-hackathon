"""CPU-only audit of historical pitcher x pitch-family command skill."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import fpipe  # noqa: E402
import joint_pitch  # noqa: E402
import tm2command_supervised_audit as core  # noqa: E402

TYPES = ["fastball", "breaking", "offspeed", "other"]
MIX_TYPES = TYPES[:3]
MIX_COLS = [f"asof_pitcher_{x}_rate" for x in MIX_TYPES]
K = 80.0
ALPHA = 100.0
REPS = 400


def exact_linked(cache: Path, data: pd.DataFrame) -> pd.DataFrame:
    if cache.exists(): return pd.read_pickle(cache)
    key = joint_pitch.KEY
    main = data[["row_id", "control_success"] + key]
    tm = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=joint_pitch.CONTEXT +
                     ["pitcher_trackman_id", "batter_trackman_id", "pitch_type_group"])
    pm = pd.read_csv(ROOT / "data/processed/pitcher_map2.csv")
    bm = pd.read_csv(ROOT / "data/processed/batter_map2.csv")
    tm["pitcher_id"] = tm.pitcher_trackman_id.map(
        pm.drop_duplicates("tm_id").set_index("tm_id").pitcher_id)
    tm["batter_id"] = tm.batter_trackman_id.map(
        bm.drop_duplicates("tm_batter_id").set_index("tm_batter_id").batter_id)
    tm["top_bottom"] = tm.top_bottom.map({"Top": "T", "Bottom": "B"})
    for c in ("pitcher_hand", "batter_hand"):
        tm[c] = tm[c].map({"Left": 1, "Right": 2})
    tm = tm[tm.pitcher_id.notna() & tm.batter_id.notna()].copy()
    tm[["pitcher_id", "batter_id"]] = tm[["pitcher_id", "batter_id"]].astype(np.int64)
    tm["pitch_type_group"] = tm.pitch_type_group.where(
        tm.pitch_type_group.isin(MIX_TYPES), "other")
    mc = main.groupby(key, dropna=False).size().rename("main_n")
    tg = tm.groupby(key, dropna=False).agg(
        tm_n=("pitch_type_group", "size"), pitch_type_group=("pitch_type_group", "first"))
    safe = (mc[mc.eq(1)].reset_index().drop(columns="main_n")
            .merge(tg[tg.tm_n.eq(1)].reset_index().drop(columns="tm_n"), on=key))
    out = main.merge(safe, on=key, how="inner")[["row_id", "season", "pitcher_id",
                                                  "control_success", "pitch_type_group"]]
    cache.parent.mkdir(parents=True, exist_ok=True); out.to_pickle(cache)
    return out


def profile(linked, cutoff):
    h = linked[linked.season < cutoff]
    prior = h.groupby("pitch_type_group").control_success.mean().reindex(TYPES)
    prior = prior.fillna(h.control_success.mean())
    g = h.groupby(["pitcher_id", "pitch_type_group"]).control_success.agg(["sum", "size"])
    wide = pd.DataFrame(index=h.pitcher_id.unique())
    for t in TYPES:
        q = g.xs(t, level=1, drop_level=True) if t in g.index.get_level_values(1) else pd.DataFrame()
        if len(q): wide[t] = (q["sum"] + K * prior[t]) / (q["size"] + K)
        wide[t] = wide.get(t, pd.Series(index=wide.index, dtype=float)).fillna(prior[t])
    shares = h.pitch_type_group.value_counts(normalize=True).reindex(MIX_TYPES).fillna(0)
    shares = shares / shares.sum()
    return wide[TYPES], prior, shares


def candidate(rows, prof, prior, shares):
    w = rows[MIX_COLS].fillna(0).to_numpy(float)
    den = w.sum(1); bad = den <= 0
    w[~bad] /= den[~bad, None]; w[bad] = shares.to_numpy(float)
    s = prof.reindex(rows.pitcher_id).to_numpy(float)
    miss = ~np.isfinite(s); s[miss] = np.broadcast_to(prior.to_numpy(float), s.shape)[miss]
    return (w * s[:, :3]).sum(1), np.ptp(s, axis=1), w


def persistence(linked):
    tabs = {}
    for year, d in linked.groupby("season"):
        pri = d.groupby("pitch_type_group").control_success.mean().reindex(TYPES)
        g = d.groupby(["pitcher_id", "pitch_type_group"]).control_success.agg(["sum", "size"])
        g["skill"] = (g["sum"] + K * g.index.get_level_values(1).map(pri)) / (g["size"] + K)
        pg = d.groupby("pitcher_id").control_success.agg(["sum", "size"])
        ov = (pg["sum"] + K * d.control_success.mean()) / (pg["size"] + K)
        g["dev"] = g.skill - g.index.get_level_values(0).map(ov)
        tabs[int(year)] = g.reset_index()
    out = {}
    for a, b in ((2021, 2022), (2022, 2023), (2023, 2024)):
        q = tabs[a].merge(tabs[b], on=["pitcher_id", "pitch_type_group"], suffixes=("_a", "_b"))
        out[f"{a}->{b}"] = {k: {"pearson": core._safe_corr(q[f"{k}_a"], q[f"{k}_b"]),
                                    "spearman": float(q[f"{k}_a"].corr(q[f"{k}_b"], method="spearman")),
                                    "n": int(len(q))} for k in ("skill", "dev")}
    return out


def raw_null(rows_s, rows_t, ps, pt, pri_s, pri_t, sh_s, sh_t, rs, rt, reps=REPS):
    pids = np.union1d(rows_s.pitcher_id.unique(), rows_t.pitcher_id.unique())
    as_ = ps.reindex(pids).fillna(pri_s).to_numpy(float)
    at = pt.reindex(pids).fillna(pri_t).to_numpy(float)
    vals=[]
    for k in range(reps):
        p=np.random.default_rng(20260816+k).permutation(len(pids))
        qs=pd.DataFrame(as_[p],index=pids,columns=TYPES); qt=pd.DataFrame(at[p],index=pids,columns=TYPES)
        zs,_,_=candidate(rows_s,qs,pri_s,sh_s); zt,_,_=candidate(rows_t,qt,pri_t,sh_t)
        _,add=core.ridge_transfer(zs[:,None],zt[:,None],rs)
        vals.append(abs(core._safe_corr(add,rt)))
    return np.asarray(vals)


def reconstruct_pair(xs, xt, zs, zt):
    cv=GroupKFold(5); groups=xs.index.to_numpy()
    oof=cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)),xs,zs,cv=cv,groups=groups)
    m=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)).fit(xs,zs)
    return zs-oof, zt-m.predict(xt)


def boundary(data, linked, source, target, base, cell):
    pack=joblib.load(ROOT/f"model/cat_{base}_s3.pkl")
    sr=data[data.season.eq(source)].reset_index(drop=True); tr=data[data.season.eq(target)].reset_index(drop=True)
    sx=fpipe.transform(sr.drop(columns="control_success").copy(),pack["fpipe"])
    tx=fpipe.transform(tr.drop(columns="control_success").copy(),pack["fpipe"])
    nums=[c for c in pack["features"] if c not in pack["cat_cols"]]
    prof_s,pri_s,sh_s=profile(linked,source); prof_t,pri_t,sh_t=profile(linked,target)
    zs,spread_s,_=candidate(sr,prof_s,pri_s,sh_s); zt,spread_t,_=candidate(tr,prof_t,pri_t,sh_t)
    ps,ys=core.core_predictions(base,cell,"val"); pt,yt=core.core_predictions(base,cell,"test")
    rs,rt=ys-ps,yt-pt; fit,add=core.ridge_transfer(zs[:,None],zt[:,None],rs)
    null=raw_null(sr,tr,prof_s,prof_t,pri_s,pri_t,sh_s,sh_t,rs,rt)
    masks={"R":tr.game_type.eq("R"),"F":tr.game_type.eq("F"),"early":tr.game_month.le(6),
           "late":tr.game_month.gt(6),"known":tr.pitcher_id.isin(prof_t.index),
           "cold":~tr.pitcher_id.isin(prof_t.index)}
    raw={"rho":core._safe_corr(add,rt),"rho_source":core._safe_corr(fit,rs),
         "null_percentile":float(np.mean(null<=abs(core._safe_corr(add,rt)))*100),
         "null_p95":float(np.quantile(null,.95)),"null_p99":float(np.quantile(null,.99)),
         "segments":{k:core._safe_corr(add[m],rt[m]) for k,m in masks.items()}}
    ax=sx[nums].assign(pitcher_id=sr.pitcher_id).groupby("pitcher_id").mean()
    at=tx[nums].assign(pitcher_id=tr.pitcher_id).groupby("pitcher_id").mean()
    zsa=pd.Series(zs,index=sr.pitcher_id).groupby(level=0).mean(); zta=pd.Series(zt,index=tr.pitcher_id).groupby(level=0).mean()
    common=ax.index.intersection(at.index).intersection(zsa.index).intersection(zta.index)
    ax,at=ax.loc[common],at.loc[common]; med=ax.median().fillna(0); ax=ax.fillna(med); at=at.fillna(med)
    us,ut=reconstruct_pair(ax,at,zsa.loc[common].to_numpy(),zta.loc[common].to_numpy())
    usm=pd.Series(us,index=common); utm=pd.Series(ut,index=common)
    urs=sr.pitcher_id.map(usm).fillna(0).to_numpy(float)[:,None]
    urt=tr.pitcher_id.map(utm).fillna(0).to_numpy(float)[:,None]
    _,uadd=core.ridge_transfer(urs,urt,rs); urho=core._safe_corr(uadd,rt)
    unull=[]
    pair=np.column_stack([zsa.loc[common],zta.loc[common]])
    for k in range(REPS):
        p=np.random.default_rng(20261816+k).permutation(len(common))
        nus,nut=reconstruct_pair(ax,at,pair[p,0],pair[p,1])
        ns=sr.pitcher_id.map(pd.Series(nus,index=common)).fillna(0).to_numpy(float)[:,None]
        nt=tr.pitcher_id.map(pd.Series(nut,index=common)).fillna(0).to_numpy(float)[:,None]
        _,na=core.ridge_transfer(ns,nt,rs); unull.append(abs(core._safe_corr(na,rt)))
    unique={"rho":urho,"null_percentile":float(np.mean(np.asarray(unull)<=abs(urho))*100),
            "null_p95":float(np.quantile(unull,.95)),"null_p99":float(np.quantile(unull,.99)),
            "var_z":float(np.var(np.r_[zsa.loc[common],zta.loc[common]])),
            "var_unique":float(np.var(np.r_[us,ut])),
            "variance_fraction":float(np.var(np.r_[us,ut])/np.var(np.r_[zsa.loc[common],zta.loc[common]]))}
    diag=pd.DataFrame({"pitcher_id":common,"eval_season":target,"z":zta.loc[common],"unique":ut}).set_index("pitcher_id").join(at)
    return raw,unique,diag,nums,{"mean":float(np.mean(zt)),"sd":float(np.std(zt)),
                                 "coverage":float(tr.pitcher_id.isin(prof_t.index).mean()),
                                 "spread_mean":float(np.mean(spread_t))}


def novelty(data,diags,nums):
    z=pd.concat(diags).reset_index(); x=z[nums].to_numpy(float)
    med=np.nanmedian(x,axis=0); x=np.where(np.isfinite(x),x,np.where(np.isfinite(med),med,0))
    cv=GroupKFold(5); yp=cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)),x,z.z,cv=cv,groups=z.pitcher_id)
    uyp=cross_val_predict(make_pipeline(StandardScaler(),Ridge(alpha=ALPHA)),x,z.unique,cv=cv,groups=z.pitcher_id)
    # requested correlations on the latest season; general skill comes from shipped cell pack
    q=z[z.eval_season.eq(2024)].copy(); gpack=joblib.load(ROOT/"model/cat_GSKDEP_cell_s3.pkl")
    r24=data[data.season.eq(2024)].reset_index(drop=True); gx=fpipe.transform(r24.drop(columns="control_success"),gpack["fpipe"])
    gh=gx.assign(pitcher_id=r24.pitcher_id).groupby("pitcher_id").skill_hat.mean(); q["skill_hat"]=q.pitcher_id.map(gh)
    cs={"raw_pitcher":core._safe_corr(q.z,q.asof_pitcher_success_rate),
        "std_pitcher":core._safe_corr(q.z,q.std_asof_pitcher_success_rate),
        "skill_pc":core._safe_corr(q.z,q.skill_pc_hat),"skill_hat":core._safe_corr(q.z,q.skill_hat)}
    for c in MIX_COLS: cs[c]=core._safe_corr(q.z,q[c])
    return {"cv_r2":float(r2_score(z.z,yp)),
            "unique_cv_r2":float(r2_score(z.unique,uyp)),"correlations":cs}


def main():
    header=list(pd.read_csv(ROOT/"data/test.csv",nrows=0).columns)
    data=pd.read_csv(ROOT/"data/train.csv",usecols=header+["control_success"])
    linked=exact_linked(ROOT/"out/tm_type_command_linked.pkl",data)
    a,ua,da,ca,dista=boundary(data,linked,2022,2023,"BND22_base","BND22_cell")
    b,ub,db,cb,distb=boundary(data,linked,2023,2024,"B1J6_base","B1J6_cell")
    cols=[c for c in ca if c in cb]; nov=novelty(data,[da[["eval_season","z","unique"]+cols],db[["eval_season","z","unique"]+cols]],cols)
    bys=(linked.groupby("season").size()/data.groupby("season").size()).to_dict()
    counts=linked.pitch_type_group.value_counts().reindex(TYPES,fill_value=0)
    ns=linked.groupby(["pitcher_id","pitch_type_group"]).size()
    result={"contract":{"k":K,"types":TYPES,"mix_formula":"renormalized official 3-group asof mix","null_reps":REPS},
      "linkage":{"linked_rows":int(len(linked)),"coverage":float(len(linked)/len(data)),
       "pitcher_coverage":float(linked.pitcher_id.nunique()/data.pitcher_id.nunique()),
       "season_coverage":{str(int(k)):float(v) for k,v in bys.items()},"type_counts":counts.to_dict()},
      "persistence":persistence(linked),"distribution":{"2023":dista,"2024":distb},
      "novelty":nov,"transfer":{"2022->2023":{"raw":a,"unique":ua},"2023->2024":{"raw":b,"unique":ub}},
      "sample":{"effective_types_mean":float(linked.groupby("pitcher_id").pitch_type_group.nunique().mean()),
       "median_pitcher_type_n":float(ns.median()),"n_lt10":float((ns<10).mean()),"n_lt30":float((ns<30).mean()),"n_lt50":float((ns<50).mean())},
      "independence":{"pass":True,"reason":"train-only exact linkage and frozen pitcher lookup; row-local official asof mix"}}
    p=ROOT/"out/tm_type_command_mix_audit.json"; p.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())

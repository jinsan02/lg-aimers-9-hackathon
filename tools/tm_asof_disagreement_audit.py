"""CPU gate for Trackman-implied command minus official row-local ASOF state."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tm2command_supervised_audit as tm2

ROOT = Path(__file__).resolve().parents[1]
RATE = {
    "success": "asof_pitcher_success_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "reverse": "asof_pitcher_reverse_rate",
}
COLS = [f"gap_{c}" for c in tm2.OUTCOMES]


def frames(source, target, base, cell, outputs):
    use = ["season", "game_month", "game_type", "pitcher_id"] + list(RATE.values())
    d = pd.read_csv(ROOT / "data/train.csv", usecols=use)
    s = d[d.season == source].reset_index(drop=True)
    t = d[d.season == target].reset_index(drop=True)
    ps, ys = tm2.core_predictions(base, cell, "val")
    pt, yt = tm2.core_predictions(base, cell, "test")
    if len(s) != len(ys) or len(t) != len(yt): raise RuntimeError("row mismatch")
    so = outputs[outputs.season == source].set_index("pitcher_id")
    to = outputs[outputs.season == target].set_index("pitcher_id")
    xs, xt = [], []
    for c in tm2.OUTCOMES:
        a = s.pitcher_id.map(so[f"tm_cmd_{c}"]).to_numpy(float)
        b = t.pitcher_id.map(to[f"tm_cmd_{c}"]).to_numpy(float)
        xs.append(a-s[RATE[c]].to_numpy(float)); xt.append(b-t[RATE[c]].to_numpy(float))
    xs, xt = np.column_stack(xs), np.column_stack(xt)
    known_s = np.isfinite(xs).all(1); known_t = np.isfinite(xt).all(1)
    xs[~known_s] = 0.; xt[~known_t] = 0.
    return s, t, ps, pt, ys, yt, xs, xt, so, to


def matched_null(s, t, so, to, resid_s, resid_t, reps=400):
    pids = np.union1d(so.index.to_numpy(), to.index.to_numpy())
    vals = []
    for k in range(reps):
        rng = np.random.default_rng(2026081600+k)
        perm = pd.Series(rng.permutation(pids), index=pids)
        a, b = [], []
        for c in tm2.OUTCOMES:
            sv = s.pitcher_id.map(perm).map(so[f"tm_cmd_{c}"]).to_numpy(float)
            tv = t.pitcher_id.map(perm).map(to[f"tm_cmd_{c}"]).to_numpy(float)
            a.append(sv-s[RATE[c]].to_numpy(float)); b.append(tv-t[RATE[c]].to_numpy(float))
        a, b = np.column_stack(a), np.column_stack(b)
        a[~np.isfinite(a).all(1)] = 0.; b[~np.isfinite(b).all(1)] = 0.
        _, add = tm2.ridge_transfer(a, b, resid_s)
        vals.append(abs(tm2._safe_corr(add, resid_t)))
    return np.asarray(vals)


def duplicate_r2(x, rows, pack_path):
    pack = joblib.load(pack_path)
    raw_cols = list(pd.read_csv(ROOT / "data/test.csv", nrows=0).columns)
    raw = pd.read_csv(ROOT / "data/train.csv", usecols=raw_cols)
    raw = raw[raw.season == int(rows.season.iloc[0])].reset_index(drop=True)
    z = tm2.fpipe.transform(raw.copy(), pack["fpipe"])
    nums = [c for c in pack["features"] if c not in pack["cat_cols"]]
    agg = z.assign(pitcher_id=raw.pitcher_id).groupby("pitcher_id")[nums].mean()
    gx = pd.DataFrame(x, columns=COLS).assign(pitcher_id=rows.pitcher_id).groupby("pitcher_id").mean()
    q = gx.join(agg, how="inner"); xx = np.nan_to_num(q[nums].to_numpy(float))
    cv = KFold(5, shuffle=True, random_state=20260816)
    return {c: float(r2_score(q[c], cross_val_predict(
        make_pipeline(StandardScaler(), Ridge(alpha=tm2.ALPHA)), xx, q[c], cv=cv))) for c in COLS}


def one(outputs, source, target, base, cell, pack_tag):
    s,t,ps,pt,ys,yt,xs,xt,so,to = frames(source,target,base,cell,outputs)
    rs, rt = ys-ps, yt-pt
    a,b = tm2.ridge_transfer(xs,xt,rs); rho=tm2._safe_corr(b,rt)
    null=matched_null(s,t,so,to,rs,rt)
    masks={"R":t.game_type.eq("R"),"F":t.game_type.eq("F"),
           "early":t.game_month.le(6),"late":t.game_month.gt(6),
           "known":np.any(xt != 0,axis=1),"cold":np.all(xt == 0,axis=1)}
    return {"source":source,"target":target,"rho_source":tm2._safe_corr(a,rs),
            "rho_frozen":rho,"ceiling":1e5*rho*rho,
            "null_median":float(np.nanmedian(null)),"null_p99":float(np.nanquantile(null,.99)),
            "null_percentile":float(np.mean(null<=abs(rho))*100),
            "segments":{k:tm2._safe_corr(b[m],rt[m]) for k,m in masks.items()},
            "duplicate_cv_r2":duplicate_r2(
                xt, t, ROOT/f"model/cat_{pack_tag}_base_s3.pkl")}


def main():
    outputs=pd.read_pickle(ROOT/"out/tm2command_outputs.pkl")
    result={"contract":{"features":COLS,"null_reps":400,"ridge_alpha":tm2.ALPHA},
            "transfers":[one(outputs,2022,2023,"BND22_base","BND22_cell","BND22"),
                         one(outputs,2023,2024,"B1J6_base","B1J6_cell","B1J6")]}
    p=ROOT/"out/tm_asof_disagreement_audit.json"
    p.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())

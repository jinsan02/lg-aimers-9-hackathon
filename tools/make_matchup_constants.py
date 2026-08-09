"""Freeze row-independent 2025 matchup residual maps from 2024 OOF predictions."""

from glob import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}
W, SLOPE, SHIFT = .55, 1.0416, .0052
MID_K = 500.
BO_K, PB_K = 3000., 500.
MID_COL = "asof_pitcher_prev5_game_middle_rate"


def ensemble(tag, seeds=None):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_val_preds.npz")))
    if seeds is not None:
        fs = [f for f in fs if f.rsplit("_s", 1)[1].split("_", 1)[0] in seeds]
    z = [np.load(f) for f in fs]
    if not z:
        raise FileNotFoundError(tag)
    return np.mean([q["pred"].astype(float) for q in z], 0), z[0]["y"].astype(float)


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


def fit_map(frame, cols, resid, k):
    tmp = frame[list(cols)].copy()
    tmp["_r"] = resid
    tab = tmp.groupby(list(cols), dropna=False)._r.agg(["sum", "size"]).reset_index()
    tab["offset"] = tab["sum"]/(tab["size"]+k)
    src = frame.merge(tab[list(cols)+["offset"]], on=list(cols), how="left")["offset"]
    mean = float(src.fillna(0.).mean())
    tab["offset"] -= mean
    return tab, src.fillna(0.).to_numpy(float)-mean, mean


def middle_adjustment(d, y, p):
    x = d[MID_COL].to_numpy(float)
    good = x[np.isfinite(x)]
    edges = np.unique(np.quantile(good, np.linspace(0, 1, 9)))
    edges[0], edges[-1] = -np.inf, np.inf
    bn = np.searchsorted(edges[1:-1], x, side="right")
    bn[~np.isfinite(x)] = -1
    r = y-p; r -= r.mean()
    tab = pd.DataFrame({"bin": bn, "r": r}).groupby("bin").r.agg(["sum", "size"])
    tab["offset"] = tab["sum"]/(tab["size"]+MID_K)
    tab["offset"] -= float(np.average(tab.offset, weights=tab["size"]))
    return pd.Series(bn).map(tab.offset).fillna(0.).to_numpy(float)


def main():
    b, y = ensemble("VB2_base")
    c, yc = ensemble("ZD5", CELL_SEEDS)
    if not np.array_equal(y, yc):
        raise ValueError("target mismatch")
    cols = ["season", MID_COL, "pitcher_id", "batter_id", "pitcher_team_id"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    d = d[d.season == 2024].reset_index(drop=True)
    if len(d) != len(y):
        raise ValueError(f"row mismatch {len(d)} != {len(y)}")
    p = post((1-W)*b+W*c)
    mid = middle_adjustment(d, y, p)
    base = p+mid

    r = y-base; r -= r.mean()
    bo, bo_src, bo_mean = fit_map(d, ["batter_id", "pitcher_team_id"], r, BO_K)
    pb0, pb0_src, pb0_mean = fit_map(d, ["pitcher_id", "batter_id"], r, PB_K)
    r2 = y-(base+bo_src); r2 -= r2.mean()
    pb, pb_src, pb_mean = fit_map(d, ["pitcher_id", "batter_id"], r2, PB_K)

    out = os.path.join(ROOT, "out", "matchup_constants_2024.npz")
    np.savez_compressed(out,
        bo_batter=bo.batter_id.to_numpy(np.int64),
        bo_team=bo.pitcher_team_id.to_numpy(np.int64),
        bo_offset=bo.offset.to_numpy(np.float64),
        pb_pitcher=pb.pitcher_id.to_numpy(np.int64),
        pb_batter=pb.batter_id.to_numpy(np.int64),
        pb_offset=pb.offset.to_numpy(np.float64),
        pb0_pitcher=pb0.pitcher_id.to_numpy(np.int64),
        pb0_batter=pb0.batter_id.to_numpy(np.int64),
        pb0_offset=pb0.offset.to_numpy(np.float64))
    meta = {
        "bo_k": BO_K, "pb_k": PB_K, "bo_groups": len(bo), "pb_groups": len(pb),
        "bo_source_mean_removed": bo_mean, "pb_source_mean_removed": pb_mean,
        "pb0_source_mean_removed": pb0_mean,
        "middle_raw_bss": score(y, base)-score(y, p),
        "bo_raw_increment": score(y, base+bo_src)-score(y, base),
        "pb_raw_increment": score(y, base+bo_src+pb_src)-score(y, base+bo_src),
        "pb0_raw_increment": score(y, base+pb0_src)-score(y, base),
        "combined_raw_increment": score(y, base+bo_src+pb_src)-score(y, base),
    }
    with open(os.path.join(ROOT, "out", "matchup_constants_2024.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"saved {out} ({os.path.getsize(out)/1e6:.3f} MB)")


if __name__ == "__main__":
    main()

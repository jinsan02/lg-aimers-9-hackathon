"""Freeze 2025 recent-middle correction from submission-equivalent 2024 residuals."""

from glob import glob
import json
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}
W = .55
SLOPE = 1.0416
SHIFT = .0052
K = 500.
Q = 8
COL = "asof_pitcher_prev5_game_middle_rate"


def ensemble(tag, seeds=None):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_val_preds.npz")))
    if seeds is not None:
        fs = [f for f in fs if f.split("_s")[-1].split("_")[0] in seeds]
    z = [np.load(f) for f in fs]
    if not z:
        raise FileNotFoundError(tag)
    return np.mean([q["pred"].astype(float) for q in z], 0), z[0]["y"].astype(float), fs


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    z = np.log(p/(1-p))
    return np.clip(1/(1+np.exp(-SLOPE*z))-SHIFT, 0, 1)


def bss(y, p):
    r = y.mean()
    return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(r*(1-r)))


def main():
    b, y, fb = ensemble("VB2_base")
    c, yc, fc = ensemble("ZD5", CELL_SEEDS)
    if not np.array_equal(y, yc):
        raise ValueError("target mismatch")
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=["season", COL])
    x = d.loc[d.season == 2024, COL].to_numpy(float)
    if len(x) != len(y):
        raise ValueError(f"row mismatch {len(x)} != {len(y)}")
    p = post((1-W)*b + W*c)
    good = x[np.isfinite(x)]
    edges = np.unique(np.quantile(good, np.linspace(0,1,Q+1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bn = np.searchsorted(edges[1:-1], x, side="right")
    bn[~np.isfinite(x)] = -1
    resid = y-p
    global_bias = float(resid.mean())
    resid -= global_bias
    tab = pd.DataFrame({"bin": bn, "r": resid}).groupby("bin").r.agg(["sum","size"])
    tab["offset"] = tab["sum"]/(tab["size"]+K)
    # The conditional map must not silently change the global SHIFT on its source.
    source_mean = float(np.average(tab.offset, weights=tab["size"]))
    tab["offset"] -= source_mean
    adj = pd.Series(bn).map(tab.offset).fillna(0).to_numpy(float)
    payload = {
        "column": COL, "q": Q, "k": K,
        "thresholds": [float(v) for v in edges[1:-1]],
        "nan_offset": float(tab.loc[-1, "offset"]) if -1 in tab.index else 0.,
        "offsets": [float(tab.loc[i, "offset"]) for i in range(Q)],
        "source_global_bias": global_bias,
        "source_adjustment_mean": float(adj.mean()),
        "base_bss": float(bss(y,p)), "self_fit_bss": float(bss(y,p+adj)),
        "base_models": [os.path.basename(f) for f in fb],
        "cell_models": [os.path.basename(f) for f in fc],
    }
    path = os.path.join(ROOT, "out", "middle_constants_2024.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("\nbin table")
    print(tab.to_string(float_format=lambda v: f"{v:+.8f}"))
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

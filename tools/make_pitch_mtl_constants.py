"""Freeze 2025 row-local corrections for the masked-pitch NN candidate.

The blend weight and shrinkage constants were fixed by the rolling experiment;
this script only refits the existing v11 recent-middle and exact-PB estimators
on the submission-equivalent 2024 source residuals.
"""

from __future__ import annotations

import argparse
from glob import glob
import json
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}
NN_SEEDS = {"3", "4", "5", "6", "8", "13"}
CELL_W, NN_W = 0.55, 0.10
SLOPE, SHIFT = 1.0416, 0.0052
MID_K, PB_K = 500.0, 500.0
MID_COL = "asof_pitcher_prev5_game_middle_rate"


def ensemble(pattern, seeds=None):
    paths = sorted(glob(os.path.join(ROOT, pattern)))
    if seeds is not None:
        paths = [p for p in paths if p.rsplit("_s", 1)[1].split("_", 1)[0]
                 in seeds]
    if not paths:
        raise FileNotFoundError(pattern)
    zs = [np.load(p) for p in paths]
    y = zs[0]["y"].astype(float)
    if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
        raise ValueError(f"target mismatch: {pattern}")
    return np.mean([z["pred"].astype(float) for z in zs], 0), y, paths


def post(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(1 / (1 + np.exp(-SLOPE * np.log(p / (1 - p)))) - SHIFT,
                   0, 1)


def bss(y, p):
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) /
                  (y.mean() * (1 - y.mean())))


def fit_middle(frame, y, p):
    x = frame[MID_COL].to_numpy(float)
    finite = np.isfinite(x)
    edges = np.quantile(x[finite], np.linspace(0, 1, 9))
    edges[0], edges[-1] = -np.inf, np.inf
    bins = np.searchsorted(edges[1:-1], x, side="right")
    bins[~finite] = -1
    resid = y - p
    resid -= resid.mean()
    tab = pd.DataFrame({"bin": bins, "r": resid}).groupby("bin").r.agg(
        ["sum", "size"])
    tab["offset"] = tab["sum"] / (tab["size"] + MID_K)
    tab["offset"] -= float(np.average(tab.offset, weights=tab["size"]))
    adj = pd.Series(bins).map(tab.offset).fillna(0).to_numpy(float)
    return edges[1:-1], tab, adj


def fit_pb(frame, resid):
    tmp = frame[["pitcher_id", "batter_id"]].copy()
    tmp["r"] = resid
    tab = tmp.groupby(["pitcher_id", "batter_id"]).r.agg(
        ["sum", "size"]).reset_index()
    tab["offset"] = tab["sum"] / (tab["size"] + PB_K)
    src = frame[["pitcher_id", "batter_id"]].merge(
        tab[["pitcher_id", "batter_id", "offset"]],
        on=["pitcher_id", "batter_id"], how="left")["offset"].fillna(0)
    mean = float(src.mean())
    tab["offset"] -= mean
    return tab, src.to_numpy(float) - mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model/pitch_mtl/corrections_2024.npz")
    args = ap.parse_args()
    base, y, base_paths = ensemble("out/cat_VB2_base_s*_val_preds.npz")
    cell, yc, cell_paths = ensemble("out/cat_ZD5_s*_val_preds.npz", CELL_SEEDS)
    nn, yn, nn_paths = ensemble(
        "out/mtnn_PMT1_aux6_fix3_s*_test_preds.npz", NN_SEEDS)
    if not (np.array_equal(y, yc) and np.array_equal(y, yn)):
        raise ValueError("base/cell/NN target mismatch")

    cols = ["season", MID_COL, "pitcher_id", "batter_id"]
    frame = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    frame = frame[frame.season == 2024].reset_index(drop=True)
    if len(frame) != len(y):
        raise ValueError(f"2024 row mismatch {len(frame)} != {len(y)}")

    core = (1 - CELL_W) * base + CELL_W * cell
    raw = (1 - NN_W) * core + NN_W * nn
    p = post(raw)
    edges, mid_tab, mid = fit_middle(frame, y, p)
    pm = np.clip(p + mid, 0, 1)
    resid = y - pm
    resid -= resid.mean()
    pb, pb_adj = fit_pb(frame, resid)
    q = np.clip(pm + pb_adj, 0, 1)

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    out = os.path.join(ROOT, args.out)
    np.savez_compressed(
        out,
        nn_weight=np.asarray(NN_W), cell_weight=np.asarray(CELL_W),
        slope=np.asarray(SLOPE), shift=np.asarray(SHIFT),
        middle_thresholds=np.asarray(edges, np.float64),
        middle_offsets=np.asarray([mid_tab.loc[i, "offset"] for i in range(8)]),
        middle_nan_offset=np.asarray(mid_tab.loc[-1, "offset"]
                                     if -1 in mid_tab.index else 0.0),
        pb_pitcher=pb.pitcher_id.to_numpy(np.int64),
        pb_batter=pb.batter_id.to_numpy(np.int64),
        pb_offset=pb.offset.to_numpy(np.float64),
    )
    meta = {
        "core_bss": bss(y, post(core)),
        "nn_blend_core_bss": bss(y, p),
        "middle_bss": bss(y, pm),
        "route_bss": bss(y, q),
        "nn_weight": NN_W, "cell_weight": CELL_W,
        "base_members": len(base_paths), "cell_members": len(cell_paths),
        "nn_members": len(nn_paths), "pb_groups": len(pb),
    }
    with open(out.replace(".npz", ".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"saved {out} ({os.path.getsize(out)/1e6:.3f} MB)")


if __name__ == "__main__":
    main()

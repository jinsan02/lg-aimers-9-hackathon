"""Evaluate the masked pitch auxiliary ensemble on the frozen v11 analogue.

The neural prediction is row-local and contains only its control head.  This
report never reads pitch labels: it combines already-saved held-out control
predictions with the exact v11 recent-middle + PB offline analogue.
"""

from __future__ import annotations

from glob import glob
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
W_CELL, SLOPE, SHIFT = 0.55, 1.0416, 0.0052
CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}
MID = "asof_pitcher_prev5_game_middle_rate"
THRESHOLDS = np.asarray([0.114286, 0.141026, 0.159091, 0.174312,
                         0.188889, 0.203837, 0.227273], float)
OFFSETS = np.asarray([0.009128596327376929, 0.0014582307357119428,
                      0.0016645796882829116, 0.0031305197564435523,
                      0.00028882666473504875, -0.005475006685108483,
                      -0.00413854957909932, -0.005074360453772326])
NAN_OFFSET = -0.008102692247033697


def ensemble(pattern: str, seeds: set[str] | None = None):
    fs = sorted(glob(os.path.join(OUT, pattern)))
    if seeds is not None:
        fs = [f for f in fs if f.split("_s")[-1].split("_")[0] in seeds]
    if not fs:
        raise FileNotFoundError(pattern)
    zs = [np.load(f) for f in fs]
    y = zs[0]["y"].astype(float)
    if any(not np.array_equal(z["y"], y) for z in zs[1:]):
        raise ValueError(f"target mismatch: {pattern}")
    return np.mean([z["pred"].astype(float) for z in zs], 0), y, len(fs)


def bss(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1-r)))


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.clip(1/(1+np.exp(-SLOPE*np.log(p/(1-p))))-SHIFT, 0, 1)


def corrections(d):
    x = pd.to_numeric(d[MID], errors="coerce").to_numpy(float)
    mid = np.full(len(d), NAN_OFFSET)
    ok = np.isfinite(x)
    mid[ok] = OFFSETS[np.searchsorted(THRESHOLDS, x[ok], side="right")]
    z = np.load(os.path.join(OUT, "matchup_constants_2024.npz"))
    tab = {(int(p), int(b)): float(v) for p, b, v in
           zip(z["pb0_pitcher"], z["pb0_batter"], z["pb0_offset"])}
    pb = np.fromiter((tab.get((int(p), int(b)), 0.0) for p, b in
                      zip(d.pitcher_id, d.batter_id)), float, count=len(d))
    return mid, pb


def main():
    base, y, nb = ensemble("cat_VB2_base_s*_val_preds.npz")
    cell, yc, nc = ensemble("cat_ZD5_s*_val_preds.npz", CELL_SEEDS)
    aux, yn, nn = ensemble("mtnn_PMT1_aux6_fix3_s*_test_preds.npz")
    if not (np.array_equal(y, yc) and np.array_equal(y, yn)):
        raise ValueError("v11/NN target mismatch")
    cols = ["season", "game_month", "game_type", MID, "pitcher_id", "batter_id"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    d = d[d.season == 2024].reset_index(drop=True)
    mid, pb = corrections(d)
    champ = np.clip(post((1-W_CELL)*base + W_CELL*cell) + mid + pb, 0, 1)
    masks = {
        "all": np.ones(len(y), bool),
        "R": d.game_type.eq("R").to_numpy(),
        "F": d.game_type.eq("F").to_numpy(),
        "early": d.game_month.le(6).to_numpy(),
        "late": d.game_month.gt(6).to_numpy(),
    }
    print(f"members base={nb} cell={nc} pitch_aux={nn} | "
          f"rms(aux,v11)={np.sqrt(np.mean((aux-champ)**2)):.7f}")
    print(f"v11={bss(y, champ):.3f} aux={bss(y, aux):.3f}")
    for w in (0.10, 0.15, 0.20, 0.25, 0.30):
        q = (1-w)*champ + w*aux
        vals = " ".join(f"{name}={bss(y[m],q[m])-bss(y[m],champ[m]):+.3f}"
                        for name, m in masks.items())
        print(f"w={w:.2f} {vals}")


if __name__ == "__main__":
    main()

"""Compare a base-member replacement under the exact frozen v11 recipe."""

import argparse
import glob
import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, SLOPE, SHIFT = .55, 1.0416, .0052
MID = "asof_pitcher_prev5_game_middle_rate"
THRESHOLDS = np.asarray([0.114286, 0.141026, 0.159091, 0.174312,
                         0.188889, 0.203837, 0.227273], float)
OFFSETS = np.asarray([0.009128596327376929, 0.0014582307357119428,
                      0.0016645796882829116, 0.0031305197564435523,
                      0.00028882666473504875, -0.005475006685108483,
                      -0.00413854957909932, -0.005074360453772326])
NAN_OFFSET = -0.008102692247033697


def load(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    z = [np.load(f) for f in fs]
    y = z[0]["y"].astype(float)
    return np.mean([q["pred"].astype(float) for q in z], 0), y, len(fs)


def post(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.clip(1/(1+np.exp(-SLOPE*np.log(p/(1-p))))-SHIFT, 0, 1)


def bss(y, p, center=False):
    p = np.asarray(p).copy()
    if center:
        p += y.mean()-p.mean()
    return 1e5*(1-np.mean((np.clip(p,0,1)-y)**2)/(y.mean()*(1-y.mean())))


def corrections(d, path):
    x = pd.to_numeric(d[MID], errors="coerce").to_numpy(float)
    mid = np.full(len(d), NAN_OFFSET)
    ok = np.isfinite(x)
    mid[ok] = OFFSETS[np.searchsorted(THRESHOLDS, x[ok], side="right")]
    z = np.load(path)
    tab = {(int(p), int(b)): float(v) for p, b, v in
           zip(z["pb0_pitcher"], z["pb0_batter"], z["pb0_offset"])}
    pb = np.fromiter((tab.get((int(p), int(b)), 0.) for p, b in
                      zip(d.pitcher_id, d.batter_id)), float, count=len(d))
    return mid, pb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--constants", default="out/matchup_constants_2024.npz")
    args = ap.parse_args()
    b, y, nb = load(args.base)
    a, ya, na = load(args.alt)
    c, yc, nc = load(args.cell)
    if not (np.array_equal(y, ya) and np.array_equal(y, yc)):
        raise ValueError("target mismatch")
    cols = ["season", "game_month", "game_type", MID, "pitcher_id", "batter_id"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    d = d[d.season == args.season].reset_index(drop=True)
    mid, pb = corrections(d, args.constants)
    p0, p1 = post((1-W)*b+W*c), post((1-W)*a+W*c)
    q0, q1 = np.clip(p0+mid+pb,0,1), np.clip(p1+mid+pb,0,1)
    masks = {"all": np.ones(len(y),bool), "R": d.game_type.eq("R").to_numpy(),
             "F": d.game_type.eq("F").to_numpy(), "early": d.game_month.le(6).to_numpy(),
             "late": d.game_month.gt(6).to_numpy()}
    print(f"models base={nb} alt={na} cell={nc} middle_sd={mid.std():.7f} "
          f"pb_sd={pb.std():.7f}")
    for name,m in masks.items():
        before, after = bss(y[m],q0[m]), bss(y[m],q1[m])
        cb, ca = bss(y[m],q0[m],True), bss(y[m],q1[m],True)
        print(f"{name:<6} v11 {before:9.3f}->{after:9.3f} {after-before:+8.3f} | "
              f"centered {cb:9.3f}->{ca:9.3f} {ca-cb:+8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

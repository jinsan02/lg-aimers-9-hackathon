"""Describe the exact v11 -> v12 frozen-adjustment change on the 2025 test.

This intentionally uses no leaderboard-derived fitting.  It only compares the
two already-submitted deterministic correction tables and reports where their
predictions differ.
"""

import os

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.environ.get("V12_TEST_CSV", os.path.join(ROOT, "data", "test.csv"))
OLD = os.environ.get("V12_OLD_CONSTANTS",
                     os.path.join(ROOT, "out", "matchup_constants_2024.npz"))
NEW = os.environ.get("V12_NEW_CONSTANTS",
                     os.path.join(ROOT, "out", "final_constants_2024.npz"))

PREV_THRESHOLDS = np.asarray([
    0.114286, 0.141026, 0.159091, 0.174312,
    0.188889, 0.203837, 0.227273,
])
PREV_OFFSETS = np.asarray([
    0.009128596327376929, 0.0014582307357119428,
    0.0016645796882829116, 0.0031305197564435523,
    0.00028882666473504875, -0.005475006685108483,
    -0.00413854957909932, -0.005074360453772326,
])
PREV_NAN = -0.008102692247033697


def binned(x, thresholds, offsets, nan_offset):
    x = pd.to_numeric(x, errors="coerce").to_numpy(np.float64)
    finite = np.isfinite(x)
    out = np.full(len(x), nan_offset, dtype=np.float64)
    out[finite] = offsets[np.searchsorted(thresholds, x[finite], side="right")]
    return out


def pair_map(d, z, prefix):
    table = {(int(p), int(b)): float(v) for p, b, v in zip(
        z[f"{prefix}_pitcher"], z[f"{prefix}_batter"], z[f"{prefix}_offset"]
    )}
    return np.fromiter(
        (table.get((int(p), int(b)), 0.0) for p, b in
         zip(d.pitcher_id, d.batter_id)),
        dtype=np.float64, count=len(d),
    )


def describe(name, x):
    q = np.quantile(x, [0, .01, .1, .5, .9, .99, 1])
    print(f"{name:>14}: mean={x.mean():+.7f} sd={x.std():.7f} "
          f"rms={np.sqrt(np.mean(x*x)):.7f} nonzero={(x != 0).mean():.3%}")
    print(" " * 16 + "q=" + ", ".join(f"{v:+.7f}" for v in q))


def main():
    cols = ["pitcher_id", "batter_id", "game_month", "game_type",
            "asof_pitcher_n", "asof_pitcher_middle_rate",
            "asof_pitcher_prev5_game_middle_rate"]
    d = pd.read_csv(TEST, usecols=cols)
    d["pitcher_id"] = pd.to_numeric(d.pitcher_id, errors="coerce").fillna(-1).astype(np.int64)
    d["batter_id"] = pd.to_numeric(d.batter_id, errors="coerce").fillna(-1).astype(np.int64)
    old = np.load(OLD)
    new = np.load(NEW)
    print("career thresholds:", new["thresholds"].tolist())
    print("career offsets:", new["offsets"].tolist(),
          "nan=", float(new["nan_offset"][0]))
    old_tab = {(int(p), int(b)): float(v) for p, b, v in zip(
        old["pb0_pitcher"], old["pb0_batter"], old["pb0_offset"])}
    new_tab = {(int(p), int(b)): float(v) for p, b, v in zip(
        new["pb_pitcher"], new["pb_batter"], new["pb_offset"])}
    common = sorted(set(old_tab) & set(new_tab))
    ov = np.asarray([old_tab[k] for k in common])
    nv = np.asarray([new_tab[k] for k in common])
    print(f"PB table common={len(common):,}/{len(old_tab):,}/{len(new_tab):,} "
          f"corr={np.corrcoef(ov, nv)[0,1]:.8f} "
          f"delta_rms={np.sqrt(np.mean((nv-ov)**2)):.8f} "
          f"delta_max={np.max(np.abs(nv-ov)):.8f}")

    prev_mid = binned(d.asof_pitcher_prev5_game_middle_rate,
                      PREV_THRESHOLDS, PREV_OFFSETS, PREV_NAN)
    career_mid = binned(d.asof_pitcher_middle_rate,
                        new["thresholds"], new["offsets"],
                        float(new["nan_offset"][0]))
    old_pb = pair_map(d, old, "pb0")
    new_pb = pair_map(d, new, "pb")
    mid_delta = career_mid - prev_mid
    pb_delta = new_pb - old_pb
    total = mid_delta + pb_delta

    for name, x in [("prev_mid", prev_mid), ("career_mid", career_mid),
                    ("old_pb", old_pb), ("new_pb", new_pb),
                    ("mid_delta", mid_delta), ("pb_delta", pb_delta),
                    ("total_delta", total)]:
        describe(name, x)
    print(f"corr(mid_delta,pb_delta)={np.corrcoef(mid_delta, pb_delta)[0,1]:+.5f}")

    frame = d.assign(mid_delta=mid_delta, pb_delta=pb_delta, total=total)
    frame["half"] = np.where(frame.game_month <= 6, "early", "late")
    frame["experience"] = pd.cut(frame.asof_pitcher_n,
                                 [-np.inf, 50, 200, 1000, np.inf],
                                 labels=["n<=50", "51-200", "201-1000", ">1000"])
    for key in ["game_type", "half", "experience"]:
        print(f"\nBY {key}")
        print(frame.groupby(key, observed=True).agg(
            n=("total", "size"), mid_mean=("mid_delta", "mean"),
            pb_mean=("pb_delta", "mean"), total_mean=("total", "mean"),
            total_sd=("total", "std"), total_rms=("total", lambda x: np.sqrt(np.mean(x*x))),
        ).to_string())


if __name__ == "__main__":
    main()

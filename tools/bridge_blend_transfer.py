"""Frozen F-to-R bridge transfer on the current base+cell submission surface.

The bridge is estimated only from 2023 R holdout predictions.  Two constants
whose weighted mean is zero on the *source* R population are then frozen and
applied to 2024 rows.  Nothing is re-centred from the target batch, so this is
a row-independent post-model correction that could be shipped unchanged.
"""

from glob import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import roster_transition as rt

TARGET = "control_success"
CELL_SEEDS = {"42", "7", "13", "3", "4", "5"}
W_BASE = 0.45
W_CELL = 0.55
SLOPE = 1.0416
SHIFT = 0.0052


def load_ensemble(pattern, seeds=None):
    paths = sorted(glob(pattern))
    if seeds is not None:
        paths = [p for p in paths
                 if p.split("_s")[-1].split("_")[0] in seeds]
    if not paths:
        raise FileNotFoundError(pattern)
    arrays = [np.load(p) for p in paths]
    y = arrays[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, z["y"]) for z in arrays[1:]):
        raise ValueError(f"target mismatch: {pattern}")
    p = np.mean([z["pred"].astype(np.float64) for z in arrays], axis=0)
    return p, y, paths


def bss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 0, 1)
    r = float(y.mean())
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def post(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return np.clip(1 / (1 + np.exp(-SLOPE * z)) - SHIFT, 0, 1)


def centered(y, p):
    return bss(y, np.asarray(p) + np.mean(y) - np.mean(p))


def prepare():
    cols = ["row_id", "season", "game_month", "game_type", "pitcher_id", TARGET]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    table = rt.build_table(d)
    d, _ = rt.add_features(d, table)
    status = np.full(len(d), "other", dtype=object)
    status[d.rt_same_cont.to_numpy() == 1] = "same_cont"
    status[d.rt_f_to_r.to_numpy() == 1] = "F_to_R"
    d["status"] = status
    return d


def frozen_offsets(source):
    f = source.status.to_numpy() == "F_to_R"
    c = source.status.to_numpy() == "same_cont"
    residual = source[TARGET].to_numpy() - source.pred.to_numpy()
    gap = float(residual[f].mean() - residual[c].mean())
    share = float(f.mean())
    # Zero mean on source R. These constants remain frozen on every target row.
    a_f = gap * (1 - share)
    a_other = -gap * share
    return gap, share, a_f, a_other


def adjustment(frame, a_f, a_other):
    is_r = frame.game_type.to_numpy() == "R"
    is_ftr = frame.status.to_numpy() == "F_to_R"
    return np.where(is_r, np.where(is_ftr, a_f, a_other), 0.0)


def score_line(label, y, cur, cand):
    s0, s1 = bss(y, cur), bss(y, cand)
    c0, c1 = centered(y, cur), centered(y, cand)
    print(f"{label:<18} raw {s0:9.3f} -> {s1:9.3f} ({s1-s0:+7.3f}) | "
          f"centered {c0:9.3f} -> {c1:9.3f} ({c1-c0:+7.3f})")
    return s1 - s0


def main():
    d = prepare()
    p23, y23, src_paths = load_ensemble(
        os.path.join(ROOT, "out", "cat_H3_base_s*_val_preds.npz"))
    p24, y24, base_paths = load_ensemble(
        os.path.join(ROOT, "out", "cat_VB2_base_s*_val_preds.npz"))
    cell24, yc, cell_paths = load_ensemble(
        os.path.join(ROOT, "out", "cat_ZD5_s*_val_preds.npz"), CELL_SEEDS)

    source = d[(d.season == 2023) & (d.game_type == "R")].copy()
    target = d[d.season == 2024].copy()
    if len(source) != len(p23) or not np.array_equal(source[TARGET].to_numpy(), y23):
        raise ValueError("H3 prediction order does not match 2023 R")
    if len(target) != len(p24) or not np.array_equal(target[TARGET].to_numpy(), y24):
        raise ValueError("VB2 prediction order does not match 2024")
    if not np.array_equal(y24, yc):
        raise ValueError("VB2/ZD5 target mismatch")
    source["pred"] = p23

    gap, share, a_f, a_other = frozen_offsets(source)
    adj = adjustment(target, a_f, a_other)
    cur_pre = W_BASE * p24 + W_CELL * cell24
    cand_pre = W_BASE * (p24 + adj) + W_CELL * cell24

    print("=== BR1 frozen base-component F-to-R bridge ===")
    print(f"source H3 seeds={len(src_paths)} | target base={len(base_paths)} "
          f"cell={len(cell_paths)}")
    print(f"source conditional residual gap={gap:+.6f} | F_to_R share(R)={share:.6f}")
    print(f"frozen base offsets: F_to_R={a_f:+.6f}, other_R={a_other:+.6f}, F=0")
    print(f"target bridge mean={adj.mean():+.8f}; base weight makes "
          f"blend delta mean={(W_BASE*adj).mean():+.8f}")
    print()
    score_line("base only", y24, p24, p24 + adj)
    score_line("blend pre-post", y24, cur_pre, cand_pre)
    gain = score_line("blend slope+shift", y24, post(cur_pre), post(cand_pre))

    for label, mask in (("Mar-Jun", target.game_month <= 6),
                        ("Jul-Oct", target.game_month >= 7),
                        ("R only", target.game_type == "R"),
                        ("F only", target.game_type == "F")):
        m = mask.to_numpy()
        score_line(label, y24[m], post(cur_pre[m]), post(cand_pre[m]))

    # Four common seeds give a small robustness check without tuning the bridge.
    common = ["42", "7", "13", "3"]
    seed_gains = []
    for seed in common:
        hs = np.load(os.path.join(ROOT, "out", f"cat_H3_base_s{seed}_val_preds.npz"))["pred"]
        bs = np.load(os.path.join(ROOT, "out", f"cat_VB2_base_s{seed}_val_preds.npz"))["pred"]
        cs = np.load(os.path.join(ROOT, "out", f"cat_ZD5_s{seed}_val_preds.npz"))["pred"]
        source["pred"] = hs
        _, _, sf, so = frozen_offsets(source)
        sa = adjustment(target, sf, so)
        q0 = post(W_BASE * bs + W_CELL * cs)
        q1 = post(W_BASE * (bs + sa) + W_CELL * cs)
        seed_gains.append(bss(y24, q1) - bss(y24, q0))
    mean = float(np.mean(seed_gains))
    se = float(np.std(seed_gains, ddof=1) / np.sqrt(len(seed_gains)))
    t = mean / se if se else np.inf
    print(f"\ncommon-seed gains {dict(zip(common, np.round(seed_gains, 3)))}")
    print(f"mean={mean:+.3f}, SE={se:.3f}, t={t:.3f}")
    passed = gain >= 3 and mean > 0
    print(f"\nGATE {'PASS' if passed else 'FAIL'}: fixed legal gain={gain:+.3f}; "
          "requires ensemble +3 and positive seed mean")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

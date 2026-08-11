"""EV1: transfer audit for within-family CatBoost seed variance.

The source (2023 R) and target (2024) use the same four common random seeds.
Only the source labels define uncertainty bins and corrections.  Target rows
are scored independently; no target-batch statistic is used at inference.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pbmf_transfer_audit import bss, fit_map, middle_keys, pair_keys, post


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 7, 13, 3)
ALL_BASE = (42, 7, 13, 3, 4, 5, 6, 8)
CELL = (42, 7, 13, 3, 4, 5)
MID = "asof_pitcher_prev5_game_middle_rate"


def load_seed_matrix(stem, seeds):
    zs = [np.load(ROOT / "out" / f"cat_{stem}_s{s}_val_preds.npz") for s in seeds]
    y = zs[0]["y"].astype(float)
    if any(not np.array_equal(y, z["y"]) for z in zs[1:]):
        raise ValueError(f"target mismatch within {stem}")
    return y, np.stack([z["pred"].astype(float) for z in zs])


def frozen_k0(src, tgt, ys, ps, pt):
    ms, mt = middle_keys(src[MID].to_numpy(float), tgt[MID].to_numpy(float))
    r = ys - ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.0)
    r = ys - (ps + ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.0)
    return np.clip(ps + ma + ea, 0, 1), np.clip(pt + mb + eb, 0, 1)


def source_edges(x, q=8):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        raise ValueError("degenerate uncertainty distribution")
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def qcode(x, edges):
    return np.searchsorted(edges[1:-1], x, side="right")


def report(label, frame, y, before, after):
    masks = {
        "all": np.ones(len(frame), bool),
        "R": frame.game_type.eq("R").to_numpy(),
        "F": frame.game_type.eq("F").to_numpy(),
        "early": frame.game_month.le(6).to_numpy(),
        "late": frame.game_month.gt(6).to_numpy(),
    }
    vals = {name: bss(y[m], after[m]) - bss(y[m], before[m])
            for name, m in masks.items() if m.sum() >= 100}
    print(f"{label:<24} " + " ".join(f"{k}={v:+.3f}" for k, v in vals.items()))
    return vals


def main():
    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id", MID]
    data = pd.read_csv(ROOT / "data" / "train.csv", usecols=cols)
    src = data[(data.season.eq(2023)) & data.game_type.eq("R")].reset_index(drop=True)
    tgt = data[data.season.eq(2024)].reset_index(drop=True)

    ys, hs = load_seed_matrix("H3_base", SEEDS)
    yt, vb_common = load_seed_matrix("VB2_base", SEEDS)
    yt2, vb_all = load_seed_matrix("VB2_base", ALL_BASE)
    ytc, cell = load_seed_matrix("ZD5", CELL)
    if not (len(src) == len(ys) and len(tgt) == len(yt)):
        raise ValueError("frame/prediction row mismatch")
    if not (np.array_equal(yt, yt2) and np.array_equal(yt, ytc)):
        raise ValueError("target mismatch across target members")

    ps = post(hs.mean(axis=0))
    base_t = vb_all.mean(axis=0)
    core_t = post(.45 * base_t + .55 * cell.mean(axis=0))
    k0s, k0t = frozen_k0(src, tgt, ys, ps, core_t)

    # Same four seeds on both seasons; population std keeps the definition fixed.
    us = hs.std(axis=0)
    ut = vb_common.std(axis=0)
    print("=== EV1 within-base seed variance ===")
    print(f"rows source/target={len(src):,}/{len(tgt):,}; seeds={SEEDS}")
    print(f"uncertainty mean source={us.mean():.6f} target={ut.mean():.6f}; "
          f"p90={np.quantile(us,.9):.6f}/{np.quantile(ut,.9):.6f}")
    print(f"K0 target BSS={bss(yt, k0t):.3f}")

    residual = ys - k0s
    residual -= residual.mean()
    edges = source_edges(us, 8)
    ks, kt = qcode(us, edges), qcode(ut, edges)
    _, qadj = fit_map(ks, kt, residual, 5000.0)
    for scale in (.25, .5, 1.0):
        cand = np.clip(k0t + scale * qadj, 0, 1)
        report(f"q8 k5000 x{scale:g}", tgt, yt, k0t, cand)

    # One-degree structural shrink: uncertain rows move toward 0.5.  The
    # coefficient is fitted on source only with a strong ridge denominator.
    us_norm = us / max(float(np.median(us)), 1e-8)
    ut_norm = ut / max(float(np.median(us)), 1e-8)
    xs = us_norm * (.5 - k0s)
    xt = ut_norm * (.5 - k0t)
    alpha = float(np.dot(xs, residual) / (np.dot(xs, xs) + 1000.0))
    print(f"source shrink coefficient={alpha:+.6f}")
    for scale in (.25, .5, 1.0):
        cand = np.clip(k0t + scale * alpha * xt, 0, 1)
        report(f"shrink x{scale:g}", tgt, yt, k0t, cand)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

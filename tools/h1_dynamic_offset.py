"""H1: previous-season pitcher/batter residual state applied one year ahead.

The transition hyperparameters are selected only on 2022 -> 2023 R-league
transfer, then frozen for the single 2023 -> 2024 evaluation.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def bss(y, p):
    r = float(y.mean())
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1-r)))


def center_score(y, p):
    return bss(y, p - (p.mean() - y.mean()))


def effect(frame, key, k):
    x = frame.copy()
    x["resid_c"] = x["resid"] - x["resid"].mean()
    g = x.groupby(key, observed=True).resid_c.agg(["sum", "size"])
    return g["sum"] / (g["size"] + float(k))


def features(frame, state_p, state_b):
    return np.column_stack([
        frame.pitcher_id.map(state_p).fillna(0).to_numpy(float),
        frame.batter_id.map(state_b).fillna(0).to_numpy(float),
    ])


def fit_beta(X, residual, alpha=1e-3):
    y = residual - residual.mean()
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base22", default="out/cat_H1B22_val_preds.npz")
    ap.add_argument("--base23", default="out/cat_AB_base_s3_val_preds.npz")
    ap.add_argument("--target24", default="out/cat_AB_base_s3_test_preds.npz")
    args = ap.parse_args()

    z22, z23, z24 = (np.load(args.base22, allow_pickle=True),
                     np.load(args.base23, allow_pickle=True),
                     np.load(args.target24, allow_pickle=True))
    raw = pd.read_csv("data/train.csv", encoding="utf-8-sig",
                      usecols=["row_id", "season", "game_type", "pitcher_id",
                               "batter_id", "control_success"])
    m22 = raw[(raw.season == 2022) & (raw.game_type == "R")].reset_index(drop=True)
    m23 = raw[raw.season == 2023].reset_index(drop=True)
    m24 = pd.DataFrame({"row_id": z24["row_id"]}).merge(
        raw, on="row_id", how="left", validate="one_to_one")
    if len(m22) != len(z22["y"]) or len(m23) != len(z23["y"]):
        raise ValueError(f"row alignment: 2022 {len(m22)}/{len(z22['y'])}, "
                         f"2023 {len(m23)}/{len(z23['y'])}")
    for frame, z in ((m22, z22), (m23, z23), (m24, z24)):
        frame["y"] = z["y"].astype(float)
        frame["pred"] = z["pred"].astype(float)
        frame["resid"] = frame.y - frame.pred

    s22, s23 = m22, m23[m23.game_type == "R"].copy()
    t24 = m24
    base_raw, base_ctr = bss(t24.y.to_numpy(), t24.pred.to_numpy()), \
                         center_score(t24.y.to_numpy(), t24.pred.to_numpy())
    rows = []
    for kp in (30, 100, 300, 1000):
        for kb in (100, 300, 1000, 3000):
            p22, b22 = effect(s22, "pitcher_id", kp), effect(s22, "batter_id", kb)
            X23 = features(s23, p22, b22)
            beta = fit_beta(X23, s23.resid.to_numpy())
            adj23 = X23 @ beta
            gain23 = center_score(s23.y.to_numpy(), s23.pred.to_numpy() + adj23) - \
                     center_score(s23.y.to_numpy(), s23.pred.to_numpy())
            rows.append((gain23, kp, kb, beta))
    gain23, kp, kb, beta = max(rows, key=lambda x: x[0])

    p22, b22 = effect(s22, "pitcher_id", kp), effect(s22, "batter_id", kb)
    p23, b23 = effect(s23, "pitcher_id", kp), effect(s23, "batter_id", kb)
    cp = p22.rename("a").to_frame().join(p23.rename("b"), how="inner")
    cb = b22.rename("a").to_frame().join(b23.rename("b"), how="inner")
    X24 = features(t24, p23, b23)
    adj = X24 @ beta
    # F had no comparable post-regime source in 2022; do not extrapolate R states to it.
    adj[t24.game_type.to_numpy() != "R"] = 0.0
    pred = t24.pred.to_numpy() + adj
    raw_gain = bss(t24.y.to_numpy(), pred) - base_raw
    ctr_gain = center_score(t24.y.to_numpy(), pred) - base_ctr
    print(f"selected on 2022->2023R: kp={kp} kb={kb} "
          f"beta_p={beta[0]:+.4f} beta_b={beta[1]:+.4f} source_gain={gain23:+.3f}")
    print(f"state corr pitcher={cp.a.corr(cp.b):+.4f} n={len(cp)} "
          f"batter={cb.a.corr(cb.b):+.4f} n={len(cb)}")
    print(f"2024 hit pitcher={t24.pitcher_id.isin(p23.index).mean():.1%} "
          f"batter={t24.batter_id.isin(b23.index).mean():.1%} adj_sd={adj.std():.6f}")
    print(f"2024 base raw={base_raw:.3f} centered={base_ctr:.3f}")
    print(f"2024 H1   raw={bss(t24.y.to_numpy(), pred):.3f} "
          f"centered={center_score(t24.y.to_numpy(), pred):.3f} "
          f"gain_raw={raw_gain:+.3f} gain_centered={ctr_gain:+.3f}")
    np.savez_compressed("out/H1_dynamic_test_preds.npz", y=t24.y.to_numpy(), pred=pred,
                        row_id=t24.row_id.to_numpy(), adj=adj, beta=beta,
                        kp=np.array(kp), kb=np.array(kb))


if __name__ == "__main__":
    main()

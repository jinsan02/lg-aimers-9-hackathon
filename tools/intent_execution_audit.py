"""Cheap rolling gate for a soft intent x execution bilinear residual."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbmf_transfer_audit as core
from state_workload_audit import k0_route


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = [
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]
RANK = 4


def context_code(d):
    balls = d.balls_before.to_numpy()
    strikes = d.strikes_before.to_numpy()
    fam = np.select([balls == 3, (strikes == 2) & (balls < 3), balls == strikes],
                    [0, 1, 2], default=3).astype(int)
    runner = (d.num_runners_on.to_numpy() > 0).astype(int)
    same = (d.pitcher_hand.astype(str).to_numpy() ==
            d.batter_hand.astype(str).to_numpy()).astype(int)
    return fam*4 + runner*2 + same


def design(src, tgt):
    xs = src[STATE].to_numpy(float)
    xt = tgt[STATE].to_numpy(float)
    med = np.nanmedian(xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    xs, xt = np.where(np.isfinite(xs), xs, med), np.where(np.isfinite(xt), xt, med)
    mu, sd = xs.mean(0), xs.std(0)
    sd[sd < 1e-8] = 1.
    xs, xt = (xs-mu)/sd, (xt-mu)/sd
    cs, ct = context_code(src), context_code(tgt)
    ohs = np.eye(16, dtype=np.float64)[cs]
    oht = np.eye(16, dtype=np.float64)[ct]
    zs = np.einsum("ni,nj->nij", ohs, xs).reshape(len(src), -1)
    zt = np.einsum("ni,nj->nij", oht, xt).reshape(len(tgt), -1)
    return zs, zt, ohs, oht, xs, xt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--drop-f-pre", type=int, default=0)
    args = ap.parse_args()

    bv, ys, _ = core.ensemble(args.base, "val")
    cv, ysc, _ = core.ensemble(args.cell, "val")
    bt, yt, _ = core.ensemble(args.base, "test")
    ct, ytc, _ = core.ensemble(args.cell, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)):
        raise ValueError("target mismatch")
    ps = core.post((1-core.W_CELL)*bv+core.W_CELL*cv)
    pt = core.post((1-core.W_CELL)*bt+core.W_CELL*ct)

    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id",
            "balls_before", "strikes_before", "num_runners_on",
            "pitcher_hand", "batter_hand"] + STATE
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    if args.drop_f_pre:
        d = d[~((d.game_type == "F") & (d.season <= args.drop_f_pre))]
    src = d[d.season == args.source].reset_index(drop=True)
    tgt = d[d.season == args.target].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError("row mismatch")
    k0s, k0t = k0_route(src, tgt, ys, yt, ps, pt)
    resid = ys-k0s; resid -= resid.mean()
    zs, zt, ohs, oht, xs, xt = design(src, tgt)

    ridge = Ridge(alpha=float(len(src)), fit_intercept=False)
    ridge.fit(zs, resid)
    coef = ridge.coef_.reshape(16, len(STATE))
    u, s, vt = np.linalg.svd(coef, full_matrices=False)
    coef4 = (u[:, :RANK]*s[:RANK]) @ vt[:RANK]
    a = np.sum((ohs@coef4)*xs, axis=1)
    b = np.sum((oht@coef4)*xt, axis=1)
    mean = float(a.mean()); a -= mean; b -= mean

    print(f"surface {args.source}->{args.target} rank={RANK} "
          f"coef_norm={np.linalg.norm(coef4):.6f} adj_sd={b.std():.6f}")
    core.report("intent_execution", tgt, yt, k0t, k0t+b)
    print(f"source diagnostic={core.bss(ys,k0s+a)-core.bss(ys,k0s):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Rolling audit of a low-capacity posterior-uncertainty residual head.

The champion already exposes current-season rates to CatBoost.  This audit
instead treats current-season success/middle counts as binomial posteriors and
uses their posterior uncertainty to modulate the champion probability.  The
head is a strongly regularized, source-fitted, zero-mean Ridge correction;
there is no global intercept/slope calibration and no target-season fitting.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbmf_transfer_audit as core


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES = ["success", "middle"]


def add_posterior_state(df, k=80.0):
    d = df.copy()
    ncol = "asof_pitcher_n"
    rcols = [f"asof_pitcher_{x}_rate" for x in RATES]
    last = (d[["pitcher_id", "season", ncol] + rcols]
            .sort_values(ncol).groupby(["pitcher_id", "season"], sort=False)
            .tail(1).sort_values(["pitcher_id", "season"]))
    anchor = last[["pitcher_id", "season", ncol]].copy()
    anchor["n0"] = anchor[ncol]
    for c in rcols:
        anchor[f"S0_{c}"] = anchor[ncol] * last[c]
    anchor["season"] += 1
    anchor = anchor.drop(columns=[ncol])
    d = d.merge(anchor, on=["pitcher_id", "season"], how="left")
    n = d[ncol].to_numpy(float)
    n0 = d.n0.fillna(0).to_numpy(float)
    sn = np.maximum(n-n0, 0)
    d["post_log_precision"] = np.log1p(sn+k)
    d["post_confidence"] = sn/(sn+k)
    for stem, c in zip(RATES, rcols):
        total = n*d[c].fillna(.5).to_numpy(float)
        s0 = d[f"S0_{c}"].fillna(0).to_numpy(float)
        ss = np.clip(total-s0, 0, sn)
        prior = d[c].fillna(.5).to_numpy(float)
        mu = (ss+k*prior)/(sn+k)
        d[f"post_{stem}_mean"] = mu
        d[f"post_{stem}_sd"] = np.sqrt(np.maximum(mu*(1-mu)/(sn+k+1), 0))
        d[f"post_{stem}_delta"] = mu-prior
    return d


def matrix(d, p0):
    pc = np.asarray(p0, float)-.5
    raw = np.column_stack([
        d.post_success_mean, d.post_success_sd, d.post_success_delta,
        d.post_middle_mean, d.post_middle_sd, d.post_middle_delta,
        d.post_log_precision, d.post_confidence,
        pc*d.post_success_sd, pc*d.post_middle_sd,
        pc*d.post_confidence,
    ]).astype(float)
    names = ["success_mean", "success_sd", "success_delta", "middle_mean",
             "middle_sd", "middle_delta", "log_precision", "confidence",
             "p_x_success_sd", "p_x_middle_sd", "p_x_confidence"]
    return raw, names


def fit_adjust(xs, xt, residual):
    med = np.nanmedian(xs, axis=0)
    med = np.where(np.isfinite(med), med, 0)
    xs = np.where(np.isfinite(xs), xs, med)
    xt = np.where(np.isfinite(xt), xt, med)
    mu, sd = xs.mean(0), xs.std(0)
    sd[sd < 1e-8] = 1
    xs, xt = (xs-mu)/sd, (xt-mu)/sd
    y = residual-residual.mean()
    model = Ridge(alpha=float(len(xs)), fit_intercept=False)
    model.fit(xs, y)
    a, b = model.predict(xs), model.predict(xt)
    return a-a.mean(), b-a.mean(), model.coef_


def k0_route(src, tgt, ys, ps, pt):
    ms, mt = core.middle_keys(src[core.MID].to_numpy(float),
                              tgt[core.MID].to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = core.fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = core.fit_map(core.pair_keys(src), core.pair_keys(tgt), r, 500.)
    return ps+ma+ea, pt+mb+eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--source", type=int, required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--drop-f-pre", type=int, default=0)
    args = ap.parse_args()

    bv, ys, nb = core.ensemble(args.base, "val")
    cv, ysc, nc = core.ensemble(args.cell, "val")
    bt, yt, _ = core.ensemble(args.base, "test")
    ct, ytc, _ = core.ensemble(args.cell, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)):
        raise ValueError("base/cell target mismatch")
    ps = core.post((1-core.W_CELL)*bv+core.W_CELL*cv)
    pt = core.post((1-core.W_CELL)*bt+core.W_CELL*ct)

    cols = ["season", "game_month", "game_type", "pitcher_id", "batter_id",
            "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate", core.MID]
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    full = add_posterior_state(full)
    if args.drop_f_pre:
        full = full[~((full.game_type == "F") &
                      (full.season <= args.drop_f_pre))]
    src = full[full.season == args.source].reset_index(drop=True)
    tgt = full[full.season == args.target].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError(f"row mismatch {len(src)}/{len(tgt)} vs {len(ys)}/{len(yt)}")
    k0s, k0t = k0_route(src, tgt, ys, ps, pt)
    xs, names = matrix(src, k0s)
    xt, _ = matrix(tgt, k0t)
    a, b, coef = fit_adjust(xs, xt, ys-k0s)
    print(f"surface {args.source}->{args.target} base={args.base}({nb}) "
          f"cell={args.cell}({nc}) rows={len(src):,}/{len(tgt):,}")
    core.report("posterior-head", tgt, yt, k0t, np.clip(k0t+b, 0, 1))
    # A negative infinitesimal derivative rules out the explanation that the
    # head merely needs more shrinkage.  Fixed scales are diagnostics, not
    # target-selected candidate weights.
    for w in (.05, .10, .25):
        core.report(f"posterior-w{w:g}", tgt, yt, k0t,
                    np.clip(k0t+w*b, 0, 1))
    err = yt-k0t
    direction = float(np.mean(err*b))
    print(f"target_direction_E[(y-p)*adj]={direction:+.9g}")
    top = sorted(zip(names, coef), key=lambda z: abs(z[1]), reverse=True)[:6]
    print("coefficients " + ", ".join(f"{n}={v:+.7f}" for n, v in top))
    print(f"source_gain={core.bss(ys, np.clip(k0s+a,0,1))-core.bss(ys,k0s):+.3f} "
          f"source_sd={a.std():.7f} target_sd={b.std():.7f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

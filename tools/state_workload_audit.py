"""Rolling transfer audit for recent-state and workload-pace residuals."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbmf_transfer_audit as core


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATE_COLS = [
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
]


def add_season_usage(d):
    out = d.copy()
    last = (out[["pitcher_id", "season", "asof_pitcher_n"]]
            .sort_values("asof_pitcher_n")
            .groupby(["pitcher_id", "season"], sort=False).tail(1)
            .sort_values(["pitcher_id", "season"]))
    last["season_total_n"] = (last["asof_pitcher_n"]
                              - last.groupby("pitcher_id")["asof_pitcher_n"].shift(1).fillna(0))
    prev = last[["pitcher_id", "season", "asof_pitcher_n", "season_total_n"]].copy()
    prev["season"] += 1
    prev = prev.rename(columns={"asof_pitcher_n": "season_start_n",
                                "season_total_n": "prior_season_n"})
    out = out.merge(prev, on=["pitcher_id", "season"], how="left")
    out["current_season_n"] = np.maximum(
        out.asof_pitcher_n-out.season_start_n.fillna(0), 0)

    role = (out.groupby(["pitcher_id", "season"])
            .agg(prior_inning_med=("inning", "median"),
                 prior_start_share=("inning", lambda x: np.mean(np.asarray(x) == 1)),
                 prior_late_share=("inning", lambda x: np.mean(np.asarray(x) >= 8)))
            .reset_index())
    role["season"] += 1
    out = out.merge(role, on=["pitcher_id", "season"], how="left")

    progress = np.clip((out.game_month.to_numpy(float)-2.5)/7.0, .08, 1.0)
    sn = out.current_season_n.to_numpy(float)
    pn = out.prior_season_n.to_numpy(float)
    out["workload_pace"] = np.log1p(sn)-np.log(progress)
    out["workload_vs_prior"] = np.log1p(sn)-np.log1p(np.maximum(pn, 0)*progress)
    out["role_inning_gap"] = out.inning-out.prior_inning_med
    return out


def recent_matrix(d, stems=("middle", "success")):
    vals = []
    names = []
    for stem in stems:
        a = d[f"asof_pitcher_prev1_game_{stem}_rate"].to_numpy(float)
        b = d[f"asof_pitcher_prev3_game_{stem}_rate"].to_numpy(float)
        c = d[f"asof_pitcher_prev5_game_{stem}_rate"].to_numpy(float)
        # Official windows are nested; these are equal-game-weight disjoint blocks.
        x23, x45 = (3*b-a)/2, (5*c-3*b)/2
        for nm, x in ((f"{stem}_g1", a), (f"{stem}_g23", x23),
                      (f"{stem}_g45", x45), (f"{stem}_short_long", a-c),
                      (f"{stem}_mid_long", b-c)):
            vals.append(x); names.append(nm)
        vals.append(np.nanstd(np.column_stack([a, x23, x45]), axis=1))
        names.append(f"{stem}_state_sd")
    return np.column_stack(vals), names


def workload_matrix(d):
    cols = ["workload_pace", "workload_vs_prior", "prior_season_n",
            "prior_start_share", "prior_late_share", "role_inning_gap"]
    return d[cols].to_numpy(float), cols


def ridge_adjust(xs, xt, resid):
    med = np.nanmedian(xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    xs = np.where(np.isfinite(xs), xs, med)
    xt = np.where(np.isfinite(xt), xt, med)
    mu, sd = xs.mean(0), xs.std(0)
    sd[sd < 1e-8] = 1.
    xs, xt = (xs-mu)/sd, (xt-mu)/sd
    model = Ridge(alpha=float(len(xs)), fit_intercept=True)
    model.fit(xs, resid)
    a, b = model.predict(xs), model.predict(xt)
    mean = float(a.mean())
    return a-mean, b-mean, model.coef_


def k0_route(src, tgt, ys, yt, ps, pt):
    ms, mt = core.middle_keys(src[core.MID].to_numpy(float), tgt[core.MID].to_numpy(float))
    r = ys-ps; r -= r.mean()
    ma, mb = core.fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma); r -= r.mean()
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

    bv, ys, _ = core.ensemble(args.base, "val")
    cv, ysc, _ = core.ensemble(args.cell, "val")
    bt, yt, _ = core.ensemble(args.base, "test")
    ct, ytc, _ = core.ensemble(args.cell, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)):
        raise ValueError("target mismatch")
    ps = core.post((1-core.W_CELL)*bv+core.W_CELL*cv)
    pt = core.post((1-core.W_CELL)*bt+core.W_CELL*ct)

    cols = ["season", "game_month", "game_type", "inning", "pitcher_id",
            "batter_id", "asof_pitcher_n", core.MID] + RATE_COLS
    full = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=cols)
    full = add_season_usage(full)
    if args.drop_f_pre:
        full = full[~((full.game_type == "F") & (full.season <= args.drop_f_pre))]
    src = full[full.season == args.source].reset_index(drop=True)
    tgt = full[full.season == args.target].reset_index(drop=True)
    if len(src) != len(ys) or len(tgt) != len(yt):
        raise ValueError(f"row mismatch {len(src)}/{len(tgt)} vs {len(ys)}/{len(yt)}")
    k0s, k0t = k0_route(src, tgt, ys, yt, ps, pt)
    resid = ys-k0s; resid -= resid.mean()

    print(f"surface {args.source}->{args.target} rows={len(src):,}/{len(tgt):,}")
    makers = (
        ("recent_middle", lambda d: recent_matrix(d, ("middle",))),
        ("recent_success", lambda d: recent_matrix(d, ("success",))),
        ("recent_combined", recent_matrix),
        ("workload_pace", workload_matrix),
    )
    for name, maker in makers:
        xs, names = maker(src); xt, _ = maker(tgt)
        a, b, coef = ridge_adjust(xs, xt, resid)
        core.report(name, tgt, yt, k0t, k0t+b)
        top = sorted(zip(names, coef), key=lambda z: abs(z[1]), reverse=True)[:4]
        print("  coef " + ", ".join(f"{n}={v:+.6f}" for n, v in top)
              + f" adj_sd={b.std():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

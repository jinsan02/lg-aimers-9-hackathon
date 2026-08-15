"""Transfer audit for base-vs-cell disagreement residual structure."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from invalidated import guard as _guard_invalidated                # noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd

from pbmf_transfer_audit import fit_map, middle_keys, pair_keys, post


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("signed_q8", "abs_q8", "sign_abs_q4", "p_q4_x_signed_q4")


def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0, 1)
    r = y.mean()
    return 1e5*(1-np.mean((p-y)**2)/(r*(1-r)))


def load(stem, kind):
    z = np.load(ROOT/"out"/f"{stem}_{kind}_preds.npz", allow_pickle=True)
    return z["y"].astype(float), z["pred"].astype(float), \
        (z["row_id"] if "row_id" in z.files else None)


def surface(data, source, target, base_stem, cell_stem):
    ys, bv, _ = load(base_stem, "val")
    ysc, cv, _ = load(cell_stem, "val")
    yt, bt, rid = load(base_stem, "test")
    ytc, ct, ridc = load(cell_stem, "test")
    if not (np.array_equal(ys, ysc) and np.array_equal(yt, ytc)
            and np.array_equal(rid, ridc)):
        raise ValueError(f"surface mismatch {source}->{target}")
    src = data[data.season.eq(source)].reset_index(drop=True)
    tgt = data[data.season.eq(target)].reset_index(drop=True)
    tgt = tgt.set_index("row_id").loc[rid].reset_index()
    ps, pt = post(.45*bv+.55*cv), post(.45*bt+.55*ct)
    ms, mt = middle_keys(
        src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
        tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.)
    return {"frame": tgt, "y": yt, "k0": np.clip(pt+mb+eb, 0, 1),
            "delta": post(ct)-post(bt)}


def strong_2024(data, source_surface):
    def avg(stem, seeds):
        zs = [np.load(ROOT/"out"/f"cat_{stem}_s{s}_val_preds.npz") for s in seeds]
        y = zs[0]["y"].astype(float)
        return y, np.mean([z["pred"].astype(float) for z in zs], axis=0)
    y, b = avg("VB2_base", (42, 7, 13, 3, 4, 5, 6, 8))
    yc, c = avg("ZD5", (42, 7, 13, 3, 4, 5))
    if not np.array_equal(y, yc):
        raise ValueError("strong target mismatch")
    # source_surface was honestly fitted on 2022 and predicts 2023.  Refit the
    # same K0 maps on the 2023 local OOF residual before applying to strong 2024.
    src = data[data.season.eq(2023)].reset_index(drop=True)
    tgt = data[data.season.eq(2024)].reset_index(drop=True)
    ys, bv, _ = load("cat_MVA_native", "val")
    ysc, cv, _ = load("cat_MVCELL_s42", "val")
    ps = post(.45*bv+.55*cv)
    ms, mt = middle_keys(
        src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
        tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-ps
    r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(ps+ma)
    r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, 500.)
    return {"frame": tgt, "y": y,
            "k0": np.clip(post(.45*b+.55*c)+mb+eb, 0, 1),
            "delta": post(c)-post(b)}


def qcodes(xs, xt, q):
    edges = np.unique(np.quantile(xs[np.isfinite(xs)], np.linspace(0, 1, q+1)))
    if len(edges) < 3:
        return np.zeros(len(xs), int), np.zeros(len(xt), int)
    edges[0], edges[-1] = -np.inf, np.inf
    return (np.searchsorted(edges[1:-1], xs, side="right"),
            np.searchsorted(edges[1:-1], xt, side="right"))


def keys(arm, src, tgt):
    ds, dt = src["delta"], tgt["delta"]
    if arm == "signed_q8":
        return qcodes(ds, dt, 8)
    if arm == "abs_q8":
        return qcodes(np.abs(ds), np.abs(dt), 8)
    if arm == "sign_abs_q4":
        a, b = qcodes(np.abs(ds), np.abs(dt), 4)
        return (a+4*(ds > 0), b+4*(dt > 0))
    if arm == "p_q4_x_signed_q4":
        a, b = qcodes(ds, dt, 4)
        c, d = qcodes(src["k0"], tgt["k0"], 4)
        return a+4*c, b+4*d
    raise ValueError(arm)


def report(label, target, before, after):
    f, y = target["frame"], target["y"]
    masks = {"all": np.ones(len(f), bool),
             "R": f.game_type.eq("R").to_numpy(),
             "F": f.game_type.eq("F").to_numpy(),
             "early": f.game_month.le(6).to_numpy(),
             "late": f.game_month.gt(6).to_numpy()}
    vals = {n: bss(y[m], after[m])-bss(y[m], before[m]) for n,m in masks.items()}
    print(f"{label:<28} " + " ".join(f"{n}={v:+.3f}" for n,v in vals.items()))
    return vals


def apply_arm(source, target, arm, scale):
    ks, kt = keys(arm, source, target)
    r = source["y"]-source["k0"]
    r -= r.mean()
    _, adj = fit_map(ks, kt, r, 5000.)
    return np.clip(target["k0"]+scale*adj, 0, 1)


def main():
    # These runs predate 5b61fbd (2026-08-13 03:23:39): their fit
    # partitions held later seasons, so the "next season" transitions
    # below were never next-season transitions. Refuse rather than
    # reproduce the numbers. See docs/INVALIDATED.tsv.
    _guard_invalidated(['MV21_base', 'MV21_cell', 'MVB22_native', 'MVCELL22_s42'])
    cols = ["season", "row_id", "game_month", "game_type", "pitcher_id",
            "batter_id", "asof_pitcher_prev5_game_middle_rate"]
    data = pd.read_csv(ROOT/"data"/"train.csv", usecols=cols)
    s22 = surface(data, 2021, 2022, "cat_MV21_base", "cat_MV21_cell")
    s23 = surface(data, 2022, 2023, "cat_MVB22_native", "cat_MVCELL22_s42")
    s24 = surface(data, 2023, 2024, "cat_MVA_native", "cat_MVCELL_s42")
    strong = strong_2024(data, s23)
    for arm in ARMS:
        for scale in (.25, .50):
            p23 = apply_arm(s22, s23, arm, scale)
            p24 = apply_arm(s23, s24, arm, scale)
            pst = apply_arm(s23, strong, arm, scale)
            print(f"\narm={arm} scale={scale:g}")
            report("2022fit->2023", s23, s23["k0"], p23)
            report("2023fit->2024 local", s24, s24["k0"], p24)
            report("2023fit->2024 strong", strong, strong["k0"], pst)


if __name__ == "__main__":
    main()

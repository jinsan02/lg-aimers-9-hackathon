"""Deep robustness audit for frozen recent-middle residual correction."""

from glob import glob
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = "control_success"
W = .55
SEEDS = ["3", "4", "5", "6", "8", "13"]
SLOPE = 1.0416
SHIFT = .0052


def load(tag, seed, kind):
    return np.load(os.path.join(ROOT, "out", f"cat_{tag}_s{seed}_{kind}_preds.npz"))


def bss(y, p, center=True):
    y, p = np.asarray(y, float), np.asarray(p, float).copy()
    if center:
        p += y.mean() - p.mean()
    r = y.mean()
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1-r)))


def post(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1-1e-6)
    z = np.log(p / (1-p))
    return np.clip(1 / (1 + np.exp(-SLOPE*z)) - SHIFT, 0, 1)


def bins(x, z, q):
    good = x[np.isfinite(x)]
    e = np.unique(np.quantile(good, np.linspace(0, 1, q + 1)))
    e[0], e[-1] = -np.inf, np.inf
    a = np.searchsorted(e[1:-1], x, side="right")
    b = np.searchsorted(e[1:-1], z, side="right")
    a[~np.isfinite(x)], b[~np.isfinite(z)] = -1, -1
    return a, b, e


def correction(bs, bt, residual, k):
    g = pd.DataFrame({"b": bs, "r": residual}).groupby("b").r.agg(["sum", "size"])
    m = (g["sum"] / (g["size"] + k)).to_dict()
    return pd.Series(bt).map(m).fillna(0).to_numpy(float), g, m


def score_segments(d, y, p, a):
    masks = {
        "all": np.ones(len(d), bool), "early": d.game_month.to_numpy() <= 6,
        "late": d.game_month.to_numpy() >= 7,
        "R": d.game_type.to_numpy() == "R", "F": d.game_type.to_numpy() == "F",
    }
    return {n: bss(y[m], p[m]+a[m]) - bss(y[m], p[m]) for n, m in masks.items()}


def main():
    use = ["season", "game_month", "game_type", T,
           "asof_pitcher_middle_rate", "asof_pitcher_prev1_game_middle_rate",
           "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]
    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=use)
    s, t = d[d.season == 2023].reset_index(drop=True), d[d.season == 2024].reset_index(drop=True)
    P23, P24, y23, y24 = {}, {}, None, None
    for seed in SEEDS:
        b3, c3 = load("AB_base", seed, "val"), load("DW_cell", seed, "val")
        b4, c4 = load("AB_base", seed, "test"), load("DW_cell", seed, "test")
        P23[seed] = (1-W)*b3["pred"].astype(float) + W*c3["pred"].astype(float)
        P24[seed] = (1-W)*b4["pred"].astype(float) + W*c4["pred"].astype(float)
        y23, y24 = b3["y"].astype(float), b4["y"].astype(float)
    p23 = np.mean(list(P23.values()), 0)
    p24 = np.mean(list(P24.values()), 0)

    cols = ["asof_pitcher_middle_rate", "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]
    rows = []
    for col in cols:
        x, z = s[col].to_numpy(float), t[col].to_numpy(float)
        for q in (5, 8, 12):
            bs, bt, _ = bins(x, z, q)
            for k in (200., 500., 1000., 2000.):
                r = y23 - p23
                r -= r.mean()
                a, _, _ = correction(bs, bt, r, k)
                sc = score_segments(t, y24, p24, a)
                rows.append(dict(feature=col.replace("asof_pitcher_", ""), q=q, k=k,
                                 **sc, adj_sd=a.std()))
    R = pd.DataFrame(rows).sort_values("all", ascending=False)
    print("=== sensitivity (top 30 by target centered gain) ===")
    print(R.head(30).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # Primary predeclared arm from the sweep: prev5, q=8, k=500.
    col, q, k = "asof_pitcher_prev5_game_middle_rate", 8, 500.
    bs, bt, edges = bins(s[col].to_numpy(float), t[col].to_numpy(float), q)
    gains = []
    corrected = []
    for seed in SEEDS:
        r = y23 - P23[seed]
        r -= r.mean()
        a, _, _ = correction(bs, bt, r, k)
        gains.append(bss(y24, P24[seed]+a)-bss(y24, P24[seed]))
        corrected.append(P24[seed] + a)
    gains = np.asarray(gains)
    se = gains.std(ddof=1) / np.sqrt(len(gains))
    ens_gain = bss(y24, np.mean(corrected, 0)) - bss(y24, p24)
    print("\n=== primary prev5 q8 k500 seed robustness ===")
    print(dict(zip(SEEDS, np.round(gains, 3))))
    print(f"mean={gains.mean():+.3f}, SE={se:.3f}, t={gains.mean()/se:+.3f}, "
          f"ensemble={ens_gain:+.3f}")

    # Exact current submission order: probability blend -> logit slope -> shift.
    ps, pt = post(p23), post(p24)
    raw_rows = []
    for ccol in cols:
        xx, zz = s[ccol].to_numpy(float), t[ccol].to_numpy(float)
        for qq in (5, 8, 12, 16):
            bsrc, btgt, _ = bins(xx, zz, qq)
            for kk in (200., 500., 1000., 2000., 4000.):
                r0 = y23-ps; r0 -= r0.mean()
                src_adj, _, _ = correction(bsrc, bsrc, r0, kk)
                tgt_adj, _, _ = correction(bsrc, btgt, r0, kk)
                mm = float(src_adj.mean())
                tgt_adj -= mm
                early = t.game_month.to_numpy() <= 6
                raw_rows.append(dict(feature=ccol.replace("asof_pitcher_", ""),
                    q=qq, k=kk,
                    raw=bss(y24,pt+tgt_adj,False)-bss(y24,pt,False),
                    centered=bss(y24,pt+tgt_adj)-bss(y24,pt),
                    early=bss(y24[early],pt[early]+tgt_adj[early],False)-bss(y24[early],pt[early],False),
                    late=bss(y24[~early],pt[~early]+tgt_adj[~early],False)-bss(y24[~early],pt[~early],False),
                    adj_sd=tgt_adj.std()))
    raw_tab = pd.DataFrame(raw_rows).sort_values("raw", ascending=False)
    print("\n=== exact fixed-post RAW sensitivity ===")
    print(raw_tab.head(40).to_string(index=False,float_format=lambda x:f"{x:+.3f}"))
    raw_tab.to_csv(os.path.join(ROOT,"out","middle_raw_sweep.csv"),index=False)

    rr = y23 - ps
    rr -= rr.mean()
    aa, _, _ = correction(bs, bt, rr, k)
    seg = score_segments(t, y24, pt, aa)
    print("submission-post transfer:", {n: round(v, 3) for n, v in seg.items()},
          f"raw={bss(y24, pt+aa, False)-bss(y24, pt, False):+.3f}")

    r = y23 - p23
    r -= r.mean()
    a, g, m = correction(bs, bt, r, k)
    tab_s = pd.DataFrame({"bin": bs, "x": s[col], "resid": r}).groupby("bin").agg(
        source_n=("x", "size"), source_x=("x", "mean"), source_resid=("resid", "mean"))
    tab_t = pd.DataFrame({"bin": bt, "x": t[col], "resid": y24-p24,
                          "adj": a}).groupby("bin").agg(
        target_n=("x", "size"), target_x=("x", "mean"),
        target_resid=("resid", "mean"), frozen_adj=("adj", "mean"))
    tab = tab_s.join(tab_t, how="outer")
    print("\n=== bin mechanism table ===")
    print(tab.to_string(float_format=lambda x: f"{x:+.6f}"))
    common = tab.dropna()
    print(f"bin residual correlation source->target: "
          f"{common.source_resid.corr(common.target_resid):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

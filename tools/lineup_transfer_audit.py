"""Time-honest transfer audit for historical lineup roles."""

import argparse
from glob import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_CELL = 0.55


def ensemble(tag, kind):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")))
    if not fs:
        raise FileNotFoundError(f"{tag} {kind}")
    z = [np.load(f) for f in fs]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError("target mismatch")
    return np.mean([q["pred"].astype(np.float64) for q in z], axis=0), y


def bss(y, p, center=True):
    p = np.asarray(p, np.float64).copy()
    if center:
        p += y.mean() - p.mean()
    r = float(y.mean())
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def quantile_pair(s, t, col, q=6):
    x = s[col].to_numpy(np.float64)
    z = t[col].to_numpy(np.float64)
    good = x[np.isfinite(x)]
    edges = np.unique(np.quantile(good, np.linspace(0, 1, q + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.searchsorted(edges[1:-1], x, side="right").astype(str)
    b = np.searchsorted(edges[1:-1], z, side="right").astype(str)
    a[~np.isfinite(x)], b[~np.isfinite(z)] = "NA", "NA"
    return a, b


def combine(a, b):
    return np.char.add(np.char.add(np.asarray(a).astype(str), "|"),
                       np.asarray(b).astype(str))


def fit_apply(ks, kt, residual, k):
    tab = pd.DataFrame({"key": ks, "r": residual}).groupby("key").r.agg(["sum", "size"])
    mp = (tab["sum"] / (tab["size"] + k)).to_dict()
    return pd.Series(kt).map(mp).fillna(0.0).to_numpy(), len(tab)


def audit(name, ks, kt, s, t):
    rs = s.y.to_numpy() - s.pred.to_numpy()
    rs -= rs.mean()
    yt, pt = t.y.to_numpy(), t.pred.to_numpy()
    early = t.game_month.to_numpy() <= 6
    first = np.arange(len(s)) < len(s) // 2
    rows = []
    for k in (100., 300., 500., 1000.):
        adj, groups = fit_apply(ks, kt, rs, k)
        cv = []
        for fit, val in ((first, ~first), (~first, first)):
            a, _ = fit_apply(np.asarray(ks)[fit], np.asarray(ks)[val], rs[fit], k)
            cv.append(bss(s.y.to_numpy()[val], s.pred.to_numpy()[val] + a) -
                      bss(s.y.to_numpy()[val], s.pred.to_numpy()[val]))
        rows.append({
            "name": name, "k": k, "groups": groups,
            "source_cv": np.mean(cv),
            "target": bss(yt, pt + adj) - bss(yt, pt),
            "early": bss(yt[early], pt[early] + adj[early]) - bss(yt[early], pt[early]),
            "late": bss(yt[~early], pt[~early] + adj[~early]) - bss(yt[~early], pt[~early]),
            "adj_sd": adj.std(),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="AB_base")
    ap.add_argument("--cell", default="DW_cell")
    ap.add_argument("--source", type=int, default=2023)
    ap.add_argument("--target", type=int, default=2024)
    ap.add_argument("--output", default="")
    args = ap.parse_args(argv)
    b23, y23 = ensemble(args.base, "val")
    c23, yc23 = ensemble(args.cell, "val")
    b24, y24 = ensemble(args.base, "test")
    c24, yc24 = ensemble(args.cell, "test")
    if not (np.array_equal(y23, yc23) and np.array_equal(y24, yc24)):
        raise ValueError("base/cell mismatch")
    p23 = (1 - W_CELL) * b23 + W_CELL * c23
    p24 = (1 - W_CELL) * b24 + W_CELL * c24

    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    lu = pd.read_csv(os.path.join(ROOT, "data", "processed", "lineup_history.csv"))
    d = d.merge(lu, on=["batter_id", "season"], how="left")
    s = d[d.season == args.source].reset_index(drop=True)
    t = d[d.season == args.target].reset_index(drop=True)
    if len(s) != len(y23) or len(t) != len(y24):
        raise ValueError("row mismatch")
    s["y"], s["pred"] = y23, p23
    t["y"], t["pred"] = y24, p24
    print(f"surface {args.source}->{args.target} base={args.base} cell={args.cell}")
    print(f"coverage {args.source}={s.lineup_slot.notna().mean():.1%} "
          f"{args.target}={t.lineup_slot.notna().mean():.1%}")

    def slot(d0):
        return d0.lineup_slot.fillna(-1).astype(int).astype(str).to_numpy()

    def role(d0):
        x = d0.lineup_slot
        return np.select([x <= 2, x <= 5, x <= 9],
                         ["top", "core", "lower"], default="NA")

    count_s = (s.balls_before.astype(str) + "-" + s.strikes_before.astype(str)).to_numpy()
    count_t = (t.balls_before.astype(str) + "-" + t.strikes_before.astype(str)).to_numpy()
    same_s = (s.pitcher_hand == s.batter_hand).astype(str).to_numpy()
    same_t = (t.pitcher_hand == t.batter_hand).astype(str).to_numpy()
    candidates = {
        "slot": (slot(s), slot(t)),
        "role": (role(s), role(t)),
        "slot*count": (combine(slot(s), count_s), combine(slot(t), count_t)),
        "role*count": (combine(role(s), count_s), combine(role(t), count_t)),
        "slot*same_hand": (combine(slot(s), same_s), combine(slot(t), same_t)),
        "role*same_hand": (combine(role(s), same_s), combine(role(t), same_t)),
        "role*game_type": (combine(role(s), s.game_type), combine(role(t), t.game_type)),
    }
    for col in ("lineup_n", "lineup_stability", "lineup_entropy",
                "lineup_top_share", "lineup_core_share", "lineup_lower_share"):
        candidates[col] = quantile_pair(s, t, col)

    rows = []
    for name, (ks, kt) in candidates.items():
        rows.extend(audit(name, ks, kt, s, t))
    out = pd.DataFrame(rows).sort_values("target", ascending=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    stable = out[(out.source_cv > 0) & (out.early > 0) &
                 (out.late > 0) & (out.target >= 2)]
    print("\nPROMOTE")
    print(stable.to_string(index=False, float_format=lambda x: f"{x:+.3f}")
          if len(stable) else "none")
    output = args.output or os.path.join(
        ROOT, "out", f"lineup_transfer_{args.source}_{args.target}.csv")
    out.to_csv(output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

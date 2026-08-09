"""Honest 2023 -> unseen 2024 transfer audit for mechanics-history features."""

from glob import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_CELL = 0.55


def ensemble(tag, kind):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")))
    z = [np.load(f) for f in fs]
    if not z:
        raise FileNotFoundError(f"{tag} {kind}")
    y = z[0]["y"].astype(np.float64)
    return np.mean([q["pred"].astype(np.float64) for q in z], axis=0), y


def bss(y, p):
    p = np.asarray(p, np.float64).copy()
    p += y.mean() - p.mean()
    r = float(y.mean())
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def qpair(s, t, col, q=8):
    x, z = s[col].to_numpy(float), t[col].to_numpy(float)
    good = x[np.isfinite(x)]
    if len(good) < 100:
        return None
    edges = np.unique(np.quantile(good, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.searchsorted(edges[1:-1], x, side="right").astype(str)
    b = np.searchsorted(edges[1:-1], z, side="right").astype(str)
    a[~np.isfinite(x)], b[~np.isfinite(z)] = "NA", "NA"
    return a, b


def combine(a, b):
    return np.char.add(np.char.add(np.asarray(a).astype(str), "|"),
                       np.asarray(b).astype(str))


def fit_apply(ks, kt, r, k):
    tab = pd.DataFrame({"key": ks, "r": r}).groupby("key").r.agg(["sum", "size"])
    mp = (tab["sum"] / (tab["size"] + k)).to_dict()
    return pd.Series(kt).map(mp).fillna(0.0).to_numpy(), len(tab)


def audit(name, ks, kt, s, t):
    rs = s.y.to_numpy() - s.pred.to_numpy()
    rs -= rs.mean()
    yt, pt = t.y.to_numpy(), t.pred.to_numpy()
    early = t.game_month.to_numpy() <= 6
    first = np.arange(len(s)) < len(s) // 2
    out = []
    for k in (200., 500., 1000., 2000.):
        adj, ng = fit_apply(ks, kt, rs, k)
        cv = []
        for fit, val in ((first, ~first), (~first, first)):
            av, _ = fit_apply(np.asarray(ks)[fit], np.asarray(ks)[val], rs[fit], k)
            cv.append(bss(s.y.to_numpy()[val], s.pred.to_numpy()[val] + av) -
                      bss(s.y.to_numpy()[val], s.pred.to_numpy()[val]))
        out.append({"name": name, "k": k, "groups": ng,
                    "source_cv": np.mean(cv),
                    "target": bss(yt, pt + adj) - bss(yt, pt),
                    "early": bss(yt[early], pt[early] + adj[early]) - bss(yt[early], pt[early]),
                    "late": bss(yt[~early], pt[~early] + adj[~early]) - bss(yt[~early], pt[~early]),
                    "adj_sd": adj.std()})
    return out


def main():
    b23, y23 = ensemble("AB_base", "val")
    c23, yc23 = ensemble("DW_cell", "val")
    b24, y24 = ensemble("AB_base", "test")
    c24, yc24 = ensemble("DW_cell", "test")
    if not (np.array_equal(y23, yc23) and np.array_equal(y24, yc24)):
        raise ValueError("target mismatch")
    p23, p24 = (1-W_CELL)*b23 + W_CELL*c23, (1-W_CELL)*b24 + W_CELL*c24

    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    mh = pd.read_csv(os.path.join(ROOT, "data", "processed", "mechanics_history.csv"))
    d = d.merge(mh, on=["pitcher_id", "season"], how="left")
    s, t = d[d.season == 2023].reset_index(drop=True), d[d.season == 2024].reset_index(drop=True)
    if len(s) != len(y23) or len(t) != len(y24):
        raise ValueError("row mismatch")
    s["y"], s["pred"] = y23, p23
    t["y"], t["pred"] = y24, p24
    print(f"coverage 2023={s.mech_last_arm_angle.notna().mean():.1%} "
          f"2024={t.mech_last_arm_angle.notna().mean():.1%}")

    pairs = {}
    mech_cols = [c for c in mh.columns if c.startswith("mech_") and
                 c not in ("mech_last_season",)]
    for c in mech_cols:
        q = qpair(s, t, c)
        if q is not None:
            pairs[c] = q
    recent = qpair(s, t, "asof_pitcher_prev5_game_middle_rate")
    if recent:
        for c in ("mech_last_arm_angle", "mech_delta_arm_angle",
                  "mech_last_rel_scatter", "mech_delta_rel_scatter",
                  "mech_last_extension", "mech_delta_extension",
                  "mech_last_fb_speed", "mech_delta_fb_speed",
                  "mech_last_fb_ivb", "mech_delta_fb_ivb"):
            if c in pairs:
                pairs[f"{c}*recent_middle"] = (combine(pairs[c][0], recent[0]),
                                                 combine(pairs[c][1], recent[1]))

    rows = []
    for name, (ks, kt) in pairs.items():
        rows.extend(audit(name, ks, kt, s, t))
    out = pd.DataFrame(rows).sort_values("target", ascending=False)
    print(out.head(60).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    stable = out[(out.source_cv > 0) & (out.early > 0) &
                 (out.late > 0) & (out.target >= 2)]
    print("\nPROMOTE")
    print(stable.to_string(index=False, float_format=lambda x: f"{x:+.3f}")
          if len(stable) else "none")
    out.to_csv(os.path.join(ROOT, "out", "mechanics_transfer_audit.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

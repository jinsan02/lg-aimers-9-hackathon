"""Exhaustive source-season residual transfer audit for feature discovery.

Build every correction from 2023 AB_base+DW_cell predictions, freeze bin edges
and shrinkage, and apply it once to unseen 2024.  Target centering is printed
only to isolate resolution; the actual correction never inspects target rows.
"""

from glob import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = "control_success"
W_CELL = 0.55
SHRINK = 500.0
Q = 8


def ensemble(tag, kind):
    fs = sorted(glob(os.path.join(ROOT, "out", f"cat_{tag}_s*_{kind}_preds.npz")))
    if not fs:
        raise FileNotFoundError(f"{tag} {kind}")
    z = [np.load(f) for f in fs]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError(f"target mismatch: {tag} {kind}")
    return np.mean([q["pred"].astype(np.float64) for q in z], axis=0), y, len(fs)


def bss(y, p, center=False):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64).copy()
    if center:
        p += y.mean() - p.mean()
    p = np.clip(p, 0, 1)
    r = float(y.mean())
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def num_bin(source, target, col):
    x = pd.to_numeric(source[col], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(target[col], errors="coerce").to_numpy(np.float64)
    good = x[np.isfinite(x)]
    if len(good) == 0:
        return None
    edges = np.unique(np.nanquantile(good, np.linspace(0, 1, Q + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    bx = np.searchsorted(edges[1:-1], x, side="right")
    bz = np.searchsorted(edges[1:-1], z, side="right")
    bx[~np.isfinite(x)], bz[~np.isfinite(z)] = -1, -1
    return bx, bz


def cat_pair(source, target, cols):
    def make(d):
        if len(cols) == 1:
            return d[cols[0]].fillna("NA").astype(str).to_numpy()
        return d[list(cols)].fillna("NA").astype(str).agg("|".join, axis=1).to_numpy()
    return make(source), make(target)


def derived(source, target):
    out = {}

    def add(name, fn):
        out[name] = (fn(source), fn(target))

    add("count", lambda d: d.balls_before.astype(str) + "-" + d.strikes_before.astype(str))
    add("phase", lambda d: np.where(d.game_month <= 6, "early", "late"))
    add("inning_role", lambda d: pd.cut(d.inning, [0, 3, 5, 7, 99], labels=False).astype(str))
    add("score_state", lambda d: pd.cut(d.score_diff_pitcher_team,
                                         [-np.inf, -4, -1, 0, 1, 4, np.inf],
                                         labels=False).astype(str))
    add("leverage", lambda d: pd.cut(d.li, [-np.inf, .5, 1, 2, np.inf],
                                      labels=False).astype(str))
    add("p_exp", lambda d: pd.cut(d.asof_pitcher_n,
                                   [-np.inf, 50, 300, 1000, 3000, 7000, np.inf],
                                   labels=False).astype(str))
    add("b_exp", lambda d: pd.cut(d.asof_batter_n,
                                   [-np.inf, 50, 200, 1000, 3000, 10000, np.inf],
                                   labels=False).astype(str))
    add("same_hand", lambda d: (d.pitcher_hand.astype(str) ==
                                 d.batter_hand.astype(str)).astype(str))
    add("form_delta1", lambda d: d.asof_pitcher_prev1_game_success_rate -
                                  d.asof_pitcher_success_rate)
    add("form_delta3", lambda d: d.asof_pitcher_prev3_game_success_rate -
                                  d.asof_pitcher_success_rate)
    add("form_delta5", lambda d: d.asof_pitcher_prev5_game_success_rate -
                                  d.asof_pitcher_success_rate)
    add("recent_spread", lambda d: d[["asof_pitcher_prev1_game_success_rate",
                                      "asof_pitcher_prev3_game_success_rate",
                                      "asof_pitcher_prev5_game_success_rate"]].max(axis=1) -
                                   d[["asof_pitcher_prev1_game_success_rate",
                                      "asof_pitcher_prev3_game_success_rate",
                                      "asof_pitcher_prev5_game_success_rate"]].min(axis=1))
    add("style_aggr", lambda d: d.asof_pitcher_middle_rate - d.asof_pitcher_ball_rate)
    add("zone_minus_cmd", lambda d: d.asof_pitcher_strike_rate -
                                     d.asof_pitcher_success_rate)
    add("fail_rev_share", lambda d: d.asof_pitcher_reverse_rate /
        (d.asof_pitcher_middle_rate + d.asof_pitcher_ball_rate +
         d.asof_pitcher_reverse_rate).replace(0, np.nan))

    # Numeric derived values get source-frozen quantile bins.
    for name in ["form_delta1", "form_delta3", "form_delta5", "recent_spread",
                 "style_aggr", "zone_minus_cmd", "fail_rev_share"]:
        a, b = out[name]
        sa, sb = source.copy(), target.copy()
        sa["_v"], sb["_v"] = a, b
        out[name] = num_bin(sa, sb, "_v")

    # Interactions that encode baseball mechanisms rather than arbitrary pairs.
    base = dict(out)
    pairs = [
        ("count", "inning_role"), ("count", "game_type"),
        ("count", "same_hand"), ("count", "leverage"),
        ("count", "b_exp"), ("phase", "game_type"),
        ("phase", "inning_role"), ("phase", "p_exp"),
        ("inning_role", "p_exp"), ("inning_role", "score_state"),
        ("game_type", "p_exp"), ("game_type", "b_exp"),
        ("p_exp", "recent_spread"), ("p_exp", "style_aggr"),
        ("style_aggr", "count"), ("fail_rev_share", "count"),
    ]
    raw = {
        "game_type": (source.game_type.astype(str).to_numpy(),
                      target.game_type.astype(str).to_numpy()),
    }
    base.update(raw)
    for x, y in pairs:
        ax, bx = base[x]
        ay, by = base[y]
        ax, bx = np.asarray(ax).astype(str), np.asarray(bx).astype(str)
        ay, by = np.asarray(ay).astype(str), np.asarray(by).astype(str)
        out[f"{x}*{y}"] = (np.char.add(np.char.add(ax, "|"), ay),
                             np.char.add(np.char.add(bx, "|"), by))
    return out


def fit_map(keys, residual):
    tab = pd.DataFrame({"k": keys, "r": residual}).groupby("k", sort=False).r.agg(["sum", "size"])
    return (tab["sum"] / (tab["size"] + SHRINK)).to_dict()


def apply_map(keys, table):
    return pd.Series(keys).map(table).fillna(0.0).to_numpy(np.float64)


def audit_one(name, ks, kt, source, target):
    rs = source.y.to_numpy() - source.pred.to_numpy()
    yt, pt = target.y.to_numpy(), target.pred.to_numpy()
    # Source-global bias belongs to SHIFT; feature maps only conditional residual.
    rs = rs - rs.mean()
    table = fit_map(ks, rs)
    adj = apply_map(kt, table)

    raw_gain = bss(yt, pt + adj) - bss(yt, pt)
    cen_gain = bss(yt, pt + adj, True) - bss(yt, pt, True)
    early = target.game_month.to_numpy() <= 6
    ge = bss(yt[early], pt[early] + adj[early], True) - bss(yt[early], pt[early], True)
    gl = bss(yt[~early], pt[~early] + adj[~early], True) - bss(yt[~early], pt[~early], True)

    # Honest within-source temporal halves: first -> second and second -> first.
    first = np.arange(len(source)) < len(source) // 2
    cv = []
    for fit, val in ((first, ~first), (~first, first)):
        m = fit_map(np.asarray(ks)[fit], rs[fit])
        a = apply_map(np.asarray(ks)[val], m)
        y, p = source.y.to_numpy()[val], source.pred.to_numpy()[val]
        cv.append(bss(y, p + a, True) - bss(y, p, True))
    return dict(name=name, groups=len(table), source_cv=np.mean(cv),
                raw=raw_gain, centered=cen_gain, early=ge, late=gl,
                adj_sd=float(adj.std()), coverage=float(np.mean(pd.Series(kt).isin(table))))


def main():
    b23, y23, nb = ensemble("AB_base", "val")
    c23, yc23, nc = ensemble("DW_cell", "val")
    b24, y24, _ = ensemble("AB_base", "test")
    c24, yc24, _ = ensemble("DW_cell", "test")
    if not (np.array_equal(y23, yc23) and np.array_equal(y24, yc24)):
        raise ValueError("base/cell target mismatch")
    pred23 = (1 - W_CELL) * b23 + W_CELL * c23
    pred24 = (1 - W_CELL) * b24 + W_CELL * c24

    d = pd.read_csv(os.path.join(ROOT, "data", "train.csv"))
    s = d[d.season == 2023].reset_index(drop=True)
    t = d[d.season == 2024].reset_index(drop=True)
    if len(s) != len(y23) or len(t) != len(y24):
        raise ValueError(f"row mismatch: {len(s)}/{len(y23)} {len(t)}/{len(y24)}")
    s["y"], s["pred"] = y23, pred23
    t["y"], t["pred"] = y24, pred24
    print(f"AB seeds={nb}, DW seeds={nc} | source={len(s):,}, target={len(t):,}")
    print(f"target base raw={bss(y24,pred24):.3f}, centered={bss(y24,pred24,True):.3f}")

    candidates = {}
    cats = ["game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
            "balls_before", "strikes_before", "outs_before", "num_runners_on",
            "base_state", "pitcher_hand", "batter_hand", "pitcher_team_id",
            "batter_team_id"]
    for col in cats:
        candidates[col] = cat_pair(s, t, [col])
    nums = ["run_total_before", "score_diff_home", "score_diff_pitcher_team",
            "home_win_expectancy", "li", "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
            "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
            "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
            "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
            "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
            "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    for col in nums:
        q = num_bin(s, t, col)
        if q is not None:
            candidates[f"bin:{col}"] = q
    candidates.update(derived(s, t))

    rows = [audit_one(n, a, b, s, t) for n, (a, b) in candidates.items()]
    out = pd.DataFrame(rows).sort_values("centered", ascending=False)
    print("\n=== source-frozen 2023 -> unseen 2024 feature transfer (k=500) ===")
    print(out.head(35).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    stable = out[(out.source_cv > 0) & (out.early > 0) & (out.late > 0) & (out.centered >= 2)]
    print("\n=== PROMOTE: source-CV+, early+, late+, target centered >= +2 ===")
    print(stable.to_string(index=False, float_format=lambda x: f"{x:+.3f}") if len(stable)
          else "none")
    out.to_csv(os.path.join(ROOT, "out", "feature_transfer_sweep.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

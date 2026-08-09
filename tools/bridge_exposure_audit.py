"""F→R 리그 브리지 효과가 현재 R 경험량에 따라 감쇠하는지 감사한다.

2023 R 예측 잔차에서 감쇠계수를 고정하고 2024 R에 적용한다. 시즌 S의
현재 경험량은 공식 asof_pitcher_n에서 시즌 시작 앵커를 뺀 값이라 행 독립이다.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import roster_transition as rt
import season_std as ss

T = "control_success"
BINS = [-1, 0, 25, 75, 150, 300, 600, np.inf]
K_GRID = (25, 50, 100, 200, 400, 800, 1600)


def ensemble(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        raise FileNotFoundError(pattern)
    vals, y = [], None
    for f in fs:
        z = np.load(f)
        vals.append(z["pred"].astype(np.float64))
        if y is None:
            y = z["y"].astype(np.float64)
        elif not np.array_equal(y, z["y"]):
            raise ValueError(f"target mismatch: {f}")
    return np.mean(vals, axis=0), y, len(fs)


def bss(y, p, center=True):
    y, p = np.asarray(y), np.asarray(p).copy()
    if center:
        p += y.mean() - p.mean()
    r = float(y.mean())
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


def prepare():
    cols = ["row_id", "season", "game_month", "game_type", "pitcher_id",
            "asof_pitcher_n", T]
    d = pd.read_csv("./data/train.csv", usecols=cols)
    anchors = ss.build_anchors(d)["pitcher"][["pitcher_id", "season", "n0"]]
    d = d.merge(anchors, on=["pitcher_id", "season"], how="left",
                validate="many_to_one", sort=False)
    d["cur_n"] = np.maximum(d.asof_pitcher_n - d.n0.fillna(0), 0)
    tab = rt.build_table(d)
    d, _ = rt.add_features(d, tab)
    status = np.full(len(d), "other", dtype=object)
    status[d.rt_same_cont.to_numpy() == 1] = "same_cont"
    status[d.rt_same_return.to_numpy() == 1] = "same_return"
    status[d.rt_f_to_r.to_numpy() == 1] = "F_to_R"
    status[d.rt_r_to_f.to_numpy() == 1] = "R_to_F"
    status[d.rt_seen_before.to_numpy() == 0] = "unseen"
    d["status"] = status
    d["exp_bin"] = pd.cut(d.cur_n, BINS, include_lowest=True)
    return d


def attach_predictions(d):
    p23, y23, n23 = ensemble("./out/cat_H3_base_s*_val_preds.npz")
    p24, y24, n24 = ensemble("./out/cat_VB2_base_s*_val_preds.npz")
    s = d[(d.season == 2023) & (d.game_type == "R")].copy()
    t = d[d.season == 2024].copy()
    if len(s) != len(p23) or not np.array_equal(s[T].to_numpy(), y23):
        raise ValueError(f"2023 R 정렬 불일치: {len(s)} vs {len(p23)}")
    if len(t) != len(p24) or not np.array_equal(t[T].to_numpy(), y24):
        raise ValueError(f"2024 정렬 불일치: {len(t)} vs {len(p24)}")
    s["pred"], t["pred"] = p23, p24
    s["resid"], t["resid"] = s[T] - p23, t[T] - p24
    print(f"pred ensembles: H3={n23}, VB2={n24}")
    return s, t[t.game_type == "R"].copy()


def table(d, title):
    q = d[d.status.isin(["F_to_R", "same_cont"])].copy()
    z = (q.groupby(["exp_bin", "status"], observed=True)
           .agg(n=(T, "size"), pitchers=("pitcher_id", "nunique"),
                actual=(T, "mean"), pred=("pred", "mean"),
                resid=("resid", "mean")).reset_index())
    print(f"\n=== {title}: current-season R exposure ===")
    print(z.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    p = z.pivot(index="exp_bin", columns="status", values="resid")
    n = z.pivot(index="exp_bin", columns="status", values="n")
    if {"F_to_R", "same_cont"} <= set(p.columns):
        g = pd.DataFrame({"gap": p.F_to_R - p.same_cont,
                          "F_to_R_n": n.F_to_R,
                          "same_cont_n": n.same_cont})
        print("\nconditional residual gap (F_to_R - same_cont):")
        print(g.to_string(float_format=lambda x: f"{x:.6f}"))
    return z


def fit_decay(source, target):
    # 일반적인 시즌 진행 편향은 같은 exposure bin의 same_cont 잔차로 제거한다.
    base = (source[source.status == "same_cont"]
            .groupby("exp_bin", observed=True).resid.mean())
    f = source[source.status == "F_to_R"].copy()
    f["rel_resid"] = f.resid - f.exp_bin.map(base).astype(float)
    f = f[np.isfinite(f.rel_resid)]

    fits = []
    n = f.cur_n.to_numpy(np.float64)
    y = f.rel_resid.to_numpy(np.float64)
    for k in K_GRID:
        x = np.exp(-n / k)
        a = float(np.dot(x, y) / np.dot(x, x))
        mse = float(np.mean((y - a * x) ** 2))
        fits.append((mse, k, a))
    mse, k, a = min(fits)
    print(f"\nsource decay fit: a={a:+.6f}, k={k}, mse={mse:.6f}, n={len(f):,}")

    yv, p0 = target[T].to_numpy(), target.pred.to_numpy()
    m = target.status.to_numpy() == "F_to_R"
    decay = np.zeros(len(target))
    decay[m] = a * np.exp(-target.loc[m, "cur_n"].to_numpy(np.float64) / k)
    # F→R 조정과 전역 수준을 직교화한다.
    decay -= decay.mean()

    # 정적 source gap도 같은 방식으로 비교한다.
    const = np.zeros(len(target))
    const[m] = float(y.mean())
    const -= const.mean()
    s0 = bss(yv, p0)
    sd = bss(yv, p0 + decay)
    sc = bss(yv, p0 + const)
    print(f"target centered: base={s0:.3f} decay={sd:.3f} ({sd-s0:+.3f}) "
          f"constant={sc:.3f} ({sc-s0:+.3f})")

    for label, mask in (("Mar-Jun", target.game_month <= 6),
                        ("Jul-Oct", target.game_month >= 7)):
        q = mask.to_numpy()
        print(f"  {label}: decay {bss(yv[q], p0[q]+decay[q])-bss(yv[q],p0[q]):+.3f} "
              f"constant {bss(yv[q],p0[q]+const[q])-bss(yv[q],p0[q]):+.3f}")
    return sd - s0, k, a


def main():
    d = prepare()
    source, target = attach_predictions(d)
    table(source, "source 2023 R")
    table(target, "target 2024 R")
    gain, k, a = fit_decay(source, target)
    print(f"\nGATE {'PASS' if gain >= 3 else 'FAIL'}: target decay gain={gain:+.3f} "
          f"(필요 +3), k={k}, a={a:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""미학습 시즌 예측의 세그먼트별 손실·해상도·정직한 순위 해상도를 보고한다.

`honest`는 타깃을 A/B로 나눠 A의 예측분위별 실제율을 B에 적용하고 반대 방향도
평가한 값이다. 같은 타깃으로 보정하고 평가하는 낙관 편의를 제거한다.
"""

import argparse
import glob

import numpy as np
import pandas as pd


def load(tag, split="val"):
    # Anchor both ends: a loose "*MVCELL*" also matches MVCELL22 (a different
    # season) and MVCELL5K. Fall back to the no-seed filename form.
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_{split}_preds.npz")
                or glob.glob(f"./out/*_{tag}_{split}_preds.npz"))
    if not fs:
        raise FileNotFoundError(f"예측 없음: {tag}")
    z = [np.load(f, allow_pickle=True) for f in fs]
    y = z[0]["y"].astype(np.float64)
    return np.mean([q["pred"] for q in z], axis=0).astype(np.float64), y, len(z)


def honest_rank_resolution(pred, y, global_base, rng, bins=10, k=100.0):
    if len(y) < bins * 100:
        return float("nan")
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    b = np.digitize(pred, edges[1:-1])
    half = rng.random(len(y)) < 0.5
    gains = []
    weights = []
    for fit, test in ((half, ~half), (~half, half)):
        r = y[fit].mean()
        ng = len(edges) - 1
        n = np.bincount(b[fit], minlength=ng)
        s = np.bincount(b[fit], weights=y[fit], minlength=ng)
        rate = (s + k * r) / (n + k)
        mse0 = np.mean((r - y[test]) ** 2)
        mse1 = np.mean((rate[b[test]] - y[test]) ** 2)
        gains.append(mse0 - mse1)
        weights.append(test.sum())
    return 1e5 * np.average(gains, weights=weights) / global_base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v13f")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    args = ap.parse_args()
    pred, y, nseed = load(args.tag, args.split)
    cols = ["season", "game_type", "game_month", "asof_pitcher_n",
            "asof_batter_n", "pitcher_id", "control_success"]
    full = pd.read_csv("./data/train.csv", encoding="utf-8-sig", usecols=cols)
    # "cold" = this pitcher did not appear in the previous season, so every
    # per-pitcher TE falls back to a shrunk prior. That is ~20% of 2024 rows and
    # its resolution has never been measured.
    prev_ids = set(full.loc[full.season == args.season - 1, "pitcher_id"])
    raw = full[full.season == args.season].reset_index(drop=True)
    raw["cold"] = np.where(raw.pitcher_id.isin(prev_ids), "warm", "cold")
    if len(raw) != len(y) or not np.array_equal(raw.control_success.to_numpy(), y):
        raise ValueError("원본과 예측 행 정렬 불일치")
    raw["league"] = raw.game_type
    raw["month"] = raw.game_month.astype(str)
    edges = [-np.inf, 200, 1000, 3000, 10000, np.inf]
    labels = ["~200", "~1k", "~3k", "~10k", "10k+"]
    raw["pitcher_exp"] = pd.cut(raw.asof_pitcher_n.fillna(-1), edges, labels=labels)
    raw["batter_exp"] = pd.cut(raw.asof_batter_n.fillna(-1), edges, labels=labels)
    global_base = y.mean() * (1 - y.mean())
    se = (pred - y) ** 2
    print(f"{args.tag} {nseed}시드 | {args.season} {len(y):,}행 | "
          f"BSS {1e5*(1-se.mean()/global_base):.2f}")
    print("해상도 열은 전체 BSS 좌표의 기여도다. 세그먼트 BSS로 손실을 판단하지 않는다.")
    for axis in ("cold", "league", "month", "pitcher_exp", "batter_exp"):
        rows = []
        for value, idx in raw.groupby(axis, observed=False).groups.items():
            idx = np.asarray(idx, dtype=int)
            if len(idx) < 500:
                continue
            ps, ys = pred[idx], y[idx]
            q = np.unique(np.quantile(ps, np.linspace(0, 1, 11)))
            bb = np.digitize(ps, q[1:-1])
            means = pd.DataFrame({"b": bb, "y": ys}).groupby("b").y.agg(["mean", "size"])
            empirical = np.sum(means["size"] * (means["mean"] - ys.mean()) ** 2) / len(ys)
            honest = honest_rank_resolution(ps, ys, global_base,
                                            np.random.default_rng(20260808))
            rows.append((str(value), len(idx), ys.mean(), ps.std(),
                         se[idx].sum() / se.sum(),
                         1e5 * (len(idx) / len(y)) * empirical / global_base,
                         (len(idx) / len(y)) * honest))
        print(f"\n[{axis}] {'값':<10}{'n':>9}{'rate':>9}{'pred_sd':>10}"
              f"{'loss%':>9}{'emp_res':>11}{'honest':>11}{'hon/norm':>11}")
        for value, n, rate, psd, loss, emp, honest in rows:
            print(f"{value:<10}{n:>9,}{rate:>9.4f}{psd:>10.4f}"
                  f"{loss:>9.1%}{emp:>11.1f}{honest:>11.1f}"
                  f"{honest/(n/len(y)):>11.1f}")


if __name__ == "__main__":
    main()

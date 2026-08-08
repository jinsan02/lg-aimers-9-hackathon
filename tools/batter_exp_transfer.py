"""타자 경험 버킷 효과가 2023에서 2024로 이전되는지 검정한다.

버킷 경계는 고정 절대값이다. 2024 분포로 qcut 하지 않는다. 2023 타깃으로 만든
상대 오프셋을 미학습 2024 예측에 beta=1로 그대로 더하는 것이 판정 열이다.
"""

import argparse
import glob

import numpy as np
import pandas as pd


EDGES = [-np.inf, 200, 1000, 3000, 10000, np.inf]
LABELS = ["~200", "~1k", "~3k", "~10k", "10k+"]


def load_preds(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_val_preds.npz"))
    if not fs:
        raise FileNotFoundError(f"예측 없음: {tag}")
    z = [np.load(f, allow_pickle=True) for f in fs]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError("시드 간 검증 라벨 불일치")
    return np.mean([q["pred"] for q in z], axis=0).astype(np.float64), y, len(z)


def bss(pred, y):
    base = y.mean() * (1 - y.mean())
    return 1e5 * (1 - np.mean((np.clip(pred, 0, 1) - y) ** 2) / base)


def offsets(df, league, k):
    d = df if league == "ALL" else df[df.game_type == league]
    rate = d.control_success.mean()
    g = d.groupby("_bn", observed=False).control_success.agg(["sum", "size"])
    g["offset"] = (g["sum"] - g["size"] * rate) / (g["size"] + k)
    return g, rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v13f")
    ap.add_argument("--season", type=int, default=2024)
    ap.add_argument("--k", type=float, default=100.0)
    args = ap.parse_args()

    pred, y, nseed = load_preds(args.tag)
    use = ["season", "game_type", "asof_batter_n", "control_success"]
    df = pd.read_csv("./data/train.csv", encoding="utf-8-sig", usecols=use)
    df["_bn"] = pd.cut(df.asof_batter_n.fillna(-1), EDGES, labels=LABELS,
                       include_lowest=True)
    cur = df[df.season == args.season].reset_index(drop=True)
    if len(cur) != len(y) or not np.array_equal(cur.control_success.to_numpy(), y):
        raise ValueError(f"행 정렬 불일치: raw {len(cur):,} vs pred {len(y):,}")
    past = df[df.season == args.season - 1]

    print(f"{args.tag} {nseed}시드 | {args.season-1}->{args.season} | k={args.k:g}")
    for league in ("ALL", "R", "F"):
        g23, _ = offsets(past, league, args.k)
        g24, _ = offsets(cur, league, args.k)
        tab = g23[["size", "offset"]].rename(
            columns={"size": "n_prev", "offset": "off_prev"}).join(
                g24[["size", "offset"]].rename(
                    columns={"size": "n_cur", "offset": "off_cur"}))
        mask = np.ones(len(cur), dtype=bool) if league == "ALL" else \
            cur.game_type.to_numpy() == league
        p0 = pred[mask] - (pred[mask].mean() - y[mask].mean())
        off = cur.loc[mask, "_bn"].map(g23["offset"]).astype(float).fillna(0).to_numpy()
        before = bss(p0, y[mask])
        after = bss(p0 + off, y[mask])
        valid = tab[["off_prev", "off_cur"]].dropna()
        corr = (np.corrcoef(valid.off_prev, valid.off_cur)[0, 1]
                if len(valid) > 1 else float("nan"))
        print(f"\n[{league}] n={mask.sum():,} corr={corr:+.3f} "
              f"beta=1 이전 이득={after-before:+.3f}")
        print((tab.assign(off_prev=lambda x: x.off_prev * 1000,
                          off_cur=lambda x: x.off_cur * 1000)
               .round({"off_prev": 2, "off_cur": 2}).to_string()))


if __name__ == "__main__":
    main()

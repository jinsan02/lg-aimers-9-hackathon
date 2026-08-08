"""미학습 시즌 전·후반 사이에서 설정 단위 블렌드 가중 전이성을 검문한다.

실행: python tools/blend_time_transfer.py AB_base DW_cell TMC1_expand
각 태그의 시드는 먼저 균등 평균한다. 전반기에 NNLS 가중을 적합해 후반기에
평가하고, 반대 방향도 반복한다. 수준은 각 평가 구간에서 제거해 해상도만 본다.
"""

import glob
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls


def load(tag):
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    if not fs:
        return None, None, 0
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], axis=0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def score(y, p):
    r = float(y.mean())
    p = p - (p.mean() - r)
    return float(1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean()
                            / (r * (1 - r))))


def weights(y, P):
    r = float(y.mean())
    C = P - (P.mean(0, keepdims=True) - r)
    big = 1e3 * float(np.abs(y).mean())
    A = np.vstack([C, np.full((1, C.shape[1]), big)])
    w, _ = nnls(A, np.concatenate([y, [big]]))
    return w / w.sum()


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        print(__doc__)
        return 1
    ps, keep, y = [], [], None
    for tag in tags:
        p, yy, n = load(tag)
        if p is None:
            print(f"없음: {tag}")
            continue
        ps.append(p)
        keep.append(tag)
        y = yy
        print(f"{tag}: {n}시드")
    P = np.column_stack(ps)

    raw = pd.read_csv("./data/train.csv", encoding="utf-8-sig",
                      usecols=["season", "game_month"])
    months = raw.loc[raw.season == 2024, "game_month"].to_numpy()
    if len(months) != len(y):
        raise ValueError(f"2024 행수 불일치: month {len(months)} vs pred {len(y)}")
    first = months <= 6

    def one(train_mask, test_mask, label):
        w = weights(y[train_mask], P[train_mask])
        s = score(y[test_mask], P[test_mask] @ w)
        print(f"{label}: test BSS {s:.2f} | "
              + " ".join(f"{t}={v:.3f}" for t, v in zip(keep, w)))
        return w, s

    w1, s1 = one(first, ~first, "전반→후반")
    w2, s2 = one(~first, first, "후반→전반")
    wf = (w1 + w2) / 2
    print("양방향 평균 가중: " + " ".join(
        f"{t}={v:.3f}" for t, v in zip(keep, wf)))
    print(f"전체 참고 BSS {score(y, P @ wf):.2f} | 양방향 평균 {np.mean([s1, s2]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

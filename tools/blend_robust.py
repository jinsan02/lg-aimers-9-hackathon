"""제안 ④ 검증 — greedy 선택의 검증 과적합 여부.

greedy는 27개 후보에서 검증셋으로 가중치를 골랐다. 그 이득이 진짜인지,
아니면 검증셋에 맞춘 것인지 **홀드아웃 분할 검정**으로 확인한다.

방법: 2024 검증셋을 전반기(3~6월) / 후반기(7~10월)로 나눠
      전반기로 greedy 가중치를 정하고 **후반기에서 평가**.
      비교군: 단순 평균 (가중치 학습 없음).
"""

import sys

import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def greedy(y, preds, steps=50):
    chosen, cur = [], np.zeros_like(y)
    hist = []
    for _ in range(steps):
        best_i, best_s = None, -np.inf
        for i, p in enumerate(preds):
            s = bss(y, (cur + p) / (len(chosen) + 1))
            if s > best_s:
                best_i, best_s = i, s
        if hist and best_s <= hist[-1] + 1e-9:
            break
        chosen.append(best_i)
        cur = cur + preds[best_i]
        hist.append(best_s)
    w = np.bincount(chosen, minlength=len(preds)) / max(len(chosen), 1)
    return w


def main():
    paths = sys.argv[1:]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["season", "game_month", T])
    v = df[df.season == 2024].reset_index(drop=True)
    first = (v.game_month <= 6).to_numpy()

    y = None
    preds, names = [], []
    for p in paths:
        d = np.load(p)
        if y is None:
            y = d["y"]
        pr = d["pred"] if "pred" in d else d["ensemble"]
        preds.append(pr.astype(np.float64))
        names.append(p.split("/")[-1].replace("_val_preds.npz", ""))
    assert len(y) == len(v), f"행수 불일치 {len(y)} vs {len(v)}"
    P = np.array(preds)

    # 개별 상위 N개 단순 평균 (전반기 기준 선발)
    solo_first = np.array([bss(y[first], p[first]) for p in P])
    order = np.argsort(-solo_first)

    print(f"검증 분할: 전반기 {first.sum()} / 후반기 {(~first).sum()}\n")
    print("=== 후반기 성능 (전반기로 정한 방식들) ===")

    w = greedy(y[first], [p[first] for p in P])
    g_second = bss(y[~first], (P[:, ~first] * w[:, None]).sum(0))
    print(f"greedy(전반기 학습)   후반기 BSS {g_second:8.2f}")

    for n in [3, 5, 7, 10, 15]:
        sel = order[:n]
        s = bss(y[~first], P[sel][:, ~first].mean(0))
        print(f"상위 {n:2d}개 단순평균     후반기 BSS {s:8.2f}")

    print(f"\n참고: 전체 검증 기준 greedy 가중 = "
          f"{bss(y, (P * w[:, None]).sum(0)):.2f}")
    top = order[:7]
    print(f"      상위 7개 단순평균(전체) = {bss(y, P[top].mean(0)):.2f}")
    print(f"      선발된 상위 7: {[names[i] for i in top]}")


if __name__ == "__main__":
    main()

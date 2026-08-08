"""다양성 우선 블렌드 — greedy가 노이즈에 흔들리는 문제 대응.

greedy는 검증셋 점수를 최대화하지만 GPU 비결정성·시드 노이즈(±10)에 흔들린다.
대안: **서로 다른 피처 세트/구조의 모델을 균등 가중**해 안정성을 얻는다.

비교:
  A) greedy 가중 (현행)
  B) 다양성 균등 평균 (피처 계열별 대표 1개씩)
  C) 전체 균등 평균
후반기 홀드아웃(전반기로 선택 → 후반기 평가)으로 일반화까지 확인.
"""

import sys

import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def greedy_w(y, P, steps=40):
    chosen, cur, hist = [], np.zeros_like(y), []
    for _ in range(steps):
        best_i, best_s = None, -np.inf
        for i, p in enumerate(P):
            s = bss(y, (cur + p) / (len(chosen) + 1))
            if s > best_s:
                best_i, best_s = i, s
        if hist and best_s <= hist[-1] + 1e-9:
            break
        chosen.append(best_i)
        cur = cur + P[best_i]
        hist.append(best_s)
    return np.bincount(chosen, minlength=len(P)) / max(len(chosen), 1)


def main():
    paths = sys.argv[1:]
    y = None
    P, names = [], []
    for p in paths:
        d = np.load(p)
        if y is None:
            y = d["y"]
        P.append((d["pred"] if "pred" in d else d["ensemble"]).astype(np.float64))
        names.append(p.split("/")[-1].replace("_val_preds.npz", ""))
    P = np.array(P)

    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["season", "game_month"])
    v = df[df.season == 2024].reset_index(drop=True)
    first = (v.game_month <= 6).to_numpy()

    print(f"모델 {len(P)}개 | 전반기 {first.sum()} / 후반기 {(~first).sum()}\n")
    print("=== 전체 검증 기준 ===")
    w = greedy_w(y, P)
    print(f"  A) greedy 가중      {bss(y, (P * w[:, None]).sum(0)):8.2f}")
    print(f"  C) 전체 균등 평균    {bss(y, P.mean(0)):8.2f}")

    # 계열별 대표 1개 — 이름 접두사로 그룹
    groups = {}
    for i, n in enumerate(names):
        key = ("hs" if n.startswith("cat_hs") else
               "role" if "role" in n else
               "fat" if "fat" in n else
               "mgr" if "mgr" in n else
               "xgb" if n.startswith("xgb") else
               "pt" if "pt" in n or "ptype" in n else
               "tmc" if "tmc" in n else "base")
        groups.setdefault(key, []).append(i)
    reps = [max(g, key=lambda i: bss(y, P[i])) for g in groups.values()]
    print(f"  B) 다양성 균등({len(reps)}종) {bss(y, P[reps].mean(0)):8.2f}"
          f"  ← {[names[i] for i in reps]}")

    print("\n=== 일반화 검정 (전반기로 결정 → 후반기 평가) ===")
    w2 = greedy_w(y[first], P[:, first])
    print(f"  A) greedy      후반기 {bss(y[~first], (P[:, ~first] * w2[:, None]).sum(0)):8.2f}")
    print(f"  B) 다양성 균등  후반기 {bss(y[~first], P[reps][:, ~first].mean(0)):8.2f}")
    print(f"  C) 전체 균등    후반기 {bss(y[~first], P[:, ~first].mean(0)):8.2f}")


if __name__ == "__main__":
    main()

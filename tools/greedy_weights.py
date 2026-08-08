"""2024 홀드아웃 val 예측으로 greedy ensemble selection (with replacement).

사용: python tools/greedy_weights.py n1_d7l10 n2_d8l3 ... [--rounds 20]
입력: out/{model}_{tag}_val_preds.npz  (train_gbdt2.py 가 자동 저장)
출력: 정규화된 가중치 — script_blend_*.py 의 WEIGHTS 에 그대로 붙인다.

E86 근거: 실제 2025 평가에서 greedy(861.66)가 다양성 균등(838.13)을 이겼다.
E84의 "균등이 일반화 우세" 결론은 같은 시즌 내 분할이라 레짐 전환을 포함하지 못했고,
실제 LB에서 재현되지 않았다.
"""

import argparse
import glob
import os
import sys

import numpy as np


def bss(y, p):
    r = y.mean()
    return 100000 * (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / (r * (1 - r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--rounds", type=int, default=20)
    a = ap.parse_args()

    preds, names, y = [], [], None
    for t in a.tags:
        hits = glob.glob(f"./out/*_{t}_val_preds.npz")
        if not hits:
            print(f"  건너뜀(파일 없음): {t}")
            continue
        z = np.load(hits[0])
        if y is None:
            y = z["y"]
        elif len(z["y"]) != len(y):
            raise ValueError(f"{t}: 검증 행수 불일치 {len(z['y'])} != {len(y)}")
        preds.append(z["pred"])
        names.append(t)
    if not preds:
        raise SystemExit("사용 가능한 val_preds 없음")
    P = np.vstack(preds)
    print(f"후보 {len(names)}개 / 검증 {len(y):,}행 | r={y.mean():.4f}\n")
    for n, p in zip(names, P):
        print(f"  {n:16s} 단독 {bss(y, p):8.2f}")

    # greedy with replacement
    counts = np.zeros(len(names))
    acc = np.zeros(len(y))
    print()
    for i in range(a.rounds):
        scores = [bss(y, (acc * i + p) / (i + 1)) for p in P]
        j = int(np.argmax(scores))
        acc = (acc * i + P[j]) / (i + 1)
        counts[j] += 1
        print(f"  라운드 {i + 1:2d}: +{names[j]:16s} → {scores[j]:8.2f}")

    w = counts / counts.sum()
    print(f"\n최종 블렌드 BSS {bss(y, acc):.2f}  "
          f"(균등 평균은 {bss(y, P.mean(0)):.2f})")
    print("\nWEIGHTS = [")
    for n, ww in sorted(zip(names, w), key=lambda t: -t[1]):
        if ww > 0:
            kind = "xgb" if "xgb" in n else "cat"
            print(f'           ("./model/{kind}_{n}.pkl", {ww:.3f}),')
    print("]")


if __name__ == "__main__":
    sys.exit(main())

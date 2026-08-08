"""설정 단위 블렌드 — 시드는 균등 평균, 설정끼리만 greedy.

왜: greedy를 시드 수준에서 돌리면 **2024에서 우연히 잘 나온 시드**에 큰 가중이 붙는다
(실측: 9시드 중 809.28짜리 하나에 0.40). 시드는 다양성이 아니라 노이즈이므로
2025에서 그 시드가 나을 이유가 없다 — 전형적인 검증셋 과적합이다.

올바른 처리:
  1) 같은 하이퍼파라미터의 시드들은 **균등 평균**(분산 감소, 편향 없음)
  2) 서로 다른 설정(깊이/정규화/알고리즘/피처)끼리만 greedy — 여긴 진짜 다양성이 있다
"""

import glob
import os
import sys

import numpy as np

# 설정 이름 -> 그 설정에 속한 태그들 (시드 변형)
# v7 = lr 0.02 + 층화 TE + dev (E95, TE없음 대비 +27.24 t=4.7) + 2024x3 refit
SEEDS_A = [42, 7, 13, 3, 4, 5, 6, 8]
GROUPS = {
    # v11/v12 = k80 + lr 0.01 + **최신시즌 가중 제거**(E85 철회).
    # 검증 구조와 제출 구조가 일치하는 첫 세대이므로 제출 블렌드는 여기서만 고른다.
    "F11_lr01_dom": [f"v11f_s{s}" for s in (42, 7, 13, 3, 4, 5, 6, 8)],
    "G12_lr01_nodom": [f"v12g_s{s}" for s in (42, 7, 13, 3, 4, 5)],
    "A11_d8_lr02": [f"v11a_s{s}" for s in (42, 7, 13, 3, 4, 5, 6, 8)],
    "B11_d7_lr02": [f"v11b_s{s}" for s in (42, 7, 13)],
}


def bss(y, p):
    r = y.mean()
    return 100000 * (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / (r * (1 - r)))


def load(tag):
    hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
    return np.load(hits[0]) if hits else None


def model_path(tag):
    """예측 파일명에서 실제 접두어(cat/xgb)를 읽어 모델 경로를 만든다.
    태그 문자열에 'xgb'가 들어있는지로 판단하면 v7x_s42 같은 태그에서 틀린다."""
    hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
    base = os.path.basename(hits[0]).replace("_val_preds.npz", "")
    return f"./model/{base}.pkl"


def main():
    y, group_pred, members = None, {}, {}
    for g, tags in GROUPS.items():
        ps = []
        keep = []
        for t in tags:
            z = load(t)
            if z is None:
                print(f"  없음: {t}")
                continue
            if y is None:
                y = z["y"]
            ps.append(z["pred"])
            keep.append(t)
        if ps:
            group_pred[g] = np.mean(ps, 0)
            members[g] = keep

    print(f"검증 {len(y):,}행 | r={y.mean():.4f}\n")
    print("=== 설정별 (시드 균등평균) ===")
    for g, p in group_pred.items():
        singles = [bss(y, load(t)["pred"]) for t in members[g]]
        print(f"  {g:18s} 시드 {len(members[g])}개 | 개별 "
              f"{min(singles):7.2f}~{max(singles):7.2f} (평균 {np.mean(singles):7.2f})"
              f" | **평균본 {bss(y, p):7.2f}**")

    names = list(group_pred)
    P = np.vstack([group_pred[g] for g in names])
    counts = np.zeros(len(names))
    acc = np.zeros(len(y))
    print("\n=== 설정 단위 greedy ===")
    for i in range(12):
        sc = [bss(y, (acc * i + p) / (i + 1)) for p in P]
        j = int(np.argmax(sc))
        acc = (acc * i + P[j]) / (i + 1)
        counts[j] += 1
        print(f"  {i + 1:2d}: +{names[j]:18s} -> {sc[j]:7.2f}")

    w = counts / counts.sum()
    print(f"\n설정 greedy 최종 {bss(y, acc):.2f} | 설정 균등 {bss(y, P.mean(0)):.2f}")
    print("\n최종 모델별 가중 (설정가중 / 그 설정의 시드수):")
    out = []
    for g, gw in zip(names, w):
        if gw <= 0:
            continue
        for t in members[g]:
            out.append((model_path(t), gw / len(members[g])))
    print("WEIGHTS = [")
    for path, ww in sorted(out, key=lambda t: -t[1]):
        print(f'           ("{path}", {ww:.4f}),')
    print("]")
    print(f"모델 {len(out)}개")


if __name__ == "__main__":
    sys.exit(main())

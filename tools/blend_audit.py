"""블렌드 구성원 감사 - 중복/무용 모델 색출.

물음: 지금 블렌드에 들어간 모델 중 서로 사실상 같은 것(중복)이나
      빼도 점수가 안 떨어지는 것(무용)이 있는가?

방법:
  1) 구성원 간 예측 상관 - 0.999+ 면 사실상 같은 모델
  2) leave-one-out - 그 모델을 빼고 나머지로 다시 정규화했을 때 BSS 변화
     (+면 그 모델이 오히려 해가 됨)
  3) 한계 기여 - 그 모델 하나만 추가로 넣었을 때의 개선
"""

import glob
import sys

import numpy as np


def bss(y, p):
    r = y.mean()
    return 100000 * (1 - ((np.clip(p, 1e-6, 1 - 1e-6) - y) ** 2).mean() / (r * (1 - r)))


def load(tag):
    hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
    if not hits:
        return None
    return np.load(hits[0])


def main(members):
    """members: [(tag, weight), ...]"""
    y = None
    tags, W, P = [], [], []
    for t, w in members:
        z = load(t)
        if z is None:
            print(f"  건너뜀(없음): {t}")
            continue
        if y is None:
            y = z["y"]
        tags.append(t)
        W.append(w)
        P.append(z["pred"])
    P = np.vstack(P)
    W = np.array(W, float)
    W = W / W.sum()
    full = W @ P
    print(f"구성원 {len(tags)}개 | 검증 {len(y):,}행 | 블렌드 BSS {bss(y, full):.2f}\n")

    print("=== 1) 예측 상관 (0.999+ = 사실상 중복) ===")
    C = np.corrcoef(P)
    dup = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            if C[i, j] >= 0.995:
                dup.append((C[i, j], tags[i], tags[j]))
    for c, a, b in sorted(dup, reverse=True)[:15]:
        print(f"  {c:.5f}  {a} <-> {b}")
    if not dup:
        print("  0.995 이상 쌍 없음 - 중복 없음")
    off = C[np.triu_indices(len(tags), 1)]
    print(f"  상관 분포: 최소 {off.min():.4f} 중앙 {np.median(off):.4f} 최대 {off.max():.4f}")

    print("\n=== 2) leave-one-out (양수 = 빼는 게 이득 = 무용/유해) ===")
    rows = []
    for i, t in enumerate(tags):
        m = np.ones(len(tags), bool)
        m[i] = False
        w2 = W[m] / W[m].sum()
        rows.append((bss(y, w2 @ P[m]) - bss(y, full), t, W[i], bss(y, P[i])))
    for d, t, w, s in sorted(rows, reverse=True):
        mark = "  <-- 빼는 게 낫다" if d > 0 else ""
        print(f"  {d:+7.2f}  {t:22s} (가중 {w:.3f}, 단독 {s:7.2f}){mark}")

    print("\n=== 3) 균등 대비 ===")
    print(f"  가중 블렌드 {bss(y, full):.2f} | 균등 {bss(y, P.mean(0)):.2f}")
    keep = [t for d, t, _, _ in rows if d <= 0]
    if len(keep) < len(tags):
        idx = [tags.index(t) for t in keep]
        w3 = W[idx] / W[idx].sum()
        print(f"  유해 제거 후({len(keep)}개) {bss(y, w3 @ P[idx]):.2f}")


if __name__ == "__main__":
    from script_blend_v5 import WEIGHTS
    mem = [(p.split("/")[-1].replace(".pkl", "").split("_", 1)[1], w)
           for p, w in WEIGHTS]
    sys.exit(main(mem))

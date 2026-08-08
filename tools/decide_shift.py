"""최종 블렌드의 수준 편향을 재고 제출용 SHIFT를 원칙적으로 정한다 (E96/E102).

배경: 한 시즌 앞을 예측하면 모델이 전 구간에서 균일하게 과대예측한다.
  원인 두 가지 -
  (1) 드리프트 미반영: 학습 구간 밖 시즌이라 season 피처가 외삽된다.
      제출과 같은 구조로 잰 값 H22 +0.0029 / H23 +0.0011 / H24 +0.0066.
  (2) std 피처의 수축 목표가 예측 시즌보다 높다 (E102 에서 구조적으로 제거).

상수 시프트 c의 Brier 이득은 실제 편향 b에 대해
    이득 = (2bc - c^2) / (r(1-r)) * 100000
이고 0 < c < 2b 이면 항상 이득이다. b는 알 수 없으므로 **측정값의 절반**을
쓴다: c = 0.5*b_hat 이면 최대 이득의 75%를 얻고, 실제 b가 측정값의 25%
이상이기만 하면 손해가 나지 않는다.

train(2024 홀드아웃)만으로 정한 상수를 각 행에 동일 적용하므로
'평가 데이터 분포를 이용한 후처리' 금지 규칙에 저촉되지 않는다.

사용: python tools/decide_shift.py            # script_blend_v6.py 의 WEIGHTS 사용
"""

import glob
import sys

import numpy as np


def main():
    from script_blend_v6 import WEIGHTS
    P, W, y = [], [], None
    miss = []
    for path, w in WEIGHTS:
        tag = path.split("/")[-1].replace(".pkl", "").split("_", 1)[1]
        hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
        if not hits:
            miss.append(tag)
            continue
        z = np.load(hits[0])
        y = z["y"] if y is None else y
        P.append(z["pred"])
        W.append(w)
    if miss:
        print(f"경고 - 예측 파일 없는 모델 {len(miss)}개: {miss}")
    W = np.array(W, float)
    W /= W.sum()
    p = W @ np.vstack(P)
    r = y.mean()
    base = r * (1 - r)
    b = p.mean() - r

    def bss(q):
        return 100000 * (1 - ((np.clip(q, 1e-6, 1 - 1e-6) - y) ** 2).mean() / base)

    print(f"블렌드 {len(P)}개 | 검증 {len(y):,}행")
    print(f"  실제 {r:.4f}  예측 {p.mean():.4f}  **편향 {b:+.5f}**")
    print(f"  보정 없음 BSS {bss(p):.2f}")
    print(f"  편향 완전제거(오라클) {bss(p - b):.2f}  (+{bss(p - b) - bss(p):.1f})\n")

    print(f"{'c':>9}{'2024 실측 이득':>16}   실제 편향 b 가정별 기대이득")
    hdr = "".join(f"{f'b={x:.4f}':>11}" for x in (0.002, 0.004, 0.006, 0.008, b))
    print(f"{'':>9}{'':>16}{hdr}")
    for a in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        c = a * b
        row = f"{c:9.5f}{bss(p - c) - bss(p):16.1f}   "
        for bb in (0.002, 0.004, 0.006, 0.008, b):
            row += f"{(2 * bb * c - c * c) / base * 100000:11.1f}"
        print(row + f"  ({a:.0%})")
    print(f"\n권장: c = 0.5 x {b:.5f} = **{0.5 * b:.5f}**")
    print("  (최대 이득의 75%를 얻고, 실제 편향이 측정값의 25% 이상이면 손해 없음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

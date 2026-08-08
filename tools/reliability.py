"""신뢰도 곡선 — 남은 손실이 **교정(calibration)인가 변별(discrimination)인가** (E168).

Brier 는 분해된다:  MSE = 신뢰도(reliability) - 해상도(resolution) + 불확실성
  reliability : 예측 p 를 낸 구간의 실제 성공률이 p 와 얼마나 어긋나는가 (고칠 수 있음)
  resolution  : 예측이 전체 평균에서 얼마나 벌어지는가 (모델의 실력)

교정 오차가 크면 **단조 사상 하나로** 회수된다. 작으면 모델을 더 좋게 만드는 수밖에 없다.
E30 에서 isotonic 이 -21 로 실패했으니 이미 잘 교정돼 있을 것으로 예상되지만,
그건 자기검증 표면 이야기였고 여기서는 **미학습 시즌**에서 잰다.

실행: python tools/reliability.py AB_base
"""

import glob
import sys

import numpy as np


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "AB_base"
    fs = sorted(glob.glob(f"./out/*_{tag}_s*_test_preds.npz"))
    if not fs:
        print(f"없음: {tag}")
        return 1
    z = [np.load(f, allow_pickle=True) for f in fs]
    p = np.mean([q["pred"] for q in z], 0).astype(np.float64)
    y = z[0]["y"].astype(np.float64)
    r = float(y.mean())
    base = r * (1 - r)
    mse = float(((p - y) ** 2).mean())
    print(f"{tag}: {len(y):,}행 | BSS {1e5 * (1 - mse / base):.1f} | "
          f"예측 {p.min():.4f}~{p.max():.4f} (sd {p.std():.4f}) | 실제 {r:.4f}\n")

    q = np.quantile(p, np.linspace(0, 1, 21))
    q[0] -= 1e-9
    b = np.digitize(p, q[1:-1])
    print(f"{'구간':>5}{'행수':>9}{'예측평균':>10}{'실제':>9}{'차이':>9}"
          f"{'누적 기여':>11}")
    rel = 0.0
    res = 0.0
    for i in range(20):
        m = b == i
        if m.sum() < 100:
            continue
        pm, ym = p[m].mean(), y[m].mean()
        rel += m.sum() * (pm - ym) ** 2
        res += m.sum() * (ym - r) ** 2
        print(f"{i:>5}{m.sum():>9,}{pm:>10.4f}{ym:>9.4f}{pm - ym:>+9.4f}"
              f"{1e5 * m.sum() * (pm - ym) ** 2 / len(y) / base:>11.2f}")
    rel /= len(y)
    res /= len(y)
    print(f"\n분해 (20분위)")
    print(f"  신뢰도(고칠 수 있음)  {rel:.3e}  →  BSS 상당 {1e5 * rel / base:+.1f}")
    print(f"  해상도(모델 실력)     {res:.3e}  →  BSS 상당 {1e5 * res / base:+.1f}")
    print(f"  검산: BSS ≈ {1e5 * (res - rel) / base:.1f} "
          f"(실측 {1e5 * (1 - mse / base):.1f})")
    print("\n※ 신뢰도 항이 크면 단조 사상 하나로 회수된다. 작으면 모델을 고쳐야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

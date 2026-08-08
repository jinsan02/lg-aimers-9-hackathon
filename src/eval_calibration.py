"""기저율 보정 백테스트 — 2024 홀드아웃에서 전역 로짓 시프트 효과 측정.

시나리오: 2019~2023 시즌별 성공률로 추세를 적합해 2024 기저율을 예측하고,
모델 예측 평균이 그 값이 되도록 로짓 시프트. (train 정보만 사용 — 규칙 적합)
oracle(실제 2024 기저율로 보정)은 상한선 참고용.

실행: python src/eval_calibration.py out/cat_l2_3_val_preds.npz
"""

import sys

import numpy as np
import pandas as pd


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift_to_mean(p, target_mean):
    """이분 탐색으로 로짓 시프트 d를 찾아 mean(sigmoid(logit(p)+d)) = target."""
    z = logit(p)
    lo, hi = -2.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        m = (1 / (1 + np.exp(-(z + mid)))).mean()
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))


def main():
    d = np.load(sys.argv[1])
    y, p = d["y"], d["pred"]
    print(f"원본: BSS {bss(y, p):8.2f} | pred mean {p.mean():.4f} "
          f"| 실제 r {y.mean():.4f}")

    # train 시즌별 성공률 (2019~2023) → 2024 추세 예측
    rates = pd.Series({2019: 0.5647, 2020: 0.5327, 2021: 0.5328,
                       2022: 0.5289, 2023: 0.5000})
    for name, r_hat in {
        "선형추세(19-23)": np.polyval(np.polyfit(rates.index, rates.values, 1), 2024),
        "최근2년추세": np.polyval(np.polyfit([2022, 2023], [0.5289, 0.5000], 1), 2024),
        "2023 그대로": 0.5000,
        "oracle(실제)": y.mean(),
    }.items():
        q = shift_to_mean(p, r_hat)
        print(f"{name:14s} r_hat={r_hat:.4f} → BSS {bss(y, q):8.2f} "
              f"({bss(y, q) - bss(y, p):+.2f})")


if __name__ == "__main__":
    main()

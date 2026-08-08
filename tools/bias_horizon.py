"""호라이즌별 예측 편향 측정 -> 제출용 SHIFT 상수 결정 (E96).

설정: 제출본과 **정확히 같은 구조**로 학습한 모델의 다음 시즌 예측 편향을 잰다.
  H22: 2019~2021 학습(2021 x3 가중) -> 2022 예측
  H23: 2019~2022 학습(2022 x3)      -> 2023 예측
  H24: 2019~2023 학습(2023 x3)      -> 2024 예측
  (제출본은 2019~2024 학습(2024 x3) -> 2025 예측)

상수 시프트 c의 Brier 이득은 실제 편향 b에 대해
    이득 = (2*b*c - c^2) / (r(1-r)) * 100000
이므로 0 < c < 2b 이면 항상 이득이다. c = alpha*b_hat 로 부분 보정하면
b_hat 이 과대추정이어도 안전 범위가 넓다.
"""

import glob
import sys

import numpy as np


def load(tag):
    hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
    return np.load(hits[0]) if hits else None


def main():
    seeds = (42, 7, 13)
    print(f"{'호라이즌':<8}{'시드':>5}{'행수':>10}{'실제':>9}{'예측':>9}{'편향':>10}{'BSS':>9}")
    bias = {}
    for h, yr in [("H22", 2022), ("H23", 2023), ("H24", 2024)]:
        bs = []
        for s in seeds:
            z = load(f"{h}_s{s}")
            if z is None:
                continue
            y, p = z["y"], z["pred"]
            r = y.mean()
            b = p.mean() - r
            sc = 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))
            print(f"{h}({yr})  {s:>5}{len(y):>10,}{r:>9.4f}{p.mean():>9.4f}"
                  f"{b:>+10.4f}{sc:>9.1f}")
            bs.append(b)
        if bs:
            bias[h] = np.array(bs)
    print()
    for h, v in bias.items():
        print(f"  {h}: 편향 평균 {v.mean():+.5f}  표준오차 {v.std(ddof=1) / np.sqrt(len(v)):.5f}"
              f"  (시드별 {[f'{x:+.4f}' for x in v]})")

    if "H24" not in bias:
        print("\nH24 없음 - 판정 불가")
        return 1
    allb = np.concatenate(list(bias.values()))
    print(f"\n  전 호라이즌 평균 {allb.mean():+.5f} / H24만 {bias['H24'].mean():+.5f}")

    r = 0.4861
    base = r * (1 - r)
    print("\n=== 보정폭 c 에 따른 이득 (실제 편향 b 가정별, BSS) ===")
    hats = sorted({round(float(bias["H24"].mean()), 5),
                   round(float(allb.mean()), 5)})
    print(f"{'c':>9}" + "".join(f"{f'b={b:.4f}':>12}" for b in
                                [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]))
    for a in (0.3, 0.5, 0.7, 1.0):
        c = a * bias["H24"].mean()
        row = f"{c:>9.5f}"
        for b in [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]:
            row += f"{(2 * b * c - c * c) / base * 100000:>12.1f}"
        print(row + f"   (H24 x {a:.1f})")
    print(f"\n  후보 b_hat: {hats}")
    print("  * 표에서 열=실제 편향, 행=적용할 보정폭. 음수 칸이 손해 구간이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

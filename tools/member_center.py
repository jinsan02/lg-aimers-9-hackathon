"""멤버별 수준을 맞춘 뒤 블렌드하면 이득이 있는가 (GPU 0).

발견: 같은 후보를 **원점수**로 재면 이득 5.83, **편향제거 후**로 재면 10.74 다.
차이는 전부 '멤버마다 수준(평균 예측)이 다르다'에서 온다.

지금 파이프라인은 블렌드를 **먼저** 하고 전역 SHIFT 하나를 뺀다. 그러면 멤버 간
수준 차이는 보정되지 않고, 블렌드 가중이 그 차이를 상쇄하느라 낭비된다.

고치는 법 — 멤버마다 홀드아웃 편향 b_i 를 재서 **상대 편차만** 제거한다:

    p_i' = p_i - (b_i - b_mean)

전역 수준은 건드리지 않으므로 SHIFT(0.65 x 편향) 규칙은 그대로 유효하다.
b_i 는 train 홀드아웃에서 정한 **모델별 상수**이고 모든 행에 동일 적용하므로
행 독립 원칙을 지킨다. (전부 빼버리면 2025 에 과보정된다 — 그래서 상대 편차만.)

실행: python tools/member_center.py "<기준>" "<후보1>" ...
"""

import glob
import sys

import numpy as np


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        return None, None
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def main():
    pats = sys.argv[1:]
    if len(pats) < 2:
        print(__doc__)
        return 1
    ps, y = [], None
    for p in pats:
        v, yy = load(p)
        if v is None:
            print(f"없음: {p}")
            return 1
        ps.append(v)
        y = yy
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    P = np.column_stack(ps)
    bias = P.mean(0) - r
    print(f"{len(y):,}행 | 실제 {r:.4f}")
    print(f"{'멤버':<18}{'예측평균':>10}{'편향':>10}")
    for n, b in zip(pats, bias):
        print(f"{n:<18}{r + b:>10.4f}{b:>+10.5f}")
    print(f"편향 표준편차 {bias.std():.5f}  (0 이면 중심맞춤 이득 없음)\n")

    from scipy.optimize import nnls

    def best_w(M):
        big = 1e3 * np.abs(y).mean()
        A = np.vstack([M, np.full((1, M.shape[1]), big)])
        w, _ = nnls(A, np.concatenate([y, [big]]))
        s = w.sum()
        return w / s if s > 0 else np.full(M.shape[1], 1.0 / M.shape[1])

    Pc = P - (bias - bias.mean())          # 상대 편차만 제거
    for nm, M in (("① 그대로 블렌드", P), ("② 멤버 중심맞춤 후", Pc)):
        w = best_w(M)
        p = M @ w
        b = p.mean() - r
        shift = 0.65 * b
        print(f"{nm:<20} 블렌드 {bss(p):8.2f} | 편향 {b:+.5f} | "
              f"SHIFT {shift:.4f} -> {bss(np.clip(p - shift, 0, 1)):8.2f}")
    print("\n※ ②가 ①보다 높으면 멤버별 상수 하나씩으로 공짜 이득이다.")
    print("   가중은 2024 자기적합이라 절대값은 낙관이고, **차이**만 신뢰할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

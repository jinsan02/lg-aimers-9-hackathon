"""v15 블렌드의 가중·기울기·SHIFT 를 정한다.

decide_v14 와 다른 점: **가중을 2023R 에서 정하고 2024 에서 확인한다.**
2024 자기적합 최적(NNLS 0.47)은 낙관치다 — 오늘 세운 시즌 이전 원칙에 따라
다른 시즌에서 정한 값이 넘어오는지를 먼저 본다.

순서: ① 가중(2023R 결정 → 2024 확인) ② 기울기(2023 적합) ③ SHIFT(기울기 후 편향 x 0.65)
기울기를 걸면 평균이 움직이므로 SHIFT 는 반드시 기울기 뒤에 정한다.

실행: python tools/decide_v15.py
"""

import glob
import sys

import numpy as np

Y2_FACTOR = 0.65       # ABS 2년차 복귀 계수 (2024 는 ABS 1년차라 하락폭이 컸다)
SLOPE_FIT_ON = "W_base_s*"


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        return None, None, 0
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64), len(fs))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_slope(p, y):
    X = np.column_stack([np.ones(len(p)), _logit(p)])
    w = np.array([0.0, 1.0])
    for _ in range(30):
        q = 1 / (1 + np.exp(-X @ w))
        W = q * (1 - q)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - q))
    return float(w[1])


def scorer(y):
    r = float(y.mean())
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)
    return bss, r


def sweep(pb, pc, y, label):
    bss, _ = scorer(y)
    ws = np.round(np.arange(0.0, 0.71, 0.05), 2)
    vals = [bss((1 - w) * pb + w * pc) for w in ws]
    best = int(np.argmax(vals))
    print(f"  {label}: " + " ".join(f"{w:.2f}:{v - vals[0]:+.1f}"
                                    for w, v in zip(ws, vals)))
    print(f"  → 최적 w {ws[best]:.2f} (이득 {vals[best] - vals[0]:+.2f})")
    return ws, np.array(vals), float(ws[best])


def main():
    b23, y23, n23b = load("R23B_s*")
    c23, _, n23c = load("R23C_s*")
    b24, y24, n24b = load("v14f_s*")
    c24, _, n24c = load("ZD5_s*")
    for nm, v in (("R23B", b23), ("R23C", c23), ("v14f", b24), ("ZD5", c24)):
        if v is None:
            print(f"없음: {nm}")
            return 1
    print(f"멤버 시드수  2023R base {n23b} cell {n23c} | "
          f"2024 base {n24b} cell {n24c}")

    print("\n① 가중 — 2023R 에서 결정")
    _, _, w23 = sweep(b23, c23, y23, "2023R")
    print("   같은 축을 2024 에서 확인 (자기적합이므로 최적값은 참고만)")
    ws, v24, w24 = sweep(b24, c24, y24, "2024 ")

    bss24, r = scorer(y24)
    g23 = bss24((1 - w23) * b24 + w23 * c24) - bss24(b24)
    print(f"\n   2023R 최적 {w23:.2f} 를 2024 에 적용 → {g23:+.2f}")
    print(f"   2024 자기적합 최적 {w24:.2f} → {v24.max() - v24[0]:+.2f} (낙관)")
    # 두 시즌 최적의 사이, 그리고 두 시즌 모두 양수인 구간을 고른다
    W = round(min(w23, w24) * 0.5 + max(w23, w24) * 0.5, 2)
    print(f"   → 채택 w = {W:.2f} (두 시즌 최적의 중점, 최적화가 아니라 강건성)")

    p = (1 - W) * b24 + W * c24
    print(f"\n   블렌드 2024      {bss24(p):8.2f}  (base 단독 {bss24(b24):.2f})")

    pa, ya, na = load(SLOPE_FIT_ON)
    if pa is None:
        print(f"② 기울기 적합용 {SLOPE_FIT_ON} 없음 — 2023R base 로 대체")
        pa, ya, na = b23, y23, n23b
    slope = fit_slope(pa, ya)
    p2 = 1 / (1 + np.exp(-slope * _logit(p)))
    print(f"② 기울기 (2023 {na}시드 적합) {slope:.4f} → {bss24(p2):8.2f}  "
          f"({bss24(p2) - bss24(p):+.2f})")

    b = float(p2.mean() - r)
    shift = Y2_FACTOR * b
    p3 = np.clip(p2 - shift, 0, 1)
    print(f"③ 기울기 후 편향 {b:+.5f} x {Y2_FACTOR} → SHIFT {shift:.4f} "
          f"→ {bss24(p3):8.2f}  ({bss24(p3) - bss24(p2):+.2f})")

    print("\n=== script_blend_v6.py 에 넣을 값 ===")
    print(f"SHIFT = {shift:.4f}")
    print(f"SLOPE = {slope:.4f}")
    print(f"_W_CELL = {W:.2f}   (base {n24b}시드, cell {n24c}시드)")
    raw = bss24(p)
    sl = bss24(p2) - raw
    print(f"\n로컬 원점수 {raw:.2f} | LB 환산 = {raw:.2f} + 133(구조 오프셋) "
          f"+ {sl:.2f}(기울기) + 16.3(refit 1.5) = {raw + 133 + sl + 16.3:.0f}")
    print("※ 오프셋은 SHIFT 이득이 아니다(RF 는 SHIFT 없이도 +134.1). "
          "구조적 이득이므로 원점수에 더한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

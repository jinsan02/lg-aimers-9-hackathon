"""v14 블렌드의 가중·기울기·SHIFT 를 한 번에 정한다 (노트북, GPU 0).

순서가 중요하다. 기울기를 걸면 평균이 움직이므로 **SHIFT 는 기울기 뒤에** 정해야 한다.

  ① 멤버 가중  v14f : v14c = 0.80 : 0.20
       (2024 자기적합 최적은 0.238, 2023R 최적은 0.543. 두 시즌 모두 양수이고
        최악(margin 0)에도 -1.8 에 그치는 0.20 을 쓴다. 최적화가 아니라 강건성.)
  ② 기울기     2023 홀드아웃에서 적합한 값을 그대로 쓴다 (자기적합 금지)
  ③ SHIFT      기울기 적용 후의 편향 x 0.65
       (0.65 = ABS 2년차 복귀 계수. 2024 는 ABS 1년차라 하락폭이 평시보다 컸다)

실행: python tools/decide_v14.py
"""

import glob
import sys

import numpy as np

W_CELL = 0.20          # MultiClass 멤버 가중
SLOPE_FIT_ON = "W_base_s*"     # 기울기는 2023 홀드아웃에서 적합
Y2_FACTOR = 0.65       # ABS 2년차 복귀 계수


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
    """y ~ sigmoid(a + b*logit(p)) 뉴턴법. 기울기 b 만 쓴다."""
    X = np.column_stack([np.ones(len(p)), _logit(p)])
    w = np.array([0.0, 1.0])
    for _ in range(30):
        q = 1 / (1 + np.exp(-X @ w))
        W = q * (1 - q)
        H = (X * W[:, None]).T @ X + 1e-9 * np.eye(2)
        w = w + np.linalg.solve(H, X.T @ (y - q))
    return float(w[1])


def main():
    pf, y, nf = load("v14f_s*")
    if pf is None:
        print("v14f 예측이 아직 없다 (빌드 진행 중)")
        return 1
    pc, _, nc = load("v14c_s*")
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    print(f"2024 홀드아웃 {len(y):,}행 | 실제 성공률 {r:.4f}")
    print(f"v14f {nf}시드 {bss(pf):.2f} | v14c {nc}시드 "
          f"{bss(pc):.2f}" if pc is not None else f"v14f {nf}시드 {bss(pf):.2f}")

    p = pf if pc is None else (1 - W_CELL) * pf + W_CELL * pc
    print(f"\n① 블렌드 (w_cell={W_CELL})        {bss(p):8.2f}")

    pa, ya, na = load(SLOPE_FIT_ON)
    if pa is None:
        print(f"  기울기 적합용 {SLOPE_FIT_ON} 없음 — 기울기 1.0 유지")
        slope = 1.0
    else:
        slope = fit_slope(pa, ya)
        print(f"② 기울기 (2023 {na}시드 적합)  {slope:.4f}")
    p2 = 1 / (1 + np.exp(-slope * _logit(p)))
    print(f"   기울기 적용 후                {bss(p2):8.2f}  ({bss(p2)-bss(p):+.2f})")

    b = float(p2.mean() - r)
    shift = Y2_FACTOR * b
    print(f"③ 기울기 후 편향 {b:+.5f} x {Y2_FACTOR} -> SHIFT {shift:.4f}")
    p3 = np.clip(p2 - shift, 0, 1)
    print(f"   SHIFT 적용 후                 {bss(p3):8.2f}  ({bss(p3)-bss(p2):+.2f})")
    print(f"\n=== script_blend_v6.py 에 넣을 값 ===")
    print(f"SHIFT = {shift:.4f}")
    print(f"SLOPE = {slope:.4f}")
    print(f"WEIGHTS: v14f x{nf} @ {(1-W_CELL)/max(nf,1):.4f}, "
          f"v14c x{nc} @ {W_CELL/max(nc,1):.4f}")
    # ⚠️ 오프셋 +133.46 은 **SHIFT 를 적용한 제출본**과 **SHIFT 없는 로컬 원점수**를
    #    맞춰서 얻은 값이다. 따라서 SHIFT 적용 후 점수에 오프셋을 또 더하면
    #    같은 이득을 두 번 세게 된다. 환산은 반드시 **원점수 기준**으로 한다.
    raw = bss(p)
    sl = bss(p2) - raw            # 기울기는 v13 에 없던 새 후처리라 따로 더한다
    print(f"\n로컬 원점수 {raw:.2f} (v13 927.75 대비 {raw - 927.75:+.2f})")
    print(f"LB 환산 = {raw:.2f} + 133.46(오프셋) + {sl:.2f}(기울기) "
          f"+ 8~13(refit 배수)")
    print(f"       = {raw + 133.46 + sl + 8:.0f} ~ {raw + 133.46 + sl + 13:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

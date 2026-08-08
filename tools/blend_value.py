"""새 모델을 블렌드에 넣을 가치가 있는지 **짓기 전에** 계산한다.

동기: NN(TabM 등)을 만들지 말지 결정해야 하는데, 만들고 나서 기여 0이면 하루를 버린다.

⚠️ 흔한 오류: 잔차 (p - y) 끼리 상관을 재면 공유되는 -y 항이 지배해서 어떤 두 모델이든
rho ~= 0.9999 가 나온다. y가 0/1 이라 잔차는 베르누이 잡음이 대부분이기 때문이다.
블렌드 이득을 결정하는 건 **예측값끼리의 불일치**다.

정확한 식 (p1=현재 블렌드, p2=후보):
    A = E[(p2 - p1)^2]                     # 불일치 (= 다양성의 실체)
    D = p1 의 BSS - p2 의 BSS,  delta = D * base / 1e5
    w* = (A - delta) / (2A),   이득(MSE) = (delta - A)^2 / (4A)
    -> **delta >= A 이면 최적 가중이 0 이다.** 즉 후보가 기여하려면
       성능 격차가 예측 불일치보다 작아야 한다.
    D_max = 1e5 * A / base   (이 격차까지는 기여함)

메모리: npz 하나가 253,507 x float32 = 1MB. 그룹 평균만 유지한다.

실행: python tools/blend_value.py
"""

import glob
import sys

import numpy as np

GROUPS = {
    "F11(주력)": [f"v11f_s{s}" for s in (42, 7, 13, 3, 4, 5, 6, 8)],
    "A11(lr.02)": [f"v11a_s{s}" for s in (42, 7, 13, 3, 4, 5, 6, 8)],
    "B11(depth7)": [f"v11b_s{s}" for s in (42, 7, 13)],
    "G12(도메인X)": [f"v12g_s{s}" for s in (42, 7, 13, 3, 4, 5)],
    "F11 단일시드": ["v11f_s42"],
}


def main():
    y, gp = None, {}
    for g, tags in GROUPS.items():
        ps = []
        for t in tags:
            h = glob.glob(f"./out/*_{t}_val_preds.npz")
            if not h:
                continue
            z = np.load(h[0])
            if y is None:
                y = z["y"].astype(np.float32)
            ps.append(z["pred"].astype(np.float32))
        if ps:
            gp[g] = np.mean(ps, 0)
    r = float(y.mean())
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - float(((p - y) ** 2).mean()) / base)

    ref = gp["F11(주력)"]
    b1 = bss(ref)
    print(f"검증 {len(y):,}행 | r={r:.4f} | 기준 F11 = {b1:.1f}\n")

    print("=== 보유 모델의 실측 불일치와 블렌드 가치 ===")
    print(f"{'모델':<16}{'BSS':>9}{'격차 D':>8}{'불일치 rms':>11}"
          f"{'기여상한 Dmax':>14}{'최적 w':>8}{'이득':>7}")
    for g, p in gp.items():
        if g == "F11(주력)":
            continue
        A = float(((p - ref) ** 2).mean())
        D = b1 - bss(p)
        delta = D * base / 1e5
        dmax = 1e5 * A / base
        w = (A - delta) / (2 * A) if A > 0 else 0.0
        w = min(max(w, 0.0), 1.0)
        gain = 1e5 * ((delta - A) ** 2 / (4 * A)) / base if (A > 0 and w > 0) else 0.0
        print(f"{g:<16}{bss(p):>9.1f}{D:>8.1f}{np.sqrt(A):>11.5f}"
              f"{dmax:>14.1f}{w:>8.3f}{gain:>7.1f}")

    print("\n=== 새 모델의 손익분기 (행=예측 불일치 rms, 열=성능 격차 D) ===")
    Ds = [0, 10, 20, 40, 60, 100, 150]
    print(f"{'rms':>8}{'Dmax':>8}" + "".join(f"{f'D={d}':>9}" for d in Ds))
    for rms in (0.005, 0.010, 0.015, 0.020, 0.030, 0.050):
        A = rms * rms
        dmax = 1e5 * A / base
        row = f"{rms:>8.3f}{dmax:>8.0f}"
        for D in Ds:
            delta = D * base / 1e5
            g = ((delta - A) ** 2 / (4 * A)) if delta < A else 0.0
            row += f"{1e5 * g / base:>9.1f}"
        print(row)
    print("\n읽는 법: 같은 CatBoost 설정끼리는 rms 불일치가 0.005 안팎이라 Dmax 가 10 정도다.")
    print("  -> 40점 뒤지는 모델은 rms 0.02 이상 달라야 기여한다.")
    print("  NN 이 GBDT 대비 rms 0.02~0.03 다르게 예측한다면 D=60 이어도 이득이 난다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

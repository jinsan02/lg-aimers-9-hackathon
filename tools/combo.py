"""남은 조합 후보를 **GPU 없이** 한 번에 검토한다.

이미 갖고 있는 예측(npz)만으로 답이 나오는 질문 세 가지:

  A. 다중 검증시즌 앙상블 — 조기종료를 2023 으로 한 모델과 2024 로 한 모델은
     '어느 시즌 라벨이 복잡도를 정했나'가 다르다. CatBoost 하이퍼파라미터 변주는
     rms 0.0025 로 소진됐지만 이건 다른 종류의 차이다. margin 이 양수인가?
     (S_rm* 의 test_preds 가 val2023 모델의 2024 예측이다 — 이미 있다)

  B. 캘리브레이션 기울기 — SHIFT 는 0차(절편)만 고친다. 1차(기울기)는 한 번도
     안 쟀다. **2023 에서 적합해 2024 에 적용**해서 시즌을 넘는지 본다.
     (세그먼트 가중이 이 검사에서 죽었다: -8.9 ~ -17.6)

  C. 세그먼트 SHIFT — 전역 SHIFT 후 남은 잔차가 축별로 균일한가.
     LEVERS 는 "균일해서 전역으로 충분"이라 적었지만 절제 근거가 없다.

실행: python tools/combo.py
"""

import glob
import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"


def load(pat, kind="val"):
    fs = sorted(glob.glob(f"./out/*{pat}_{kind}_preds.npz"))
    if not fs:
        return None, None
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def main():
    p1, y = load("Q_base_s*")
    if p1 is None:
        print("기준(Q_base) 예측 없음")
        return 1
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - y) ** 2).mean() / base)

    def margin(pa, pb):
        A = float(((pa - pb) ** 2).mean())
        if A <= 0:
            return 0.0, 0.0, 0.0
        m = 1e5 * 2 * float(((pa - y) * (pa - pb)).mean()) / base
        d = 1e5 * A / base
        return m, d, (m * m / (4 * d) if m > 0 else 0.0)

    m0 = bss(p1)
    print(f"기준 2024 BSS {m0:.2f} (r={r:.4f})\n")

    print("=== A. 다중 검증시즌 앙상블 (val2023 모델의 2024 예측) ===")
    found = False
    for tag in ("S_rm1.0_s*", "S_rm1.4_s*", "N_rm1.0_s*"):
        p2, y2 = load(tag, "test")
        if p2 is None or len(p2) != len(y):
            continue
        if not np.allclose(y2, y):
            print(f"  {tag}: 행 정렬 불일치 — 건너뜀")
            continue
        found = True
        m, d, g = margin(p1, p2)
        print(f"  {tag:<14} 단독 {bss(p2):>7.1f} | D {m0 - bss(p2):>+7.1f} | "
              f"rms {np.sqrt(d * base / 1e5):.4f} | margin {m:>+7.1f} | 이득 {g:>5.2f}")
    if not found:
        print("  해당 예측 없음")

    print("\n=== B. 캘리브레이션 기울기 (2023 적합 -> 2024 적용) ===")
    pa, ya = load("W_base_s*")
    if pa is None:
        print("  2023 예측(W_base) 없음")
    else:
        # 2023 에서 y ~ sigmoid(a + b*logit(p)) 적합 (뉴턴 2스텝이면 충분)
        z = np.log(np.clip(pa, 1e-6, 1 - 1e-6) / (1 - np.clip(pa, 1e-6, 1 - 1e-6)))
        X = np.column_stack([np.ones(len(z)), z])
        w = np.array([0.0, 1.0])
        for _ in range(25):
            q = 1 / (1 + np.exp(-X @ w))
            W = q * (1 - q)
            g = X.T @ (ya - q)
            H = (X * W[:, None]).T @ X + 1e-9 * np.eye(2)
            w = w + np.linalg.solve(H, g)
        print(f"  2023 적합 절편 {w[0]:+.4f} | 기울기 {w[1]:.4f} "
              f"(1.0 이면 보정 불필요)")
        zb = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / (1 - np.clip(p1, 1e-6, 1 - 1e-6)))
        for nm, ww in (("절편만", [w[0], 1.0]), ("기울기만", [0.0, w[1]]),
                       ("둘 다", w)):
            q = 1 / (1 + np.exp(-(ww[0] + ww[1] * zb)))
            print(f"    {nm:<8} 2024 적용 {bss(q) - m0:>+7.2f}")
        # 참고: 2024 자기적합 (낙관 상한)
        zz = np.column_stack([np.ones(len(zb)), zb])
        v = np.array([0.0, 1.0])
        for _ in range(25):
            q = 1 / (1 + np.exp(-zz @ v))
            W = q * (1 - q)
            v = v + np.linalg.solve((zz * W[:, None]).T @ zz + 1e-9 * np.eye(2),
                                    zz.T @ (y - q))
        q = 1 / (1 + np.exp(-zz @ v))
        print(f"  (참고) 2024 자기적합 기울기 {v[1]:.4f} -> {bss(q) - m0:+.2f} "
              f" <- 낙관 상한")

    print("\n=== C. 전역 SHIFT 후 남은 세그먼트 편향 ===")
    use = ["season", "game_type", "balls_before", "strikes_before",
           "asof_pitcher_n", TARGET]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    df = df[df.season == 2024].reset_index(drop=True)
    if len(df) != len(y):
        print("  행 수 불일치 — 생략")
        return 0
    ps = p1 - (p1.mean() - r)          # 전역 편향 제거 후
    df["cnt"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
    df["expq"] = pd.qcut(df.asof_pitcher_n.fillna(0), 5, labels=False,
                         duplicates="drop")
    for col in ("cnt", "game_type", "expq"):
        rows = []
        for v, idx in df.groupby(col, observed=True).indices.items():
            if len(idx) < 2000:
                continue
            rows.append((str(v), len(idx), float(ps[idx].mean() - y[idx].mean())))
        rows.sort(key=lambda x: -abs(x[2]))
        span = max(x[2] for x in rows) - min(x[2] for x in rows)
        # 세그먼트 시프트 오라클 (그 시즌 자기적합 = 상한)
        adj = ps.copy()
        for v, _, b in rows:
            m = (df[col].astype(str) == v).to_numpy()
            adj[m] -= b
        print(f"  {col:<10} 편차 폭 {span:.4f} | 세그먼트 시프트 오라클 "
              f"{bss(adj) - bss(ps):+.2f}")
        for v, n, b in rows[:3]:
            print(f"      {v:<6} n={n:>7,} 편향 {b:+.4f}")
    print("\n※ 오라클은 그 시즌 자기적합이라 상한이다. 시즌을 넘는지는 별도 확인이 필요하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

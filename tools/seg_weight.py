"""세그먼트별 블렌드 가중이 **시즌을 넘어가는가**.

전역 margin 이 양수여도 세그먼트에서 더 클 수 있다 (Jensen):
    이득_seg = sum_s pi_s * margin_s^2 / (4 Dmax_s)  >=  전역 이득

가중이 그 행 자신의 피처 함수이면 행 독립 원칙을 지키므로 합법이다.

⚠️ 다만 같은 시즌에서 가중을 적합하고 같은 시즌에서 보고하면 그 홀드아웃이
   오염된다. E123 이 정확히 이걸 안 해서 -4.07 을 태웠다 (교차적합 잔차가
   +31 이라 했는데 시즌 간 이전 계수를 안 쟀다).

그래서 **2023 에서 적합 -> 2024 에서 확인**한다. 살아남지 못하면 전역 가중만 쓴다.
세그먼트 가중은 표본이 작으면 잡음이라 전역 가중 쪽으로 수축한다.

실행: python tools/seg_weight.py
"""

import glob
import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
KS = (0.0, 20.0, 100.0, 500.0)      # 전역 가중으로의 수축 강도(유효표본 단위)


def load(pat):
    fs = sorted(glob.glob(f"./out/*{pat}_val_preds.npz"))
    if not fs:
        raise SystemExit(f"npz 없음: {pat}")
    z = [np.load(f) for f in fs]
    return (np.mean([q["pred"] for q in z], 0).astype(np.float64),
            z[0]["y"].astype(np.float64))


def segments(season, league, n):
    use = ["season", "game_type", "balls_before", "strikes_before", "inning",
           "asof_pitcher_n", TARGET]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    if league:
        df = df[df.game_type == league]
    df = df[df.season == season].reset_index(drop=True)
    assert len(df) == n, f"{season} 행 수 불일치 {len(df)} vs {n}"
    return {
        "cnt": (df.balls_before.astype(str) + "-"
                + df.strikes_before.astype(str)).to_numpy(),
        "inn": df.inning.clip(upper=8).astype(str).to_numpy(),
        "expq": pd.qcut(df.asof_pitcher_n.fillna(0), 5, labels=False,
                        duplicates="drop").astype(str).to_numpy(),
    }


def fit_weights(p1, p2, y, seg, base, w_glob, k):
    """세그먼트별 최적 가중. 유효표본 n_s 로 전역 가중 쪽으로 수축한다."""
    out = {}
    for v in np.unique(seg):
        m = seg == v
        A = float(((p1[m] - p2[m]) ** 2).mean())
        if A <= 0 or m.sum() < 200:
            out[v] = w_glob
            continue
        mg = 2 * float(((p1[m] - y[m]) * (p1[m] - p2[m])).mean())
        w = mg / (2 * A)
        n = m.sum()
        out[v] = (n * w + k * w_glob) / (n + k)      # 수축
    return out


def main():
    # 2023 (R리그) 에서 적합
    p1a, ya = load("W_base_s*")
    p2a, _ = load("W_cell_s*")
    sa = segments(2023, "R", len(ya))
    # 2024 (전체) 에서 확인
    p1b, yb = load("Q_base_s*")
    p2b, _ = load("Z_cell_s*")
    sb = segments(2024, "", len(yb))

    rb = yb.mean()
    base_b = rb * (1 - rb)

    def bss(p):
        return 1e5 * (1 - ((np.clip(p, 0, 1) - yb) ** 2).mean() / base_b)

    m0 = bss(p1b)
    ra = ya.mean()
    base_a = ra * (1 - ra)
    # 전역 가중 — 2023 에서 적합
    Aa = float(((p1a - p2a) ** 2).mean())
    wg = (2 * float(((p1a - ya) * (p1a - p2a)).mean())) / (2 * Aa)
    wg = float(np.clip(wg, 0.0, 1.0))
    print(f"2023R {len(ya):,}행 | 2024 {len(yb):,}행")
    print(f"2024 기준 단독 BSS {m0:.2f}\n")
    print(f"전역 가중 (2023 적합) w = {wg:.3f}")
    print(f"  -> 2024 적용 이득 {bss((1 - wg) * p1b + wg * p2b) - m0:+.2f}")

    # 참고: 2024 에서 직접 적합한 전역 가중 (낙관 편향 — 상한 표시용)
    Ab = float(((p1b - p2b) ** 2).mean())
    wb = float(np.clip((2 * float(((p1b - yb) * (p1b - p2b)).mean())) / (2 * Ab),
                       0, 1))
    print(f"  (참고) 2024 자기적합 w = {wb:.3f} -> "
          f"{bss((1 - wb) * p1b + wb * p2b) - m0:+.2f}  ← 낙관 상한\n")

    print("=== 세그먼트별 가중을 2023 에서 적합해 2024 에 적용 ===")
    print(f"{'축':<8}" + "".join(f"{f'k={k:.0f}':>10}" for k in KS))
    for name in ("cnt", "inn", "expq"):
        row = f"{name:<8}"
        for k in KS:
            w = fit_weights(p1a, p2a, ya, sa[name], base_a, wg, k)
            wv = np.array([np.clip(w.get(v, wg), 0.0, 1.0) for v in sb[name]])
            row += f"{bss((1 - wv) * p1b + wv * p2b) - m0:>10.2f}"
        print(row)
    print("\n※ 전역 가중 이득보다 크지 않으면 세그먼트 분할은 이전되지 않는 것이다.")
    print("※ k 는 전역 가중으로의 수축 강도 (k=0 은 수축 없음 = 잡음 최대).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""블렌드 기여를 **margin** 으로 잰다 — 전역 + 세그먼트별.

지금까지 후보를 "단독 BSS 가 낮다"로 기각해 왔는데 그건 D(수준 격차)만 본 것이다.
실제 기여 조건은 다르다.

    A      = E[(p1 - p2)^2]                     불일치
    A - d  = 2 E[(p1 - y)(p1 - p2)]             margin (= 후보가 잔차를 실제로 깎는가)
    Dmax   = 1e5 A / base                        BSS 단위 불일치
    이득   = margin_bss^2 / (4 Dmax),  w* = margin_bss / (2 Dmax)

여기서 두 가지가 따라온다.
  ① 이득 상한은 Dmax/4 다. +20 을 벌려면 rms >= 0.020 **이면서** D <= 50 이어야 한다.
     "약하지만 다양한 모델"은 수학적으로 배제된다.
  ② A = A_corr + A_noise 인데 margin 은 A_corr 만 집는다. 잡음으로 A 를 키우면
     분모만 커져 이득이 **줄어든다**. (random_strength / 배깅온도 금지)

전역 margin 이 0 이어도 세그먼트에서 부호가 갈릴 수 있다 (Jensen). 가중치가 그 행
자신의 피처 함수이면 행 독립 원칙을 지키므로 합법이다.

실행: python tools/margin.py <기준패턴> <후보패턴>
      python tools/margin.py "v13f_s*" "lin_*"
"""

import glob
import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
SEG_COLS = ["balls_before", "strikes_before", "game_type", "asof_pitcher_n",
            "season", "inning"]


def mean_pred(pattern):
    fs = sorted(glob.glob(f"./out/*{pattern}_val_preds.npz"))
    if not fs:
        raise SystemExit(f"패턴에 해당하는 npz 없음: {pattern}")
    z = [np.load(f) for f in fs]
    return np.mean([q["pred"] for q in z], 0).astype(np.float64), \
        z[0]["y"].astype(np.float64), len(fs)


def report(p1, p2, y, base, label, indent=""):
    """기준 p1 에 후보 p2 를 섞을 때의 margin/이득."""
    A = float(((p1 - p2) ** 2).mean())
    if A <= 0:
        return None
    margin = 1e5 * (2 * float(((p1 - y) * (p1 - p2)).mean())) / base
    dmax = 1e5 * A / base
    gain = margin ** 2 / (4 * dmax) if margin > 0 else 0.0
    w = max(0.0, min(1.0, margin / (2 * dmax)))
    print(f"{indent}{label:<22} rms {np.sqrt(A):.4f} | Dmax {dmax:>7.1f} | "
          f"margin {margin:>+8.1f} | w* {w:.3f} | 이득 {gain:>6.2f}")
    return gain


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    base_pat, cand_pat = sys.argv[1], sys.argv[2]
    p1, y, n1 = mean_pred(base_pat)
    p2, y2, n2 = mean_pred(cand_pat)
    assert len(p1) == len(p2), f"행 수 불일치 {len(p1)} vs {len(p2)}"
    r = y.mean()
    base = r * (1 - r)

    def bss(p):
        return 1e5 * (1 - ((p - y) ** 2).mean() / base)

    print(f"기준 {base_pat} (n={n1}) BSS {bss(p1):.2f} | "
          f"후보 {cand_pat} (n={n2}) BSS {bss(p2):.2f} | D {bss(p1)-bss(p2):+.1f}\n")
    print("=== 전역 ===")
    g = report(p1, p2, y, base, "후보 전체")

    print("\n=== 세그먼트별 (그 행 자신의 피처로 나눈다 = 행 독립 유지) ===")
    use = [c for c in SEG_COLS if c != "season"] + ["season", TARGET]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=list(dict.fromkeys(use)))
    df = df[df.season == 2024].reset_index(drop=True)
    if len(df) != len(y):
        print(f"  (2024 행 {len(df):,} != 예측 {len(y):,} — 세그먼트 생략)")
        return 0
    df["cnt"] = df.balls_before.astype(str) + "-" + df.strikes_before.astype(str)
    df["exp_q"] = pd.qcut(df.asof_pitcher_n.fillna(0), 5,
                          labels=False, duplicates="drop")
    df["inn"] = df.inning.clip(upper=8)
    df["p_dec"] = pd.qcut(pd.Series(p1), 10, labels=False, duplicates="drop")

    total = 0.0
    for col in ["cnt", "game_type", "exp_q", "inn", "p_dec"]:
        seg_gain = 0.0
        parts = []
        for v, idx in df.groupby(col, observed=True).indices.items():
            if len(idx) < 2000:
                continue
            q1, q2, yy = p1[idx], p2[idx], y[idx]
            A = float(((q1 - q2) ** 2).mean())
            if A <= 0:
                continue
            m = 1e5 * (2 * float(((q1 - yy) * (q1 - q2)).mean())) / base
            dm = 1e5 * A / base
            gg = m ** 2 / (4 * dm) if m > 0 else 0.0
            seg_gain += (len(idx) / len(y)) * gg
            parts.append((str(v), len(idx), m, gg))
        print(f"  {col:<10} 세그먼트 가중 이득 {seg_gain:>6.2f}  "
              f"(전역 {g:.2f} 대비 {seg_gain - (g or 0):+.2f})")
        for v, n, m, gg in sorted(parts, key=lambda x: -x[3])[:3]:
            if gg > 0.05:
                print(f"      {v:<8} n={n:>7,} margin {m:>+8.1f} 이득 {gg:>6.2f}")
        total = max(total, seg_gain)
    print(f"\n최선 세그먼트 분할 이득 {total:.2f} vs 전역 {g or 0:.2f}")
    print("※ 세그먼트 가중은 2023 에서 적합하고 2024 에서 확인해야 정직하다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

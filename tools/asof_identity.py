"""asof 컬럼 사이의 **대수적 항등식 전수 감사** — E99급 구조 발견을 노린다.

배경: E99(+72.6, 대회 최대 발견)는 "asof 가 시즌마다 리셋되는가?"를 **실제로 확인**해서
나왔다. 문서에 안 적힌 성질이 데이터에 있었다. 같은 방식으로 19개 asof 컬럼 사이의
관계를 전수 조사한다.

읽는 법:
  정확히 성립하는 항등식  -> 그 컬럼은 중복이다 (정보 없음)
  거의 성립하는데 잔차 有 -> **그 잔차가 숨은 정보**다 (E99 가 이 형태였다)
  전혀 성립 안 함        -> 독립 정보

메모리: 필요한 컬럼만, 2024 시즌만 읽는다.
실행: python tools/asof_identity.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"


def rep(name, v, extra=""):
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print(f"  {name:<44} (전부 결측)")
        return
    z = float((np.abs(v) < 1e-9).mean())
    print(f"  {name:<44} 평균 {v.mean():+.6f}  |값| 중앙 "
          f"{np.median(np.abs(v)):.6f}  정확0 {z:6.1%}  {extra}")


def main():
    cols = [c for c in pd.read_csv(f"{DATA}/test.csv", nrows=0).columns
            if c.startswith("asof_")]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=cols + ["season", "pitcher_id", "batter_id",
                                     "control_success"])
    d = df[df.season == 2024]
    print(f"2024 {len(d):,}행 | asof 컬럼 {len(cols)}개\n")

    g = lambda c: d[c].to_numpy(np.float64)      # noqa: E731

    print("=== ① 실패 모드 합이 1인가 (success + middle + ball + reverse) ===")
    s = g("asof_pitcher_success_rate")
    mi = g("asof_pitcher_middle_rate")
    ba = g("asof_pitcher_ball_rate")
    rv = g("asof_pitcher_reverse_rate")
    rep("success+middle+ball+reverse - 1", s + mi + ba + rv - 1.0)
    rep("success+middle+ball - 1", s + mi + ba - 1.0)
    rep("success+middle+reverse - 1", s + mi + rv - 1.0)

    print("\n=== ② strike_rate 는 무엇인가 ===")
    st = g("asof_pitcher_strike_rate")
    rep("strike - (success+middle)", st - (s + mi))
    rep("strike - (1-ball)", st - (1.0 - ba))
    rep("strike + ball - 1", st + ba - 1.0)
    for nm, v in [("success", s), ("middle", mi), ("ball", ba),
                  ("reverse", rv)]:
        m = np.isfinite(st) & np.isfinite(v)
        print(f"  strike vs {nm:<10} 상관 {np.corrcoef(st[m], v[m])[0, 1]:+.4f}")

    print("\n=== ③ 구종 배합 합이 1인가 ===")
    fa = g("asof_pitcher_fastball_rate")
    br = g("asof_pitcher_breaking_rate")
    of = g("asof_pitcher_offspeed_rate")
    rep("fastball+breaking+offspeed - 1", fa + br + of - 1.0)

    print("\n=== ④ 표본수 컬럼 관계 ===")
    pn = g("asof_pitcher_n")
    px = g("asof_pitcher_pitchmix_n")
    bn = g("asof_batter_n")
    rep("pitcher_n - pitchmix_n", pn - px, "(0이면 완전 중복)")
    d2 = pn - px
    print(f"    차이 분포: 0인 비율 {float((d2 == 0).mean()):.1%} | "
          f"최대 {np.nanmax(d2):.0f} | 음수 비율 {float((d2 < 0).mean()):.1%}")

    print("\n=== ⑤ 누적 성공수가 정수인가 (E99 의 핵심 가정) ===")
    S = pn * s
    frac = np.abs(S - np.round(S))
    print(f"  n x success_rate 의 소수부: 중앙 {np.median(frac):.6f} | "
          f"1e-6 미만 비율 {float((frac < 1e-6).mean()):.1%}")
    print("  -> 정수에 가까우면 n 과 rate 로 **누적 성공수를 정확히 복원**할 수 있다")
    for nm, c in [("middle", mi), ("ball", ba), ("reverse", rv),
                  ("strike", st)]:
        f2 = np.abs(pn * c - np.round(pn * c))
        print(f"     {nm:<8} 소수부 중앙 {np.median(f2):.6f} | "
              f"정수비율 {float((f2 < 1e-6).mean()):.1%}")
    Sb = bn * g("asof_batter_success_rate")
    fb = np.abs(Sb - np.round(Sb))
    print(f"     batter   소수부 중앙 {np.median(fb):.6f} | "
          f"정수비율 {float((fb < 1e-6).mean()):.1%}")

    print("\n=== ⑥ prev1/3/5 가 서로에게서 유도되는가 ===")
    p1 = g("asof_pitcher_prev1_game_success_rate")
    p3 = g("asof_pitcher_prev3_game_success_rate")
    p5 = g("asof_pitcher_prev5_game_success_rate")
    m = np.isfinite(p1) & np.isfinite(p3) & np.isfinite(p5)
    print(f"  상관 p1-p3 {np.corrcoef(p1[m], p3[m])[0, 1]:+.4f} | "
          f"p3-p5 {np.corrcoef(p3[m], p5[m])[0, 1]:+.4f} | "
          f"p1-p5 {np.corrcoef(p1[m], p5[m])[0, 1]:+.4f}")
    # p3 이 p1 과 (2~3경기)의 가중평균이면 3*p3 - p1 = 2*(2~3경기) 가 나온다
    mid = (3 * p3 - p1) / 2.0          # 2,3 번째 경기 추정
    old = (5 * p5 - 3 * p3) / 2.0      # 4,5 번째 경기 추정
    ok = np.isfinite(mid) & np.isfinite(old)
    print(f"  역산한 '2~3경기' 범위 밖 비율 "
          f"{float(((mid < -0.01) | (mid > 1.01))[ok].mean()):.1%}")
    print(f"  역산한 '4~5경기' 범위 밖 비율 "
          f"{float(((old < -0.01) | (old > 1.01))[ok].mean()):.1%}")
    print("  -> 범위 안이면 **경기 단위 궤적을 3점으로 분해**할 수 있다 (신규 정보)")
    y = d["control_success"].to_numpy(np.float64)
    for nm, v in [("p1", p1), ("p3", p3), ("p5", p5),
                  ("역산 2~3경기", mid), ("역산 4~5경기", old)]:
        k = np.isfinite(v)
        print(f"     {nm:<12} 타깃 상관 {np.corrcoef(v[k], y[k])[0, 1]:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""카운트가 '난이도'가 아니라 '의도'를 바꾸는가? (E111 근거 조사)

야구 도메인 가설:
  대회의 제구 실패 정의 3가지는 서로 다른 **의도의 실패**다.
    ① 존 한가운데   - 볼넷을 피하려고 무조건 넣을 때 나는 실패
    ② 존 밖         - 유인구를 던질 때 나는 실패 (전략상 옳아도 라벨은 실패)
    ③ 반대 방향     - 정상 승부에서 나는 순수 커맨드 실패
  그렇다면 카운트마다 **활성화되는 실패 모드가 다르고**, 예측에 필요한 건
  '이 투수가 대체로 잘하나'가 아니라 **'이 카운트가 여는 모드에서 이 투수가 약한가'** 다.

검정 방법:
  카운트별로 나눠서, 성공 여부를 그 투수의 asof 모드별 비율에 회귀한다.
  가설이 맞으면 **계수 패턴이 카운트마다 달라야 한다**.
    - 3볼 구간에서는 middle_rate 계수가 특히 나쁘고
    - 2스트라이크 구간에서는 ball_rate 계수가 특히 나쁘다
  계수 패턴이 카운트와 무관하면 가설 기각 -> 'success_rate 하나면 충분'.

실행: python tools/count_intent.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
MODES = ["asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
         "asof_pitcher_reverse_rate"]
SHORT = {"asof_pitcher_middle_rate": "middle(1)",
         "asof_pitcher_ball_rate": "ball(2)",
         "asof_pitcher_reverse_rate": "reverse(3)"}


def fit(d):
    """성공 ~ 1 + 표준화된 모드비율. 표준화계수라 크기 비교가 가능하다."""
    X = d[MODES].to_numpy(np.float64)
    y = d["control_success"].to_numpy(np.float64)
    m = np.isfinite(X).all(1)
    X, y = X[m], y[m]
    if len(y) < 3000:
        return None
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, len(y), y.mean()


def main():
    use = ["season", "game_type", "balls_before", "strikes_before",
           "control_success"] + MODES
    df = pd.read_csv(f"{DATA}/train.csv", usecols=use)
    df = df[(df.game_type == "R") & (df.season >= 2022)]
    print(f"R리그 2022~2024 {len(df):,}행 | 전체 성공률 {df.control_success.mean():.4f}\n")

    # 1) 카운트별 성공률 - 의도 차이가 있으면 성공률 자체가 크게 흔들린다
    piv = df.pivot_table(index="balls_before", columns="strikes_before",
                         values="control_success", aggfunc="mean")
    cnt = df.pivot_table(index="balls_before", columns="strikes_before",
                         values="control_success", aggfunc="size")
    print("=== 카운트별 성공률 (행=볼, 열=스트라이크) ===")
    print(piv.round(4).to_string())
    print("\n=== 표본수 ===")
    print(cnt.to_string())

    # 2) 카운트 구간별 모드 계수 - 가설의 핵심
    groups = {
        "3볼 (무조건 스트라이크)": df.balls_before == 3,
        "2스트라이크·3볼아님 (유인구)": (df.strikes_before == 2) & (df.balls_before < 3),
        "짝수 0-0/1-1/2-2 (정상승부)": df.balls_before == df.strikes_before,
        "그 외": ~((df.balls_before == 3)
                 | ((df.strikes_before == 2) & (df.balls_before < 3))
                 | (df.balls_before == df.strikes_before)),
    }
    print(f"\n=== 카운트 구간별 표준화 회귀계수 (성공 ~ 모드비율) ===")
    print(f"{'구간':<30}{'n':>10}{'성공률':>9}"
          + "".join(f"{SHORT[m]:>12}" for m in MODES))
    rows = {}
    for name, mask in groups.items():
        r = fit(df[mask])
        if r is None:
            continue
        beta, n, mu = r
        rows[name] = beta[1:]
        print(f"{name:<30}{n:>10,}{mu:>9.4f}"
              + "".join(f"{b:>+12.4f}" for b in beta[1:]))

    # 3) 구간 간 계수 격차 - 이게 크면 '카운트x스타일' 상호작용이 실재한다
    if len(rows) >= 3:
        M = np.array(list(rows.values()))
        print(f"\n{'모드':<14}{'최소':>10}{'최대':>10}{'격차':>10}"
              f"{'  <- 격차가 크면 카운트마다 다른 모드가 위험하다'}")
        for j, m in enumerate(MODES):
            lo, hi = M[:, j].min(), M[:, j].max()
            print(f"{SHORT[m]:<14}{lo:>+10.4f}{hi:>+10.4f}{hi - lo:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

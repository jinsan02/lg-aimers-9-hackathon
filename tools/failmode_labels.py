"""실패모드 라벨을 복원할 수 있는가 — E99 를 **라벨**에 적용한다.

E99 는 asof 가 통산 누적임을 알고 **시즌 차분**으로 성적을 복원했다.
그런데 갱신 단위는 투구마다 +1 이다(감사 결과 100.0%). 그렇다면 **한 투구 차분**은
그 투구 하나의 결과다.

    S_x(t) = asof_pitcher_n(t) x asof_pitcher_x_rate(t)      (x 의 누적 횟수)
    S_x(t+1) - S_x(t) = 1{그 투구 t 가 x 였는가}

즉 train 의 모든 행에 대해 **middle / ball / strike / reverse 라벨**을 복원할 수 있다.
train 에는 control_success 밖에 없지만, 사실은 실패모드별 라벨이 숨어 있는 것이다.

왜 중요한가:
  제구 실패의 정의는 ① 존 중앙 ② 존 밖 ③ 포수 요구 반대 — **서로 다른 능력**이다.
  ①은 실투, ②는 커맨드, ③은 배터리 소통. 하나의 이진 타깃으로 뭉쳐 학습하는 대신
  셋을 따로 학습해 합치면(또는 보조 과제로 얹으면) 새 감독 신호가 들어온다.
  honest_ceiling 이 "남은 이득은 단일축 그룹 구조가 아니라 **다른 종류의 정보**에서만
  나온다"고 했는데, 새 라벨이 바로 그 다른 종류다.

⚠️ 합법성: 라벨 복원은 **train 행끼리만** 한다. 평가 시점에는 그 행 자신의 컬럼만
   보고 예측하므로 행 독립 원칙을 지킨다. (같은 대수를 test 행에 쓰면 타깃 자체가
   복원되는데, 그것은 '평가 데이터 내부 다른 행 참조'라 명시적 금지 사항이다.)

실행: python tools/failmode_labels.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
RATES = ["success", "middle", "ball", "strike", "reverse"]


def main():
    use = (["row_id", "season", "pitcher_id", "asof_pitcher_n", "control_success"]
           + [f"asof_pitcher_{r}_rate" for r in RATES])
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    df = df.sort_values("row_id", kind="stable").reset_index(drop=True)
    print(f"train {len(df):,}행\n")

    n = df["asof_pitcher_n"].to_numpy(np.float64)
    g = df.groupby("pitcher_id", sort=False)
    step = g["asof_pitcher_n"].diff().to_numpy()
    ok = step == 1                      # 같은 투수의 연속 투구인 행만 복원 가능
    print(f"=== ① 연속(+1) 행 비율 {np.nanmean(ok == 1):.4f} "
          f"({int(np.nansum(ok)):,}행에서 라벨 복원 가능) ===\n")

    lab = {}
    print("=== ② 차분이 0/1 로 떨어지는가 (떨어지면 라벨 복원 성공) ===")
    for r in RATES:
        c = f"asof_pitcher_{r}_rate"
        S = n * df[c].to_numpy(np.float64)
        d = pd.Series(S).groupby(df["pitcher_id"].to_numpy()).diff().to_numpy()
        d = np.where(ok, d, np.nan)
        near = np.abs(d - np.round(d)) < 1e-6
        v = np.round(d)
        frac01 = np.nanmean((v == 0) | (v == 1))
        print(f"  {r:<8} 정수 {np.nanmean(near):.4f} | 0/1 {frac01:.4f} | "
              f"평균 {np.nanmean(v):.4f} | 최소 {np.nanmin(v):.0f} 최대 {np.nanmax(v):.0f}")
        lab[r] = v

    print("\n=== ③ 복원한 success 라벨이 control_success 와 일치하는가 ===")
    y = df["control_success"].to_numpy(np.float64)
    # 차분은 '그 행의 투구' 결과이므로 **한 칸 당겨야** 그 행의 타깃과 맞는다
    for shift, name in ((0, "그 행"), (-1, "한 칸 앞")):
        rec = pd.Series(lab["success"]).shift(shift).to_numpy()
        m = np.isfinite(rec)
        print(f"  {name:<6} 일치율 {np.mean(rec[m] == y[m]):.6f} ({int(m.sum()):,}행)")

    print("\n=== ④ 실패는 세 모드의 합집합인가 (타깃 정의 역공학) ===")
    L = {r: pd.Series(lab[r]).shift(-1).to_numpy() for r in RATES}
    m = np.isfinite(L["success"]) & np.isfinite(L["middle"])
    for name, fail in (
            ("middle|ball", (L["middle"] > 0) | (L["ball"] > 0)),
            ("middle|reverse", (L["middle"] > 0) | (L["reverse"] > 0)),
            ("middle|ball|reverse",
             (L["middle"] > 0) | (L["ball"] > 0) | (L["reverse"] > 0)),
            ("~strike|middle", (L["strike"] == 0) | (L["middle"] > 0))):
        print(f"  실패 = {name:<22} 일치율 {np.mean((1 - fail[m]) == y[m]):.6f}")
    print("\n  각 모드 기저율 (그 행의 투구 기준)")
    for r in RATES:
        print(f"    {r:<8} {np.nanmean(L[r][m]):.4f}")
    print(f"    strike+ball {np.nanmean(L['strike'][m] + L['ball'][m]):.4f} "
          f"(1.0 이면 이분할)")

    print("\n=== ⑤ 복원 가능 행의 시즌 분포 (학습에 쓸 수 있는 양) ===")
    cover = pd.Series(np.isfinite(L["middle"]) & m).groupby(df["season"]).mean()
    print((cover * 100).round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

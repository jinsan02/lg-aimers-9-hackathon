"""감독의 투수 운용 스타일 분석 — 경기 단위 복원 후 도메인 해석.

train.csv에 game_id는 없지만:
  · row_id가 시간순 정렬
  · 한 경기 동안 (투수팀, 타자팀) 쌍이 고정
→ 팀 쌍이 바뀌는 지점 = 경기 경계. 이걸로 경기를 복원한다.

분석 목표:
  ① 경기당 등판 투수 수 = 감독의 계투 성향
  ② 선발 이닝 길이 = 선발 신뢰도
  ③ 팀별·시즌별 운용 변화 (감독 교체·트렌드)
  ④ 운용 스타일이 제구 성공률과 연관되는가
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
DATA, T = "./data", "control_success"

df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_month", "game_dayofweek",
                          "inning", "top_bottom", "pitcher_id",
                          "pitcher_team_id", "batter_team_id", "outs_before", T])
df["rid"] = df.row_id.str.extract(r"(\d+)").astype(np.int64)
df = df.sort_values("rid").reset_index(drop=True)

# 경기 경계: 팀 쌍(정렬된)이 바뀌는 지점
pair = np.sort(df[["pitcher_team_id", "batter_team_id"]].to_numpy(), axis=1)
key = pair[:, 0] * 1000 + pair[:, 1]
newgame = np.r_[True, (key[1:] != key[:-1])
                | (df.game_month.to_numpy()[1:] != df.game_month.to_numpy()[:-1])]
df["game_id"] = np.cumsum(newgame)
print(f"복원된 경기 수: {df.game_id.nunique():,} "
      f"(시즌당 {df.game_id.nunique() / 6:,.0f})")
gsz = df.groupby("game_id").size()
print(f"경기당 투구 수: 중앙값 {gsz.median():.0f} | "
      f"5~95% {gsz.quantile(.05):.0f}~{gsz.quantile(.95):.0f}")

# 팀 단위 등판 투수 수 (한 경기에서 각 팀이 쓴 투수)
per = df.groupby(["game_id", "pitcher_team_id"]).agg(
    n_pitchers=("pitcher_id", "nunique"),
    n_pitches=("pitcher_id", "size"),
    rate=(T, "mean"),
    season=("season", "first")).reset_index()
per = per[per.n_pitches >= 40]         # 정상 경기만

print("\n=== ① 경기당 등판 투수 수 (팀 기준) ===")
print(per.n_pitchers.describe().round(2).to_string())
print("\n시즌별 평균:")
print(per.groupby("season")["n_pitchers"].agg(["mean", "median"]).round(2).to_string())

print("\n=== ② 등판 투수 수 × 제구 성공률 ===")
print(per.groupby("n_pitchers").agg(
    경기수=("rate", "size"), 성공률=("rate", "mean"),
    투구수=("n_pitches", "mean")).round(4).to_string())

print("\n=== ③ 팀별 운용 성향 (평균 등판 투수 수) ===")
tm = per.groupby("pitcher_team_id").agg(
    경기=("n_pitchers", "size"), 평균투수=("n_pitchers", "mean"),
    성공률=("rate", "mean")).sort_values("평균투수")
print(tm[tm.경기 > 300].round(3).to_string())

print("\n=== ④ 팀×시즌 운용 변화 (감독 교체 탐지) ===")
piv = per.pivot_table(index="pitcher_team_id", columns="season",
                      values="n_pitchers", aggfunc="mean")
piv = piv[piv.notna().sum(axis=1) >= 5]
print(piv.round(2).to_string())
d = piv.diff(axis=1).abs().max(axis=1)
print(f"\n시즌 간 최대 변화폭 상위(운용 급변 = 감독 교체 가능성):")
print(d.sort_values(ascending=False).head(5).round(2).to_string())

print("\n=== ⑤ 선발 투구 이닝 (첫 투수의 마지막 이닝) ===")
first_p = df.groupby(["game_id", "pitcher_team_id"])["pitcher_id"].transform("first")
starters = df[df.pitcher_id == first_p]
sl = starters.groupby(["game_id", "pitcher_team_id"]).agg(
    last_inn=("inning", "max"), n=("inning", "size"),
    season=("season", "first"), rate=(T, "mean")).reset_index()
sl = sl[sl.n >= 30]
print(sl.groupby("season")["last_inn"].agg(["mean", "median"]).round(2).to_string())
print("\n선발 소화 이닝 × 제구:")
print(sl.groupby(sl.last_inn.clip(upper=8)).agg(
    경기=("rate", "size"), 성공률=("rate", "mean")).round(4).to_string())

"""KBO 일정 구조 기반 가설 검증.

도메인 지식:
  ① 올스타 브레이크(7월 중순) — 전·후반기 분기, 투수 휴식 후 감각 저하
  ② 9월 확대 엔트리 — 신인·2군 투수 대량 콜업 → 제구 낮은 모집단 유입
  ③ 10월 = 포스트시즌 — 상위팀 에이스만 등판 → 제구 좋은 모집단
  ④ ABS(자동 볼판정): 퓨처스 시범 도입(~2023) → 1군 2024 도입
     ⇒ 관측된 F리그 2022→2023 급락(0.71→0.47), 1군 2023→2024 하락과 시점 일치
  ⑤ 2019 공인구 반발계수 하향 → 2019이 유독 높은 성공률(0.5647)

E27 실측: 세그먼트 편향 손실이 9월 12.6점 / 7월 11.2점으로 최대 → ①②와 부합.
이 가설들이 '모델이 아직 못 잡은 구조'인지, 이미 잡고 있는지 확인한다.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
df["exp_bin"] = pd.cut(df.asof_pitcher_n, [-1, 200, 1000, 5000, 99999],
                       labels=["신인<200", "200-1k", "1k-5k", "베테랑5k+"])

print("=== ② 확대 엔트리: 월별 신인 투수 비중 (경험<200구) ===")
df["_rookie"] = (df.asof_pitcher_n < 200).astype(float)
share = df.pivot_table(index="season", columns="game_month", values="_rookie",
                       aggfunc="mean")
print((share * 100).round(1).to_string())

print("\n=== ①③ 월 × 경험구간 성공률 (2024) ===")
v = df[df.season == 2024]
print(v.pivot_table(index="exp_bin", columns="game_month", values=T,
                    aggfunc="mean", observed=True).round(4).to_string())

print("\n=== ③ 10월 특이성: 게임타입·경험 분해 (전 시즌) ===")
oct_df = df[df.game_month == 10]
print(f"10월 행수 {len(oct_df):,} | 성공률 {oct_df[T].mean():.4f} "
      f"(9월 {df[df.game_month == 9][T].mean():.4f})")
print(oct_df.groupby(["season", "game_type"], observed=True)[T]
      .agg(["size", "mean"]).round(4).to_string())

print("\n=== ④ ABS 가설: F리그 vs R리그 시즌별 (도입 시점 대조) ===")
print(df.pivot_table(index="season", columns="game_type", values=T,
                     aggfunc="mean").round(4).to_string())

print("\n=== ① 올스타 전후 (7월 상순 vs 하순 대용: 6~7월 vs 8월) ===")
for s in [2023, 2024]:
    sub = df[df.season == s]
    for m in [6, 7, 8, 9]:
        g = sub[sub.game_month == m]
        if len(g):
            print(f"  {s} {m:2d}월  n={len(g):6d}  성공률 {g[T].mean():.4f}  "
                  f"신인비중 {(g.asof_pitcher_n < 200).mean() * 100:4.1f}%")

"""데이터에 '직접' 없는 인간적 요소 중 유도 가능한 것 탐색.

① 홈/원정: score_diff_home 과 score_diff_pitcher_team 이 같으면 투수팀이 홈.
   → 이동 피로·마운드 익숙함·관중 요인의 대리
② 포수: 직접 없음. 다만 포수는 팀·시즌 단위로 고정되므로 (팀×시즌)이 약한 대리.
③ 휴식일: game_date 없음. row_id 순서가 시간순이므로 학습 데이터 내에서는 간격 산출
   가능하나, 평가 행에 적용하려면 test 다른 행이 필요 → 규칙 위반. 투수별 '통상 등판 간격'
   만 속성으로 사용 가능.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")

print("=== ① 홈/원정 유도 검증 ===")
df["is_home_p"] = (df.score_diff_home == df.score_diff_pitcher_team).astype(int)
print("top_bottom × is_home_p 교차 (일관성 확인):")
print(pd.crosstab(df.top_bottom, df.is_home_p).to_string())
print("\n홈/원정별 제구 성공률:")
print(df.groupby("is_home_p")[T].agg(["size", "mean"]).round(4).to_string())
print("\n시즌별 홈 이점:")
h = df.pivot_table(index="season", columns="is_home_p", values=T, aggfunc="mean")
h["차이(홈-원정)"] = h[1] - h[0]
print(h.round(4).to_string())

print("\n=== ② 팀×시즌 (포수 진용 대리) ===")
ts = df.groupby(["pitcher_team_id", "season"])[T].agg(["size", "mean"])
ts = ts[ts["size"] > 5000]
print(f"  팀×시즌 셀 {len(ts)}개 | 성공률 표준편차 {ts['mean'].std():.4f}")
piv = df.pivot_table(index="pitcher_team_id", columns="season", values=T,
                     aggfunc="mean")
piv = piv[piv.notna().sum(axis=1) >= 5]
print(piv.round(4).to_string())
print("  팀별 시즌 간 순위 변동(스피어만 자기상관):")
for s in range(2020, 2025):
    prev, cur = piv[s - 1].dropna(), piv[s].dropna()
    idx = prev.index.intersection(cur.index)
    if len(idx) > 5:
        print(f"    {s-1}→{s}: {np.corrcoef(prev[idx].rank(), cur[idx].rank())[0,1]:+.3f}")

print("\n=== ③ 투수별 통상 등판 간격 (row_id 간격 = 리그 전체 투구수 경과) ===")
sub = df[["row_id", "pitcher_id", "season", T]].copy()
sub["rid"] = sub.row_id.str.extract(r"(\d+)").astype(np.int64)
sub = sub.sort_values("rid")
sub["gap"] = sub.groupby("pitcher_id")["rid"].diff()
# 같은 경기 내 연속 투구는 제외 (간격이 매우 작음)
between = sub[sub.gap > 2000]
print(f"  등판 간 간격 중앙값 {between.gap.median():.0f} row (리그 투구수 기준)")
between = between.copy()
between["gap_bin"] = pd.qcut(between.gap, 5, duplicates="drop")
print(between.groupby("gap_bin", observed=True)[T].agg(["size", "mean"]).round(4).to_string())
print("  → 간격이 클수록(휴식 김) 제구가 좋아지는지 확인")

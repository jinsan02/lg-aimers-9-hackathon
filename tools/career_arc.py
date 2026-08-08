"""KBO 투수 커리어 궤적 — 신인 유입과 베테랑 전환 시점.

도메인 배경:
  · KBO 신인 드래프트는 9월 개최, 이듬해 시즌부터 1군 등록 → 신인 유입은 시즌 초
  · 상무/경찰(군 복무)로 약 2년 이탈 → asof_pitcher_n이 시즌 통째로 비는 패턴
  · 통상 3~5년차에 보직 정착

질문: 누적 투구수(경험)에 따라 제구가 언제 안정되는가? 그 곡선이 시즌마다 같은가?
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                 usecols=["season", "game_month", "pitcher_id",
                          "asof_pitcher_n", "asof_pitcher_success_rate", T])

print("=== 1. 경험 구간별 성공률 곡선 (경험 = 통산 누적 투구수) ===")
bins = [-1, 50, 200, 500, 1000, 2000, 4000, 7000, 99999]
labels = ["0-50", "50-200", "200-500", "500-1k", "1k-2k", "2k-4k", "4k-7k", "7k+"]
df["exp"] = pd.cut(df.asof_pitcher_n, bins, labels=labels)
t = df.pivot_table(index="exp", columns="season", values=T,
                   aggfunc="mean", observed=True)
t["전체"] = df.groupby("exp", observed=True)[T].mean()
t["행수"] = df.groupby("exp", observed=True).size()
print(t.round(4).to_string())

print("\n=== 2. 안정화 지점: 구간별 성공률 '변동폭'(시즌 간 표준편차) ===")
sd = t[[c for c in t.columns if isinstance(c, (int, np.integer))]].std(axis=1)
print(sd.round(4).to_string())
print("→ 변동폭이 작아지는 지점 = 시즌 체제 변화에 덜 흔들리는 경험 수준")

print("\n=== 3. 신인 유입 시점 (시즌 첫 등장 투수의 월 분포) ===")
first = df.sort_values(["season", "game_month"]).groupby(
    ["season", "pitcher_id"], as_index=False).first()
newbie = first[first.asof_pitcher_n < 50]
print(newbie.pivot_table(index="season", columns="game_month",
                         values="pitcher_id", aggfunc="count").to_string())

print("\n=== 4. 경력 단절(군 복무 추정): 시즌 건너뛴 투수 ===")
app = df.groupby("pitcher_id")["season"].agg(["min", "max", "nunique"])
app["span"] = app["max"] - app["min"] + 1
gap = app[app["span"] > app["nunique"]]
print(f"등장 시즌에 공백이 있는 투수: {len(gap)}명 / 전체 {len(app)}명 "
      f"({len(gap) / len(app) * 100:.1f}%)")
print(f"  공백 길이 분포: {(gap['span'] - gap['nunique']).value_counts().sort_index().to_dict()}")

print("\n=== 5. 복귀 투수의 제구 (공백 후 첫 시즌 vs 공백 전 마지막 시즌) ===")
gap_ids = set(gap.index)
sub = df[df.pitcher_id.isin(gap_ids)]
per = sub.groupby(["pitcher_id", "season"])[T].mean().reset_index()
rows = []
for pid, g in per.groupby("pitcher_id"):
    ss = sorted(g.season.tolist())
    for i in range(1, len(ss)):
        if ss[i] - ss[i - 1] > 1:   # 공백 직후
            rows.append({"before": g[g.season == ss[i - 1]][T].iloc[0],
                         "after": g[g.season == ss[i]][T].iloc[0]})
r = pd.DataFrame(rows)
if len(r):
    print(f"  n={len(r)} | 공백 전 {r.before.mean():.4f} → 복귀 후 {r.after.mean():.4f} "
          f"(차이 {r.after.mean() - r.before.mean():+.4f})")

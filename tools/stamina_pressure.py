"""체력·심리 가설을 데이터로 직접 검증.

사용자 가설:
  (A) 이닝이 깊어질수록 체력 저하 → 제구 하락
  (B) 시즌 후반(하반기)일수록 누적 피로 → 제구 하락
  (C) 점수차가 작을수록 심리 압박 → 제구 하락
  (D) 체력 저하는 경험이 적을수록 크게 작용

각 효과의 **방향·크기·상호작용**을 실측한다.
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
DATA, T = "./data", "control_success"
df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
v = df[df.season == 2024]        # 최신 체제 기준으로 본다

print("=== (A) 이닝 깊이 → 제구 ===")
a = v.groupby("inning")[T].agg(["size", "mean"])
a = a[a["size"] > 3000]
print(a.round(4).T.to_string())
print(f"  1회 {a.loc[1,'mean']:.4f} → 8회 {a.loc[8,'mean']:.4f} "
      f"= {a.loc[8,'mean'] - a.loc[1,'mean']:+.4f}")

print("\n=== (B) 시즌 진행(월) → 제구 ===")
b = v.groupby("game_month")[T].agg(["size", "mean"])
b = b[b["size"] > 3000]
print(b.round(4).T.to_string())
print(f"  4월 {b.loc[4,'mean']:.4f} → 9월 {b.loc[9,'mean']:.4f} "
      f"= {b.loc[9,'mean'] - b.loc[4,'mean']:+.4f}")

print("\n=== (C) 점수차 → 제구 (심리 압박) ===")
v = v.copy()
v["absdiff"] = v.score_diff_pitcher_team.abs().clip(upper=8)
c = v.groupby("absdiff")[T].agg(["size", "mean"])
print(c.round(4).T.to_string())
print("  → 가설대로면 점수차 작을수록 낮아야 함")

print("\n=== (C-2) li(상황 중요도) 분위 → 제구 ===")
v["li_q"] = pd.qcut(v.li, 6, duplicates="drop")
print(v.groupby("li_q", observed=True)[T].agg(["size", "mean"]).round(4).to_string())

print("\n=== (D) 이닝 × 경험 — 체력 저하가 신인에게 더 큰가 ===")
v["exp_bin"] = pd.cut(v.asof_pitcher_n, [-1, 500, 2000, 99999],
                      labels=["신인<500", "중견500-2k", "베테랑2k+"])
piv = v.pivot_table(index="exp_bin", columns=pd.cut(v.inning, [0, 3, 6, 9, 13],
                                                    labels=["1-3", "4-6", "7-9", "10+"]),
                    values=T, aggfunc="mean", observed=True)
print(piv.round(4).to_string())
print("  각 행의 1-3회 대비 7-9회 변화:")
for idx in piv.index:
    if "1-3" in piv.columns and "7-9" in piv.columns:
        print(f"    {idx}: {piv.loc[idx, '7-9'] - piv.loc[idx, '1-3']:+.4f}")

print("\n=== (B-2) 월 × 경험 — 시즌 피로가 신인에게 더 큰가 ===")
print(v.pivot_table(index="exp_bin", columns="game_month", values=T,
                    aggfunc="mean", observed=True).round(4).to_string())

print("\n=== (C-3) 접전 × 이닝 — 후반 접전이 특히 나쁜가 ===")
v["tight"] = (v.score_diff_pitcher_team.abs() <= 2).map({True: "접전≤2", False: "여유3+"})
print(v.pivot_table(index="tight", columns=pd.cut(v.inning, [0, 6, 9, 13],
                                                  labels=["1-6", "7-9", "10+"]),
                    values=T, aggfunc="mean", observed=True).round(4).to_string())

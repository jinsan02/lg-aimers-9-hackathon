"""EDA — 데이터 구조/결측/타깃 분포/조인 가능성 점검.

실행: WSL에서  ~/.venvs/aimers/bin/python src/eda.py
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)

DATA = "./data"
ID, TARGET = "row_id", "control_success"


def sec(title):
    print(f"\n{'=' * 70}\n## {title}\n{'=' * 70}")


train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig")
test = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig")

sec("1. 기본 구조")
print(f"train {train.shape} | test(sample) {test.shape}")
only_train = set(train.columns) - set(test.columns)
only_test = set(test.columns) - set(train.columns)
print(f"train에만 있는 컬럼: {sorted(only_train)}")
print(f"test에만 있는 컬럼: {sorted(only_test)}")
print(f"train row_id 중복: {train[ID].duplicated().sum()}")

sec("2. 결측률 (결측 있는 컬럼만)")
miss = train.isna().mean().sort_values(ascending=False)
print((miss[miss > 0] * 100).round(2).to_string())

sec("3. 타깃 분포")
print(f"전체 성공률: {train[TARGET].mean():.4f}")
print("\n시즌별 행수/성공률:")
print(train.groupby("season")[TARGET].agg(["size", "mean"]).round(4).to_string())
print("\n볼카운트(balls-strikes)별 성공률 (상위 12):")
bs = train.groupby(["balls_before", "strikes_before"])[TARGET].agg(["size", "mean"]).round(4)
print(bs.to_string())

sec("4. 범주형/저카디널리티 컬럼")
for c in ["top_bottom", "game_type", "base_state", "pitcher_hand", "batter_hand"]:
    vc = train[c].value_counts(dropna=False)
    print(f"{c}: {dict(vc.head(10))}")

sec("5. ID 카디널리티 & cold-start")
for c in ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]:
    print(f"{c}: nunique={train[c].nunique()}")
p_old = set(train.loc[train.season < 2024, "pitcher_id"])
p_24 = set(train.loc[train.season == 2024, "pitcher_id"])
b_old = set(train.loc[train.season < 2024, "batter_id"])
b_24 = set(train.loc[train.season == 2024, "batter_id"])
print(f"2024 신규 투수: {len(p_24 - p_old)}/{len(p_24)} "
      f"| 2024 신규 타자: {len(b_24 - b_old)}/{len(b_24)}")
n24 = train.season == 2024
new_p_rows = n24 & train.pitcher_id.isin(p_24 - p_old)
print(f"2024 행 중 신규 투수 비율: {new_p_rows.sum() / n24.sum():.4f}")

sec("6. asof_* 피처와 타깃 상관 (point-biserial)")
asof_cols = [c for c in train.columns if c.startswith("asof_")]
corr = train[asof_cols + [TARGET]].corr()[TARGET].drop(TARGET)
print(corr.sort_values(key=abs, ascending=False).round(4).to_string())

sec("7. 수치 피처 요약 (주요)")
key_num = ["inning", "li", "home_win_expectancy", "asof_pitcher_n",
           "asof_pitcher_success_rate", "asof_batter_n"]
print(train[key_num].describe().round(3).to_string())

sec("8. trackman_history 조인 가능성")
tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig")
print(f"shape: {tm.shape}")
print(f"시즌: {sorted(tm.season.unique())}")
print(f"pitcher_trackman_id nunique={tm.pitcher_trackman_id.nunique()} "
      f"| train pitcher_id nunique={train.pitcher_id.nunique()}")
inter_p = set(tm.pitcher_trackman_id) & set(train.pitcher_id)
inter_b = set(tm.batter_trackman_id) & set(train.batter_id)
print(f"pitcher ID 교집합: {len(inter_p)} | batter ID 교집합: {len(inter_b)}")
print(f"tm pitcher id 예시: {sorted(set(tm.pitcher_trackman_id))[:5]}")
print(f"train pitcher id 예시: {sorted(set(train.pitcher_id))[:5]}")
print(f"\ntm 결측률(>0):")
tmiss = tm.isna().mean().sort_values(ascending=False)
print((tmiss[tmiss > 0] * 100).round(2).to_string())
print(f"\n구종군 분포: {dict(tm.pitch_type_group.value_counts(dropna=False))}")
print(f"게임타입 유사 컬럼 없음 — tm 컬럼: {list(tm.columns)}")

sec("9. 시즌별 asof 피처 평균 (drift 확인)")
drift_cols = ["asof_pitcher_success_rate", "asof_pitcher_middle_rate",
              "asof_batter_success_rate"]
print(train.groupby("season")[drift_cols + [TARGET]].mean().round(4).to_string())

"""경력 공백(군 복무 등) 피처 — train 이력에서만 산출.

발견(tools/career_arc.py):
  · 792명 중 136명(17.2%)이 시즌 공백을 가짐 (공백 2년 = 54명 → 상무/경찰 복무 패턴)
  · **공백 전 0.5969 → 복귀 후 0.5311 (-0.0658)** — 제구가 뚜렷이 나빠진다
  · asof_pitcher_n은 통산 누적이라 '공백이 있었는지'를 담지 못한다 → 새 정보

산출: (pitcher_id, season) → gap_years
  gap_years = season - (그 이전 마지막 등장 시즌).  1=연속, 2+=공백, -1=신규
평가(2025)는 train(2019~24)에서 해당 투수의 마지막 등장 시즌을 조회 — 상수 lookup이라
평가 데이터의 다른 행을 쓰지 않는다.
"""

import numpy as np
import pandas as pd

DATA = "./data"


def build_gap_table(df):
    """(pitcher_id, season) → gap_years. df는 학습 데이터(시즌·투수 컬럼 필요)."""
    pairs = df[["pitcher_id", "season"]].drop_duplicates().sort_values(
        ["pitcher_id", "season"])
    pairs["prev"] = pairs.groupby("pitcher_id")["season"].shift(1)
    pairs["gap_years"] = (pairs["season"] - pairs["prev"]).fillna(-1)
    return pairs[["pitcher_id", "season", "gap_years"]]


def build_future_lookup(df, target_season):
    """평가 시즌용: 각 투수의 target_season 직전 마지막 등장 시즌 → gap_years."""
    last = (df[df.season < target_season].groupby("pitcher_id")["season"].max()
            .rename("prev").reset_index())
    last["season"] = target_season
    last["gap_years"] = target_season - last["prev"]
    return last[["pitcher_id", "season", "gap_years"]]


def add_gap(df, table):
    out = df.merge(table, on=["pitcher_id", "season"], how="left")
    g = out["gap_years"].fillna(-1)
    out["gap_years"] = g
    out["is_return"] = (g >= 2).astype(np.int8)      # 공백 후 복귀
    out["is_debut"] = (g < 0).astype(np.int8)        # 신규 등장
    return out, ["gap_years", "is_return", "is_debut"]


def main():
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["pitcher_id", "season", "control_success",
                              "asof_pitcher_n"])
    tab = build_gap_table(df)
    out, cols = add_gap(df, tab)
    print("=== gap_years별 제구 성공률 ===")
    print(out.groupby("gap_years").agg(
        n=("control_success", "size"), rate=("control_success", "mean"),
        exp=("asof_pitcher_n", "median")).round(4).to_string())
    print("\n=== 복귀(gap>=2) vs 연속(gap==1) — 경험 수준 통제 ===")
    out["exp_bin"] = pd.cut(out.asof_pitcher_n, [-1, 500, 2000, 99999],
                            labels=["<500", "500-2k", "2k+"])
    print(out[out.gap_years > 0].pivot_table(
        index="exp_bin", columns="is_return", values="control_success",
        aggfunc=["mean", "size"], observed=True).round(4).to_string())
    tab.to_csv(f"{DATA}/processed/career_gap.csv", index=False)
    print(f"\n저장: career_gap.csv {tab.shape}")


if __name__ == "__main__":
    main()

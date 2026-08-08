"""시즌 이전 투수 이력으로 roster/league transition 피처를 만든다.

시즌 S 행은 오직 시즌 < S의 등장 이력만 사용한다. 학습 데이터의 마지막
시즌 다음 해까지 표를 만들어 제출 추론에서도 테스트 행끼리 상태를 갱신하지
않는다. 타깃은 전혀 사용하지 않는다.
"""

import numpy as np
import pandas as pd


COLS = [
    "rt_seen_before", "rt_prev_is_f", "rt_prev_is_r",
    "rt_f_to_r", "rt_r_to_f", "rt_same_cont", "rt_same_return",
    "rt_gap_years", "rt_prior_r_log", "rt_prior_f_log",
    "rt_same_league_log", "rt_other_league_log",
]


def build_table(df):
    """(pitcher_id, season)별 시즌 시작 전 이력을 반환한다."""
    need = {"pitcher_id", "season", "game_type"}
    missing = need.difference(df.columns)
    if missing:
        raise KeyError(f"roster transition 필수 컬럼 없음: {sorted(missing)}")

    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    if not seasons:
        raise ValueError("roster transition: season이 비어 있음")
    horizon = list(range(min(seasons), max(seasons) + 2))
    pitchers = df["pitcher_id"].drop_duplicates().tolist()
    grid = pd.MultiIndex.from_product(
        [pitchers, horizon], names=["pitcher_id", "season"])

    counts = (df.groupby(["pitcher_id", "season", "game_type"], sort=False)
                .size().unstack("game_type", fill_value=0))
    for league in ("R", "F"):
        if league not in counts:
            counts[league] = 0
    counts = counts[["R", "F"]].reindex(grid, fill_value=0).sort_index()

    prior = (counts.groupby(level="pitcher_id", sort=False).cumsum()
                   .groupby(level="pitcher_id", sort=False).shift(1)
                   .fillna(0))
    prior.columns = ["prior_r_n", "prior_f_n"]

    order = "row_id" if "row_id" in df.columns else None
    z = df.sort_values(order, kind="stable") if order else df
    last = (z.groupby(["pitcher_id", "season"], sort=False).tail(1)
              .set_index(["pitcher_id", "season"])["game_type"]
              .reindex(grid))
    # 비등장 시즌에는 마지막 실제 등장 리그/시즌을 유지하되, 현재 시즌은 제외한다.
    prev_league = (last.groupby(level="pitcher_id", sort=False).ffill()
                       .groupby(level="pitcher_id", sort=False).shift(1))
    appeared_season = pd.Series(
        np.where(last.notna(), last.index.get_level_values("season"), np.nan),
        index=grid, dtype=float)
    prev_season = (appeared_season.groupby(level="pitcher_id", sort=False).ffill()
                                  .groupby(level="pitcher_id", sort=False).shift(1))

    tab = prior.copy()
    tab["prev_league"] = prev_league
    tab["prev_season"] = prev_season
    return tab.reset_index()


def add_features(df, table):
    """고정 과거표를 현재 행의 리그와 결합한다 (행 독립)."""
    out = df.merge(table, on=["pitcher_id", "season"], how="left",
                   validate="many_to_one", sort=False)
    pr = out["prior_r_n"].fillna(0.0).astype(float)
    pf = out["prior_f_n"].fillna(0.0).astype(float)
    prev = out["prev_league"]
    cur = out["game_type"]
    gap = out["season"].astype(float) - out["prev_season"]
    seen = (pr + pf) > 0

    out["rt_seen_before"] = seen.astype(np.int8)
    out["rt_prev_is_f"] = (prev == "F").astype(np.int8)
    out["rt_prev_is_r"] = (prev == "R").astype(np.int8)
    out["rt_f_to_r"] = (seen & (prev == "F") & (cur == "R")).astype(np.int8)
    out["rt_r_to_f"] = (seen & (prev == "R") & (cur == "F")).astype(np.int8)
    same = seen & (prev == cur)
    out["rt_same_cont"] = (same & (gap == 1)).astype(np.int8)
    out["rt_same_return"] = (same & (gap >= 2)).astype(np.int8)
    out["rt_gap_years"] = gap.fillna(-1.0).astype(np.float32)
    out["rt_prior_r_log"] = np.log1p(pr).astype(np.float32)
    out["rt_prior_f_log"] = np.log1p(pf).astype(np.float32)
    same_n = np.where(cur == "R", pr, pf)
    other_n = np.where(cur == "R", pf, pr)
    out["rt_same_league_log"] = np.log1p(same_n).astype(np.float32)
    out["rt_other_league_log"] = np.log1p(other_n).astype(np.float32)
    return out.drop(columns=["prior_r_n", "prior_f_n", "prev_league",
                             "prev_season"]), list(COLS)

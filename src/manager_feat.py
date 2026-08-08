"""감독 투수 운용 스타일 → 팀×시즌 속성 피처.

분석(tools/manager_style.py):
  · 경기당 팀 등판 투수: 평균 4.92 (2019 4.55 → 2020~ 5.0으로 정착)
  · 팀별 성향 폭 1.23명 (20번팀 4.38 '선발 의존' ~ 13번팀 5.62 '불펜 야구')
  · 팀×시즌으로 크게 변동 → 감독 교체·전략 변화 반영

평가(2025)는 해당 팀의 **직전 시즌(2024) 스타일**을 사용 — train 집계 상수라 규칙 적합.
"""

import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"


def reconstruct_games(df):
    """row_id 시간순 + 팀 쌍 고정 → 경기 경계 복원."""
    rid = df["row_id"].str.extract(r"(\d+)").astype(np.int64).iloc[:, 0]
    order = np.argsort(rid.to_numpy())
    d = df.iloc[order].reset_index(drop=True)
    pair = np.sort(d[["pitcher_team_id", "batter_team_id"]].to_numpy(), axis=1)
    key = pair[:, 0] * 1000 + pair[:, 1]
    mon = d["game_month"].to_numpy()
    new = np.r_[True, (key[1:] != key[:-1]) | (mon[1:] != mon[:-1])]
    d["game_id"] = np.cumsum(new)
    return d


def build_style_table(df):
    """(pitcher_team_id, season) → 운용 스타일 지표."""
    d = reconstruct_games(df)
    per = d.groupby(["game_id", "pitcher_team_id", "season"]).agg(
        n_pitchers=("pitcher_id", "nunique"),
        n_pitches=("pitcher_id", "size")).reset_index()
    per = per[per.n_pitches >= 40]
    tab = per.groupby(["pitcher_team_id", "season"]).agg(
        mgr_pitchers=("n_pitchers", "mean"),
        mgr_pitches=("n_pitches", "mean")).reset_index()
    return tab


def add_style(df, tab, future_season=None):
    """future_season이 지정되면 그 시즌 행은 직전 시즌 스타일로 채운다."""
    out = df.merge(tab, on=["pitcher_team_id", "season"], how="left")
    cols = ["mgr_pitchers", "mgr_pitches"]
    if out[cols[0]].isna().any():
        latest = (tab.sort_values("season").groupby("pitcher_team_id")[cols]
                  .last().reset_index())
        fill = out[["pitcher_team_id"]].merge(latest, on="pitcher_team_id",
                                              how="left")
        for c in cols:
            v = out[c].to_numpy(dtype=float)
            f = fill[c].to_numpy(dtype=float)
            out[c] = np.where(np.isnan(v), f, v)
    return out, cols


def main():
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=["row_id", "season", "game_month", "pitcher_id",
                              "pitcher_team_id", "batter_team_id", T])
    tab = build_style_table(df)
    print("=== 팀×시즌 운용 스타일 ===")
    print(tab.pivot_table(index="pitcher_team_id", columns="season",
                          values="mgr_pitchers").round(2).to_string())
    out, cols = add_style(df, tab)
    out["style_bin"] = pd.qcut(out.mgr_pitchers, 5, duplicates="drop")
    print("\n=== 운용 스타일 → 제구 성공률 ===")
    print(out.groupby("style_bin", observed=True)[T]
          .agg(["size", "mean"]).round(4).to_string())
    tab.to_csv(f"{DATA}/processed/manager_style.csv", index=False)
    print(f"\n저장: manager_style.csv {tab.shape}")


if __name__ == "__main__":
    main()

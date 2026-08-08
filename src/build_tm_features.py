"""trackman 기반 투수 피처 생성 (leak-safe: 대상 시즌 이전 시즌들만 사용).

산출:
  data/processed/tm_pitcher_feats.csv  — (pitcher_id, season) → tm_* 피처
    season=S 행은 trackman의 시즌 < S 데이터만 집계 (2019는 결측).
    season=2025 행 = 2019~2024 전체 집계 (평가용).

실행: ~/.venvs/aimers/bin/python src/build_tm_features.py
"""

import numpy as np
import pandas as pd

DATA = "./data"

AGG_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side", "zone_speed"]


def main():
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map.csv")
    # 투수별 대표 tm_id: 시즌 매칭들의 최빈값 (동률이면 score 합 최대)
    rep = (pmap.groupby(["pitcher_id", "tm_id"])["score"].agg(["count", "sum"])
           .reset_index()
           .sort_values(["count", "sum"], ascending=False)
           .drop_duplicates("pitcher_id")[["pitcher_id", "tm_id"]])
    print(f"투수 매핑: {len(rep)}명")

    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id",
                              "pitch_type_group"] + AGG_COLS)

    g = tm.groupby(["pitcher_trackman_id", "season"])
    per_season = g[AGG_COLS].mean()
    per_season["n"] = g.size()
    # fastball 구속 (구종군별 대표 속도)
    fb = (tm[tm.pitch_type_group == "fastball"]
          .groupby(["pitcher_trackman_id", "season"])["rel_speed"].mean()
          .rename("fb_speed"))
    mix = (tm.pivot_table(index=["pitcher_trackman_id", "season"],
                          columns="pitch_type_group", values="rel_speed",
                          aggfunc="size").fillna(0))
    mix = mix.div(mix.sum(1), axis=0)[["fastball", "breaking", "offspeed"]]
    mix.columns = ["fb_rate", "br_rate", "os_rate"]
    per_season = per_season.join(fb).join(mix).reset_index()

    # 누적(이전 시즌 전용) 피처: 대상 시즌 S에 대해 season < S 가중평균(n 가중)
    feats = []
    val_cols = AGG_COLS + ["fb_speed", "fb_rate", "br_rate", "os_rate"]
    for target in range(2019, 2026):
        prior = per_season[per_season.season < target]
        if prior.empty:
            continue
        w = prior.copy()
        agg = {}
        for c in val_cols:
            agg[c] = (w[c] * w["n"]).groupby(w.pitcher_trackman_id).sum() / \
                     w["n"].groupby(w.pitcher_trackman_id).sum()
        res = pd.DataFrame(agg)
        res["tm_n_prior"] = w.groupby("pitcher_trackman_id")["n"].sum()
        res = res.reset_index().rename(columns={"pitcher_trackman_id": "tm_id"})
        res["season"] = target
        feats.append(res)
    feats = pd.concat(feats, ignore_index=True)

    out = rep.merge(feats, on="tm_id")[["pitcher_id", "season"] +
                                       val_cols + ["tm_n_prior"]]
    out.columns = ["pitcher_id", "season"] + [f"tm_{c}" for c in val_cols] + ["tm_n_prior"]
    out.to_csv(f"{DATA}/processed/tm_pitcher_feats.csv", index=False)
    print(f"저장: tm_pitcher_feats.csv {out.shape}")
    print(out.groupby("season").size().rename("투수 수").to_string())


if __name__ == "__main__":
    main()

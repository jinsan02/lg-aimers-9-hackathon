"""trackman <-> train 투수 링키지 재구축 (계획 ②-a).

기존 pitcher_map.csv 는 정확도 86% 로 추정돼 모든 tm 피처를 희석하고 있었다.

착안: trackman 에 train 과 **동일한 상황 컬럼**이 전부 있다.
  season, game_month, game_dayofweek, inning, top_bottom,
  balls_before, strikes_before, outs_before, pitcher_hand
따라서 투구를 이 튜플로 묶으면, 같은 투수는 두 데이터에서 **같은 상황 분포**를
갖는다. (pitcher_id, tm_id) 쌍마다 공유 키에서의 min(count) 를 더하면
공기(co-occurrence) 점수가 되고, 진짜 짝이 압도적으로 높게 나온다.

투구수 궤적(시즌별 투구수 벡터)만 쓰는 것보다 훨씬 강하다 - 상황 튜플은
차원이 수만 개라 우연 일치가 거의 없다.

⚠️ 용도 제한: 이 매핑은 **tm_id 를 식별하는 데만** 쓴다. 개별 투구의 trackman
실측값(구속/무브먼트/구종)을 그 train 행에 붙이면 '투구 이후 확정 정보' 규정
위반이다. 매핑으로 만드는 건 **직전 시즌까지의 투수 단위 요약**뿐이다.

메모리: usecols 로 필요한 컬럼만 읽는다 (trackman 1.79M x 9, train 1.48M x 9).
        노트북에서 돌리지 말 것 - 4070/A100 에서 실행.

실행: python src/link_pitchers.py
산출: data/processed/pitcher_map2.csv  (pitcher_id, tm_id, score, ratio, n)
"""

import os
import sys

import numpy as np
import pandas as pd

DATA = "./data"
OUT = f"{DATA}/processed/pitcher_map2.csv"

KEYS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
        "balls_before", "strikes_before", "outs_before"]
HAND = {"Right": "R", "Left": "L", "R": "R", "L": "L"}


def key_code(df):
    """상황 튜플을 하나의 정수 코드로 (해시 대신 자릿수 조합 - 충돌 없음)."""
    tb = (df["top_bottom"].astype(str).str[0].str.upper() == "T").astype(np.int64)
    inn = df["inning"].clip(1, 15).astype(np.int64)
    x = df["season"].astype(np.int64) - 2018
    # Sequential mixed-radix encoding is easier to audit than one deeply
    # parenthesized expression and is exactly the implementation used by the
    # independent linkage audit.
    for value, width in (
        (df["game_month"].astype(np.int64), 13),
        (df["game_dayofweek"].astype(np.int64), 8),
        (inn, 16), (tb, 2),
        (df["balls_before"].clip(0, 3).astype(np.int64), 4),
        (df["strikes_before"].clip(0, 2).astype(np.int64), 3),
        (df["outs_before"].clip(0, 2).astype(np.int64), 3),
        (df["bhand"].astype(np.int64), 2),
    ):
        x = x * width + value
    return x


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=KEYS + ["pitcher_id", "pitcher_hand", "batter_hand"])
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=KEYS + ["pitcher_trackman_id", "pitcher_hand", "batter_hand"])
    print(f"train {len(tr):,} | trackman {len(tm):,}")

    # train 은 정수 코드(1/2), trackman 은 Right/Left. 인코딩이 다르므로
    # **다수 클래스끼리 대응**시킨다 (우완이 양쪽 모두 ~73% 다수).
    def norm_hand(s, name):
        vc = s.value_counts()
        maj = vc.index[0]
        print(f"  {name} 손 분포: {dict(list(vc.items())[:3])} -> 다수 {maj} = R")
        return np.where(s == maj, "R", "L")

    tr["hand"] = norm_hand(tr["pitcher_hand"], "train")
    tm["hand"] = norm_hand(tm["pitcher_hand"], "trackman")
    def norm_batter_hand(s):
        if pd.api.types.is_numeric_dtype(s):
            return s.map({1: 0, 2: 1}).fillna(-1).astype(np.int8)
        return (s.astype(str).str.upper().str[0]
                .map({"L": 0, "R": 1}).fillna(-1).astype(np.int8))

    tr["bhand"] = norm_batter_hand(tr["batter_hand"])
    tm["bhand"] = norm_batter_hand(tm["batter_hand"])
    tr["k"] = key_code(tr)
    tm["k"] = key_code(tm)

    a = tr.groupby(["k", "pitcher_id", "hand"]).size().rename("na").reset_index()
    b = tm.groupby(["k", "pitcher_trackman_id", "hand"]).size() \
        .rename("nb").reset_index().rename(columns={"pitcher_trackman_id": "tm_id"})
    print(f"상황키 {tr.k.nunique():,}개 | train측 조합 {len(a):,} | tm측 {len(b):,}")

    # 손이 다르면 애초에 같은 사람이 아니다 - 여기서 후보를 크게 줄인다
    m = a.merge(b, on=["k", "hand"], how="inner")
    m["ov"] = np.minimum(m["na"], m["nb"])
    print(f"후보 쌍-키 {len(m):,}")

    sc = m.groupby(["pitcher_id", "tm_id"])["ov"].sum().rename("score") \
        .reset_index()
    tot_a = a.groupby("pitcher_id")["na"].sum().rename("n_train")
    tot_b = b.groupby("tm_id")["nb"].sum().rename("n_tm")
    sc = sc.join(tot_a, on="pitcher_id").join(tot_b, on="tm_id")
    # 정규화: 두 쪽 투구수의 작은 쪽 대비 몇 %가 겹치는가
    sc["ratio"] = sc["score"] / np.minimum(sc["n_train"], sc["n_tm"])

    sc = sc.sort_values(["pitcher_id", "score"], ascending=[True, False])
    top = sc.groupby("pitcher_id").head(2).copy()
    top["rk"] = top.groupby("pitcher_id").cumcount()
    r0 = top[top.rk == 0].set_index("pitcher_id")
    r1 = top[top.rk == 1].set_index("pitcher_id")
    best = pd.DataFrame({
        "tm_id": r0["tm_id"],
        "score": r0["score"],
        "ratio": r0["ratio"],
        "score2": r1["score"].reindex(r0.index).fillna(0.0),
    }).reset_index()
    # 1위/2위 격차 = 확신도. 1.5배 이상이면 사실상 확정.
    best["margin"] = best["score"] / np.maximum(best["score2"], 1.0)

    # tm_id 중복 배정 방지 - 점수 높은 쪽이 가져간다
    best = best.sort_values("score", ascending=False)
    best["dup"] = best.duplicated("tm_id")
    conf = best[(~best.dup) & (best.margin >= 1.5)]

    os.makedirs(f"{DATA}/processed", exist_ok=True)
    conf[["pitcher_id", "tm_id", "score", "ratio", "margin"]] \
        .sort_values("pitcher_id").to_csv(OUT, index=False)

    print(f"\n투수 {best.pitcher_id.nunique()}명 중 확정 매칭 {len(conf)}명 "
          f"({len(conf) / best.pitcher_id.nunique() * 100:.1f}%)")
    print(f"  tm_id 중복 배정 제외 {int(best.dup.sum())}건")
    print(f"  margin 분포: "
          + "  ".join(f"{q}분위 {best.margin.quantile(q / 100):.2f}"
                      for q in (10, 25, 50, 75, 90)))
    print(f"  ratio(겹침률) 중앙값 {conf.ratio.median():.3f}")
    print(f"저장 {OUT}")

    old = f"{DATA}/processed/pitcher_map.csv"
    if os.path.exists(old):
        o = pd.read_csv(old)
        rep = (o.groupby(["pitcher_id", "tm_id"])["score"].agg(["count", "sum"])
               .reset_index()
               .sort_values(["pitcher_id", "count", "sum"],
                            ascending=[True, False, False])
               .drop_duplicates("pitcher_id")[["pitcher_id", "tm_id"]])
        j = conf.merge(rep, on="pitcher_id", suffixes=("_new", "_old"))
        agree = (j.tm_id_new == j.tm_id_old).mean()
        print(f"\n기존 매핑과 일치율 {agree * 100:.1f}% (공통 {len(j)}명)")
        print("  -> 불일치분이 기존 86% 추정의 오차일 가능성이 크다")
    return 0


if __name__ == "__main__":
    sys.exit(main())

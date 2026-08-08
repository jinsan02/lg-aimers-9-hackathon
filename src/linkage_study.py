"""trackman ↔ train 레코드 링키지 타당성 분석.

ID 조인 불가 확인됨 → 상황 키(시즌/월/요일/이닝/초말/카운트/손잡이)로
얼마나 유일하게 매칭되는지 정량화한다.

실행: ~/.venvs/aimers/bin/python src/linkage_study.py
"""

import numpy as np
import pandas as pd

DATA = "./data"

KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before",
       "pitcher_hand", "batter_hand"]


def main():
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=KEY + ["pitcher_id", "batter_id"])
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig",
                     usecols=KEY + ["pitcher_trackman_id", "batter_trackman_id",
                                    "trackman_game_id", "pitch_no"])
    # 인코딩 정규화: tm(Top/Bottom, Left/Right) → train(T/B, 1/2)
    # train pitcher_hand 분포상 2가 다수(74%) → 2=Right, 1=Left
    tm["top_bottom"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    for c in ["pitcher_hand", "batter_hand"]:
        tm[c] = tm[c].map({"Left": 1, "Right": 2})
    print("정규화 후 tm hands:", sorted(tm.pitcher_hand.dropna().unique()),
          "| top_bottom:", sorted(tm.top_bottom.dropna().unique()))

    # tm hand가 문자면 train 정수 코드와 맞춰야 함 — 분포로 판단해 출력만
    n_tr, n_tm = len(tr), len(tm)
    print(f"train {n_tr} rows | trackman {n_tm} rows")

    # 상황 키 멀티플리시티
    tm_counts = tm.groupby(KEY, dropna=False).size()
    tr_key = tr.set_index(KEY).index
    mapped = tr_key.map(tm_counts)
    m = pd.Series(mapped).fillna(0)
    print("\ntrain 행 기준, 동일 상황키를 가진 trackman 행 수 분포:")
    print(f"  0개(매칭불가): {(m == 0).mean() * 100:.1f}%")
    print(f"  정확히 1개  : {(m == 1).mean() * 100:.1f}%")
    print(f"  2~10개      : {((m >= 2) & (m <= 10)).mean() * 100:.1f}%")
    print(f"  11개 이상   : {(m > 10).mean() * 100:.1f}%")
    print(f"  중앙값 {m.median():.0f} | 평균 {m.mean():.1f}")

    # 투수 단위 링키지 가능성: (season, hand)별 투구수 시그니처 상관
    tr_p = tr.groupby(["season", "pitcher_id"]).size().rename("n_tr")
    tm_p = tm.groupby(["season", "pitcher_trackman_id"]).size().rename("n_tm")
    print("\n시즌별 투수 수 (train vs trackman):")
    a = tr_p.groupby("season").size()
    b = tm_p.groupby("season").size()
    print(pd.DataFrame({"train": a, "trackman": b}).to_string())

    # 시즌×투구수 분포 유사성 → 투구수 시그니처 매칭 여지
    for s in [2019, 2024]:
        x = np.sort(tr_p.loc[s].to_numpy())[::-1][:10]
        z = np.sort(tm_p.loc[s].to_numpy())[::-1][:10]
        print(f"\n{s} 상위 10 투수 투구수  train: {x}")
        print(f"{' ' * len(str(s))} 상위 10 투수 투구수  tm   : {z}")


if __name__ == "__main__":
    main()

"""투수 '유형' 클러스터 — 원시 ID의 대안.

문제: pitcher_id(792종)를 넣으면 과거 시즌을 암기해 drift에 무너진다(E06, -205).
      그렇다고 빼면 투수 개성 정보가 asof_* 평균으로만 남는다.

해법: 행의 asof_* 프로필을 K개 유형으로 양자화한 **pitcher_type** 범주형 피처.
  - 유형 수가 적어(K≤40) 암기 불가 → 일반화됨
  - 신인도 자기 프로필로 자동 배정 → cold-start 문제 없음
  - CatBoost의 ordered target statistics가 "이 유형은 X% 성공"을 매끄럽게 학습
  - 행 단위 계산이라 평가 데이터의 다른 행을 전혀 쓰지 않음 (규칙 적합)

실행: uv run python src/pitcher_cluster.py
산출: model/pitcher_cluster.pkl (스케일러 + KMeans)
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import QuantileTransformer

DATA, TARGET = "./data", "control_success"
PROFILE = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
           "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
           "asof_pitcher_offspeed_rate", "asof_pitcher_n"]
K = 24


def fit(df_fit, k=K, seed=0):
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=500,
                             random_state=seed)
    X = qt.fit_transform(df_fit[PROFILE].fillna(df_fit[PROFILE].median()))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10,
                         batch_size=4096).fit(X)
    return {"qt": qt, "km": km, "cols": PROFILE,
            "med": df_fit[PROFILE].median().to_dict()}


def assign(df, pack):
    X = df[pack["cols"]].fillna(pd.Series(pack["med"]))
    return pack["km"].predict(pack["qt"].transform(X)).astype(np.int16)


def main():
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    feats = [c for c in test_cols if c != "row_id"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=feats + [TARGET])
    fit_mask = df.season < 2024          # 검증 누수 차단
    pack = fit(df[fit_mask])
    df["ptype"] = assign(df, pack)

    joblib.dump(pack, "model/pitcher_cluster.pkl", compress=3)
    prof = df.groupby("ptype").agg(
        n=("ptype", "size"), rate=(TARGET, "mean"),
        psr=("asof_pitcher_success_rate", "mean"),
        prev=("asof_pitcher_reverse_rate", "mean"),
        fb=("asof_pitcher_fastball_rate", "mean"),
        exp=("asof_pitcher_n", "median")).sort_values("rate")
    print(f"K={K} 유형 | 성공률 범위 {prof.rate.min():.4f} ~ {prof.rate.max():.4f} "
          f"(폭 {prof.rate.max() - prof.rate.min():.4f})")
    print(prof.round(4).to_string())
    print("\n저장: model/pitcher_cluster.pkl")


if __name__ == "__main__":
    main()

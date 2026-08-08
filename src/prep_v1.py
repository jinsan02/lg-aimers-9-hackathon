"""전처리 v1 — NN(TabM)용 배열 생성. 반드시 노트북 WSL(sklearn 1.8.0)에서 실행.

- 범주형 9개 → OrdinalEncoder(+1 시프트, unknown/결측=0) → 임베딩 인덱스
- 수치형 38개 → median 임퓨트 → QuantileTransformer(normal)
- 산출:
    data/processed/train_v1.npz  (X_num f32, X_cat i32, y i8, season i16)
    model/prep_v1.pkl            (전처리기 — 제출 script.py에서 재사용)

실행: ~/.venvs/aimers/bin/python src/prep_v1.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

DATA = "./data"
ID, TARGET = "row_id", "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id",
            "pitcher_id", "batter_id"]


def main():
    t0 = time.time()
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    num_cols = [c for c in features if c not in CAT_COLS]

    train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                        usecols=features + [TARGET])
    print(f"load {train.shape} :: {time.time() - t0:.1f}s")

    cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1, dtype=np.float64)
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("quantile", QuantileTransformer(output_distribution="normal",
                                         n_quantiles=1000, random_state=42)),
    ])

    t0 = time.time()
    X_cat = cat_enc.fit_transform(train[CAT_COLS]).astype(np.int32) + 1  # 0 = unknown
    X_num = num_pipe.fit_transform(train[num_cols]).astype(np.float32)
    print(f"transform :: {time.time() - t0:.1f}s")

    cardinalities = [len(c) + 1 for c in cat_enc.categories_]  # +1 for unknown=0
    print(f"cat cardinalities: {dict(zip(CAT_COLS, cardinalities))}")
    print(f"num features: {len(num_cols)}")

    os.makedirs(f"{DATA}/processed", exist_ok=True)
    np.savez_compressed(
        f"{DATA}/processed/train_v1.npz",
        X_num=X_num, X_cat=X_cat,
        y=train[TARGET].to_numpy(np.int8),
        season=train["season"].to_numpy(np.int16),
    )
    os.makedirs("./model", exist_ok=True)
    joblib.dump({"cat_enc": cat_enc, "num_pipe": num_pipe,
                 "cat_cols": CAT_COLS, "num_cols": num_cols,
                 "cardinalities": cardinalities},
                "./model/prep_v1.pkl", compress=3)
    size = os.path.getsize(f"{DATA}/processed/train_v1.npz") / 1e6
    print(f"저장 완료: train_v1.npz ({size:.0f} MB), prep_v1.pkl")


if __name__ == "__main__":
    main()

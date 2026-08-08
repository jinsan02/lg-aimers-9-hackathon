"""학습 파이프라인 — RandomForest 베이스라인.

2019~2023 학습 / 2024 검증으로 Brier Skill Score를 확인한 뒤,
전체 데이터로 재학습하여 ./model/rf.pkl 로 저장한다.

실행: uv run python src/train.py   (프로젝트 루트에서)
"""

import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./data"
MODEL_PATH = "./model/rf.pkl"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VAL_SEASON = 2024


def brier_skill_score(y_true, y_pred):
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    baseline = r * (1 - r)
    return max(0.0, 100000 * (1 - brier / baseline)), brier, baseline


def load_train():
    # 피처 목록은 test.csv 가 기준 — train 에만 있는 컬럼은 쓰지 않는다.
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                            encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                        encoding="utf-8-sig", usecols=features + [TARGET])
    return train, features


def build_model(features):
    num_cols = [c for c in features if c not in CAT_COLS]
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=200,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def main():
    train, features = load_train()
    print(f"train: {train.shape} | 피처 {len(features)}개 "
          f"| 시즌 {train['season'].min()}~{train['season'].max()} "
          f"| 성공률 {train[TARGET].mean():.4f}")

    model = build_model(features)

    # ---- 2024 홀드아웃 검증 ----
    is_val = train["season"] == VAL_SEASON
    t = time.time()
    model.fit(train.loc[~is_val, features], train.loc[~is_val, TARGET])
    print(f"검증용 학습 완료 :: {time.time() - t:.1f}s")

    val_pred = model.predict_proba(train.loc[is_val, features])[:, 1]
    score, brier, baseline = brier_skill_score(train.loc[is_val, TARGET], val_pred)
    print(f"[{VAL_SEASON} 검증] Brier {brier:.6f} | 기준선 {baseline:.6f} "
          f"| Score {score:.2f}")

    # ---- 전체 재학습 & 저장 ----
    t = time.time()
    model.fit(train[features], train[TARGET])
    print(f"전체 재학습 완료 :: {time.time() - t:.1f}s")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()

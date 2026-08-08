"""추론 스크립트 — 제출 zip 에 script.py 로 포함되어 평가 서버가 실행한다.

서버 기준 상대경로: ./data/test.csv, ./data/sample_submission.csv,
./model/rf.pkl → ./output/submission.csv
로컬에서도 프로젝트 루트에서 그대로 실행해 스모크 테스트한다.
"""

import os

import joblib
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


def build_features(df):
    """학습 때와 동일 — row_id 만 제외. 인코딩/결측 대치는 파이프라인이 수행."""
    return df.drop(columns=[ID_COL])


def merge_predictions(sub, ids, preds):
    """sample_submission 의 row_id 순서에 맞춰 예측 확률 병합."""
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def main():
    TEST_PATH = "./data/test.csv"
    SAMPLE_SUB_PATH = "./data/sample_submission.csv"
    MODEL_PATH = "./model/rf.pkl"
    OUT_PATH = "./output/submission.csv"

    print("Load model...")
    model = joblib.load(MODEL_PATH)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sub.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()

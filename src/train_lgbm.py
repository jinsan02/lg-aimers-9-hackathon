"""LightGBM 학습 — 2024 홀드아웃 검증 + 전체 재학습 저장.

범주형 9개는 LightGBM 네이티브 categorical로 처리.
실행: ~/.venvs/aimers/bin/python src/train_lgbm.py
"""

import argparse
import os
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = "./data"
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id",
            "pitcher_id", "batter_id"]


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--min-data", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cat-smooth", type=float, default=10.0)
    ap.add_argument("--cat-l2", type=float, default=10.0)
    ap.add_argument("--max-cat-threshold", type=int, default=32)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--drop", default="", help="제외할 피처 (쉼표 구분)")
    ap.add_argument("--recency-halflife", type=float, default=0.0,
                    help=">0이면 시즌 기준 지수 가중치 half-life(년)")
    ap.add_argument("--no-refit", action="store_true",
                    help="검증만 하고 전체 재학습 생략")
    args = ap.parse_args()

    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    drop = {c for c in args.drop.split(",") if c}
    features = [c for c in test_cols if c != ID and c not in drop]
    cat_cols = [c for c in CAT_COLS if c in features]

    train = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                        usecols=list({*features, TARGET, "season"}))
    for c in cat_cols:
        train[c] = train[c].astype("category")

    is_val = train["season"] == 2024
    X_tr, y_tr = train.loc[~is_val, features], train.loc[~is_val, TARGET]
    X_va, y_va = train.loc[is_val, features], train.loc[is_val, TARGET]

    w_tr = None
    if args.recency_halflife > 0:
        age = 2024 - train.loc[~is_val, "season"].to_numpy(np.float64)
        w_tr = 0.5 ** (age / args.recency_halflife)
        print(f"recency weight: halflife={args.recency_halflife}년 "
              f"(2019 w={w_tr.min():.3f})")

    params = dict(
        objective="binary", metric="binary_logloss",
        learning_rate=args.lr, num_leaves=args.leaves,
        min_data_in_leaf=args.min_data,
        cat_smooth=args.cat_smooth, cat_l2=args.cat_l2,
        max_cat_threshold=args.max_cat_threshold,
        feature_fraction=args.feature_fraction,
        bagging_fraction=0.8, bagging_freq=1,
        seed=args.seed, verbose=-1, num_threads=os.cpu_count(),
    )

    t0 = time.time()
    dtr = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_cols, weight=w_tr)
    dva = lgb.Dataset(X_va, y_va, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=args.rounds,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(args.early, verbose=False),
                                   lgb.log_evaluation(200)])
    print(f"학습 :: {time.time() - t0:.0f}s | best_iter={booster.best_iteration}")

    val_pred = booster.predict(X_va, num_iteration=booster.best_iteration)
    score = bss(y_va.to_numpy(np.float64), val_pred)
    print(f"[2024 검증] BSS {score:.2f}")
    os.makedirs("./out", exist_ok=True)
    np.savez_compressed(f"./out/lgbm_{args.tag}_val_preds.npz",
                        y=y_va.to_numpy(np.float64), pred=val_pred)

    if not args.no_refit:
        t0 = time.time()
        w_all = None
        if args.recency_halflife > 0:
            age = 2025 - train["season"].to_numpy(np.float64)
            w_all = 0.5 ** (age / args.recency_halflife)
        dall = lgb.Dataset(train[features], train[TARGET],
                           categorical_feature=cat_cols, weight=w_all)
        final = lgb.train({**params}, dall,
                          num_boost_round=booster.best_iteration)
        os.makedirs("./model", exist_ok=True)
        # category 레벨을 추론 시 재현하기 위해 카테고리 목록도 저장
        cat_levels = {c: list(train[c].cat.categories) for c in cat_cols}
        joblib.dump({"booster": final, "features": features,
                     "cat_cols": cat_cols, "cat_levels": cat_levels,
                     "best_iteration": booster.best_iteration,
                     "val_bss": score},
                    f"./model/lgbm_{args.tag}.pkl", compress=3)
        print(f"재학습+저장 :: {time.time() - t0:.0f}s → model/lgbm_{args.tag}.pkl")

    imp = pd.Series(booster.feature_importance("gain"), index=features)
    print("\nTop-15 피처 (gain):")
    print(imp.sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()

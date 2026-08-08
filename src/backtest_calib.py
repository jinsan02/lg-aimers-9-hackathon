"""캘리브레이션 체인 백테스트 — "마지막 시즌으로 보정기 적합 → 다음 시즌 적용" 검증.

구조: 2019~2022 학습 → 2023 예측(보정기 적합용) + 2024 예측(테스트)
비교: ① 무보정 ② 전역 δ(추세 r_hat) ③ isotonic ④ isotonic+δ ⑤ 월별 δ_m

모든 보정기는 2023까지 정보로만 적합 → 2024 적용은 행 독립(고정 함수) — 규칙 적합.
실행(4070): python -m uv run python src\backtest_calib.py
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression

from features import NEW_CAT, add_features, compute_priors

DATA = "./data"
ID, TARGET = "row_id", "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_hand",
            "batter_hand", "pitcher_team_id", "batter_team_id"]
DROP = {"pitcher_id", "batter_id"}


def bss(y, p):
    r = y.mean()
    return float(max(0.0, 100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r)))))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    import sys
    r_only = "--r-only" in sys.argv
    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != ID and c not in DROP]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=features + [TARGET])
    if r_only:
        # F리그 체제 변화(2022→2023)가 이 백테스트 구간을 오염 → R만으로 체인 검증
        df = df[df.game_type == "R"].reset_index(drop=True)
        print(f"R only: {len(df)}행")

    tr_mask = df.season <= 2022
    cal_mask = df.season == 2023
    te_mask = df.season == 2024

    priors = compute_priors(df[tr_mask])
    df, new_cols = add_features(df, priors)
    feats = features + new_cols
    cat_cols = CAT_COLS + [c for c in NEW_CAT if c in new_cols]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # 주의: 2023 검증으로 early stopping 하면 F리그 체제 변화(2022→2023) 때문에
    # iter 1에서 멈춤 → 본 모델과 유사한 고정 iteration 사용
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        task_type="GPU", devices="0", loss_function="Logloss",
        random_seed=42, verbose=0)
    model.fit(Pool(df.loc[tr_mask, feats], df.loc[tr_mask, TARGET],
                   cat_features=cat_cols))

    p_cal = model.predict_proba(df.loc[cal_mask, feats])[:, 1]
    p_te = model.predict_proba(df.loc[te_mask, feats])[:, 1]
    y_cal = df.loc[cal_mask, TARGET].to_numpy(np.float64)
    y_te = df.loc[te_mask, TARGET].to_numpy(np.float64)

    print(f"[학습 2019-22, 보정 2023, 테스트 2024] best_iter={model.get_best_iteration()}")
    print(f"2023 예측평균 {p_cal.mean():.4f} (실제 {y_cal.mean():.4f}) "
          f"| 2024 예측평균 {p_te.mean():.4f} (실제 {y_te.mean():.4f})")
    print(f"\n① 무보정            2024 BSS {bss(y_te, p_te):8.2f}")

    # ② 전역 δ: 2019~2023 실제 성공률 추세 → 2024 r_hat. δ는 2023 예측평균 기준
    rates = df[df.season <= 2023].groupby("season")[TARGET].mean()
    r_hat = float(np.polyval(np.polyfit(rates.index, rates.values, 1), 2024))
    delta = float(logit(np.array([r_hat])) - logit(np.array([p_cal.mean()])))
    p2 = sigmoid(logit(p_te) + delta)
    print(f"② 전역 δ({delta:+.4f})   2024 BSS {bss(y_te, p2):8.2f}  (r_hat={r_hat:.4f})")

    # ③ isotonic (2023 적합)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    p3 = iso.predict(p_te)
    print(f"③ isotonic          2024 BSS {bss(y_te, p3):8.2f}")

    # ④ isotonic + 전역 δ (isotonic 후 2023 수준 → r_hat 시프트)
    p3c = iso.predict(p_cal)
    delta4 = float(logit(np.array([r_hat])) - logit(np.array([p3c.mean()])))
    p4 = sigmoid(logit(p3) + delta4)
    print(f"④ isotonic+δ        2024 BSS {bss(y_te, p4):8.2f}")

    # ⑤ 월별 δ_m: 2023 월별 (실제율 − 예측평균)을 로짓 시프트로 + 연간 추세 하락분
    cal_df = pd.DataFrame({"m": df.loc[cal_mask, "game_month"].values,
                           "y": y_cal, "p": p_cal})
    yearly_drop = delta  # 전역 추세 시프트
    dm = {}
    for m, g in cal_df.groupby("m"):
        if len(g) < 3000:
            dm[m] = yearly_drop
        else:
            dm[m] = float(logit(np.array([g.y.mean()]))
                          - logit(np.array([g.p.mean()]))) + yearly_drop
    te_m = df.loc[te_mask, "game_month"].to_numpy()
    shift = np.array([dm.get(m, yearly_drop) for m in te_m])
    p5 = sigmoid(logit(p_te) + shift)
    print(f"⑤ 월별 δ_m + 추세    2024 BSS {bss(y_te, p5):8.2f}")
    print("   월별 δ:", {k: round(v, 3) for k, v in sorted(dm.items())})


if __name__ == "__main__":
    main()

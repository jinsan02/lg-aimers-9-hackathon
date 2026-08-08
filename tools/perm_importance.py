"""Permutation importance — 2024 홀드아웃에서 직접 측정.

E47 교훈: 주변부 상관은 GBDT 피처 유용성의 지표가 아니다.
→ 검증셋에서 컬럼을 섞었을 때 BSS가 얼마나 떨어지는지로 판단한다.
음수(섞으면 오히려 좋아짐) 피처가 제거 후보.

실행(4070): python -m uv run python tools/perm_importance.py
"""

import joblib
import numpy as np
import pandas as pd

from features import add_features

DATA, TARGET = "./data", "control_success"
N_SAMPLE = 120_000
SEED = 0


def bss(y, p):
    r = y.mean()
    return float(100000 * (1 - ((p - y) ** 2).mean() / (r * (1 - r))))


def main():
    pack = joblib.load("model/cat_fv2.pkl")
    feats, model, cats = pack["features"], pack["model"], pack["cat_cols"]

    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw = [c for c in test_cols if c != "row_id"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=raw + [TARGET])
    df = df[df.season == 2024].reset_index(drop=True)
    df, _ = add_features(df, pack["priors"])
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=min(N_SAMPLE, len(df)), replace=False)
    df = df.iloc[idx].reset_index(drop=True)

    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype(str)
    y = df[TARGET].to_numpy(np.float64)
    base = bss(y, model.predict_proba(X)[:, 1])
    print(f"기준 BSS(2024 표본 {len(df)}) = {base:.2f}\n")

    # 원본(파생 이전) 컬럼 단위로 섞어야 의미 있음 → 파생은 원본과 함께 섞음
    groups = {}
    for c in feats:
        root = c.replace("_shr", "").replace("_rel", "")
        groups.setdefault(root if root in raw else c, []).append(c)

    rows = []
    for root, cols in groups.items():
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        for c in cols:
            Xp[c] = Xp[c].to_numpy()[perm]
        s = bss(y, model.predict_proba(Xp)[:, 1])
        rows.append({"feature": root, "n_cols": len(cols), "bss_after": s,
                     "drop": base - s})
    r = pd.DataFrame(rows).sort_values("drop", ascending=False)
    print("=== permutation importance (drop 클수록 중요) ===")
    print(r.round(2).to_string(index=False))
    neg = r[r["drop"] <= 0]
    print(f"\n제거 후보 (섞어도 안 나빠짐): {neg.feature.tolist()}")
    r.to_csv("out/perm_importance.csv", index=False)


if __name__ == "__main__":
    main()

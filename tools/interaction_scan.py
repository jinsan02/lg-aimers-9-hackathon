"""피처 조합 후보 발굴 — 데이터에서 직접 근거를 찾는다.

1) CatBoost 내부 상호작용 중요도 (실제로 트리가 함께 쓰는 쌍)
2) 연속형 분포 특성 (왜도/유계/영과잉) → 변환 후보
3) 상위 상호작용 쌍의 조건부 성공률 표 → 파생 피처 설계 근거

실행(4070): python -m uv run python tools/interaction_scan.py
"""

import joblib
import numpy as np
import pandas as pd

from features import add_features

pd.set_option("display.width", 200)

pack = joblib.load("model/cat_fv2.pkl")
feats, model = pack["features"], pack["model"]

print("=== 1. CatBoost 상호작용 중요도 상위 25쌍 ===")
inter = model.get_feature_importance(type="Interaction")
names = model.feature_names_
rows = [{"a": names[int(i)], "b": names[int(j)], "score": float(s)}
        for i, j, s in inter[:25]]
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 2. 연속형 분포 특성 (변환 후보) ===")
df = pd.read_csv("data/train.csv", encoding="utf-8-sig", nrows=400_000)
df, _ = add_features(df, pack["priors"])
num = [c for c in feats if df[c].dtype != object and df[c].nunique() > 32]
prof = []
for c in num:
    s = df[c].dropna()
    prof.append({"col": c, "skew": float(s.skew()), "zero_frac": float((s == 0).mean()),
                 "min": float(s.min()), "max": float(s.max()),
                 "p99/p50": float(s.quantile(.99) / max(s.quantile(.5), 1e-9))})
p = pd.DataFrame(prof)
print(p.reindex(p.skew.abs().sort_values(ascending=False).index).head(12)
      .round(3).to_string(index=False))

print("\n=== 3. 상위 상호작용 쌍의 조건부 성공률 ===")
y = "control_success"
for a, b in [("balls_before", "strikes_before"),
             ("pitcher_hand", "batter_hand"),
             ("game_type", "season"),
             ("inning", "outs_before")]:
    if a in df.columns and b in df.columns:
        t = df.pivot_table(index=a, columns=b, values=y, aggfunc="mean")
        print(f"\n[{a} × {b}]")
        print(t.round(4).to_string())

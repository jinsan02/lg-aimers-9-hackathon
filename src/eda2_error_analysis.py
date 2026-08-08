"""상세 EDA v2 — 현 최고 모델(fv2 앙상블)의 오차 해부.

어디서 점수가 새는지 세그먼트 단위로 정량화한다:
  세그먼트별 (예측평균 - 실제성공률) 오차와, 그 오차가 Brier에 기여하는 점수 손실.
  손실(BSS points) ≈ n/N × (bias)² / (r(1-r)) × 100000

실행: PYTHONPATH=src python src/eda2_error_analysis.py
"""

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

d = np.load("out/cat_fv2ens_val_preds.npz")
y, p = d["y"], d["pred"]
N = len(y)
BASE = y.mean() * (1 - y.mean())


def seg_report(df, col, min_n=2000):
    g = df.groupby(col, observed=True).agg(
        n=("y", "size"), actual=("y", "mean"), pred=("p", "mean"))
    g = g[g.n >= min_n]
    g["bias"] = g["pred"] - g["actual"]
    g["loss_pts"] = (g.n / N) * (g.bias ** 2) / BASE * 100000
    return g.sort_values("loss_pts", ascending=False)


train = pd.read_csv("data/train.csv", encoding="utf-8-sig")
val = train[train.season == 2024].reset_index(drop=True)
assert len(val) == N
val = val.assign(y=y, p=p)

print(f"전체: 실제 r={y.mean():.4f} | 예측평균={p.mean():.4f} "
      f"| 전역 bias 손실 {((p.mean()-y.mean())**2)/BASE*100000:.1f}점")

# ---- 세그먼트별 캘리브레이션 오차 ----
val["month_b"] = val.game_month
val["li_q"] = pd.qcut(val.li, 5, duplicates="drop")
val["pn_b"] = pd.cut(val.asof_pitcher_n, [-1, 50, 200, 1000, 5000, 99999],
                     labels=["<50", "50-200", "200-1k", "1k-5k", "5k+"])
val["count_state"] = (val.balls_before.astype(str) + "-"
                      + val.strikes_before.astype(str))
val["wexp_q"] = pd.qcut(val.home_win_expectancy, 5, duplicates="drop")
val["psr_q"] = pd.qcut(val.asof_pitcher_success_rate, 10, duplicates="drop")

for col in ["game_type", "month_b", "pn_b", "count_state", "li_q", "inning",
            "top_bottom", "pitcher_hand", "batter_hand", "wexp_q",
            "game_dayofweek", "pitcher_team_id", "psr_q"]:
    g = seg_report(val, col)
    tot = g.loss_pts.sum()
    print(f"\n### {col} (세그먼트 bias 손실 합 {tot:.1f}점)")
    print(g.head(8).round(4).to_string())

# ---- 판별력 여지: 예측 분위별 실제 성공률 (sharpness) ----
print("\n### 예측 분위(10)별 실제 vs 예측 — 판별력/캘리브레이션 곡선")
val["p_q"] = pd.qcut(val.p, 10)
print(val.groupby("p_q", observed=True).agg(
    n=("y", "size"), actual=("y", "mean"), pred=("p", "mean")).round(4).to_string())

# ---- 아직 안 쓰는 신호 후보: 잔차와 피처의 상관 ----
val["resid"] = val.y - val.p
num_cols = [c for c in val.columns if val[c].dtype in (np.float64, np.int64)
            and c not in ("y", "p", "resid", "row_id")]
corr = val[num_cols + ["resid"]].corr(numeric_only=True)["resid"].drop("resid")
print("\n### 잔차와 상관 높은 피처 상위 12 (모델이 못 쓴 신호 후보)")
print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(12).round(4).to_string())

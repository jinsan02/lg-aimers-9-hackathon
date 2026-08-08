"""오차 분석 - 현재 최고 블렌드가 실제와 어디서 가장 크게 어긋나는가.

Brier는 세그먼트별로 분해된다:
    brier = E[(p-y)^2] = E[(p - r_seg)^2] + Var_seg
  앞항이 **편향(bias)**, 뒤가 그 세그먼트의 고유 불확실성이다.
  개선 여지는 편향에만 있다 - 뒤는 물리적으로 못 줄인다.

그래서 세그먼트별로
  - 실제 성공률 r_seg 대 예측 평균 p_seg  -> 편향
  - 그 편향이 전체 Brier에서 차지하는 몫  -> **고치면 몇 점 오르는가**
를 계산해 개선 우선순위를 매긴다.
"""

import glob
import sys

import numpy as np
import pandas as pd

DATA, T = "./data", "control_success"


def bss_of(brier, base):
    return 100000 * (1 - brier / base)


def main():
    from script_blend_v5 import WEIGHTS
    P, W = [], []
    y = None
    for path, w in WEIGHTS:
        tag = path.split("/")[-1].replace(".pkl", "").split("_", 1)[1]
        hits = glob.glob(f"./out/*_{tag}_val_preds.npz")
        if not hits:
            continue
        z = np.load(hits[0])
        y = z["y"] if y is None else y
        P.append(z["pred"])
        W.append(w)
    W = np.array(W, float)
    W /= W.sum()
    p = W @ np.vstack(P)

    cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    use = [c for c in cols if c != "row_id"]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=list(dict.fromkeys(use + [T])))
    v = df[df.season == 2024].reset_index(drop=True)
    assert len(v) == len(y), f"행수 불일치 {len(v)} vs {len(y)}"
    assert np.allclose(v[T].to_numpy(), y), "타깃 불일치 - 순서가 어긋났다"
    v["pred"] = p
    r = y.mean()
    base = r * (1 - r)
    brier = ((p - y) ** 2).mean()
    print(f"검증 {len(v):,}행 | 실제 {r:.4f} 예측평균 {p.mean():.4f} "
          f"| Brier {brier:.6f} | BSS {bss_of(brier, base):.2f}\n")

    print("=== 캘리브레이션 (예측 10분위) ===")
    v["bin"] = pd.qcut(v.pred, 10, labels=False, duplicates="drop")
    g = v.groupby("bin").agg(n=("pred", "size"), 예측=("pred", "mean"),
                             실제=(T, "mean"))
    g["편향"] = g["예측"] - g["실제"]
    # 그 구간 편향을 완전히 없애면 Brier가 얼마나 줄어드는가 = 편향^2 * 비중
    g["개선여지"] = (g["편향"] ** 2 * g["n"] / len(v)) / base * 100000
    print(g.round({"예측": 4, "실제": 4, "편향": 4, "개선여지": 1}).to_string())
    print(f"  캘리브레이션 완전보정 시 총 개선여지: {g['개선여지'].sum():.1f} BSS\n")

    print("=== 세그먼트별 편향과 개선여지 (큰 순) ===")
    v["exp_bucket"] = pd.cut(v.asof_pitcher_n.fillna(0),
                             [-1, 0, 50, 200, 1000, 5000, 1e9],
                             labels=["없음", "1-50", "51-200", "201-1k",
                                     "1k-5k", "5k+"])
    v["cnt"] = v.balls_before.astype(str) + "-" + v.strikes_before.astype(str)
    v["inn"] = v.inning.clip(1, 9)
    segs = ["exp_bucket", "cnt", "game_type", "inn", "game_month",
            "base_state", "pitcher_hand", "batter_hand", "top_bottom"]
    out = []
    for s in segs:
        g = v.groupby(s, observed=True).agg(n=("pred", "size"),
                                            예측=("pred", "mean"), 실제=(T, "mean"))
        g["편향"] = g["예측"] - g["실제"]
        g["개선여지"] = (g["편향"] ** 2 * g["n"] / len(v)) / base * 100000
        out.append((g["개선여지"].sum(), s, g))
    for tot, s, g in sorted(out, reverse=True):
        print(f"\n--- {s} (이 축 전체 개선여지 {tot:.1f} BSS) ---")
        gg = g.sort_values("개선여지", ascending=False).head(5)
        print(gg.round({"예측": 4, "실제": 4, "편향": 4, "개선여지": 1}).to_string())


if __name__ == "__main__":
    sys.exit(main())

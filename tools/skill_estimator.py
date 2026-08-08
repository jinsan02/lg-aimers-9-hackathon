"""투수 실력 추정의 여지가 있는가 — 손으로 정한 수축 vs 학습된 추정기.

착안(사용자 제안): 선수마다 시계열로 보고 회귀로 접근한다.
2025 행에서는 시퀀스를 복원할 수 없지만(다른 test 행 참조 = 규정 위반),
**한 행 안에서 합법적으로 보이는 3점 스냅샷**은 있다.
    통산      : asof_pitcher_n, asof_pitcher_success_rate
    당해 시즌 : E99 차분 (std_pitcher_n, std_asof_pitcher_success_rate)
    최근 경기 : asof_pitcher_prev1/3/5_game_success_rate

지금은 이 셋을 k=80 상수 수축으로 뭉개고 있다. 이걸 **학습**시키면 학습된
칼만 필터가 된다. 그리고 이진 타깃(신호가 분산의 0.9%)보다 '앞으로의 성공률'이
훨씬 매끄러운 학습 문제다.

이 스크립트는 **짓기 전에 여지를 잰다**.
  타깃 : 그 투수의 **남은 시즌** 성공률 (그 행 이후 실제로 어떻게 던졌나)
  비교 : ① 통산 rate  ② 우리 std(k=80)  ③ 학습된 선형결합  ④ 학습된 GBDT
  학습은 2019~2023, 평가는 2024. ③④가 ②보다 확실히 나으면 지을 가치가 있다.

메모리: 4070/A100 에서 돌릴 것.
실행: python tools/skill_estimator.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
MINF = 150          # 남은 시즌 최소 투구수 (타깃이 안정적이어야 비교가 의미있다)


def main():
    sys.path.insert(0, "src")
    use = ["season", "game_type", "pitcher_id", "row_id", TARGET,
           "asof_pitcher_n", "asof_pitcher_success_rate",
           "asof_pitcher_prev1_game_success_rate",
           "asof_pitcher_prev3_game_success_rate",
           "asof_pitcher_prev5_game_success_rate",
           "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
           "asof_batter_success_rate"]
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    tr = tr[tr.game_type == "R"].reset_index(drop=True)
    print(f"R리그 {len(tr):,}행")

    import season_std as ss
    anchors = ss.build_anchors(tr)
    pri = {c: float(tr[c].mean()) for c in tr.columns if c.startswith("asof_")}
    sp = ss.season_priors(tr)
    tr, _ = ss.add_std(tr, anchors, k=80.0, priors=pri, to_career=False,
                       season_prior=sp)

    # ---- 타깃: 그 행 **이후** 남은 시즌 성공률 (투수-시즌 내에서 역누적)
    tr = tr.sort_values(["pitcher_id", "season", "row_id"])
    g = tr.groupby(["pitcher_id", "season"], sort=False)
    y = tr[TARGET].to_numpy(np.float64)
    tot = g[TARGET].transform("sum").to_numpy()
    cnt = g[TARGET].transform("size").to_numpy()
    idx = g.cumcount().to_numpy()
    fut_n = cnt - idx - 1
    fut_s = tot - np.cumsum(y) + np.concatenate([[0], np.cumsum(y)[:-1]])
    # 위 식은 그룹 경계를 안 지키므로 그룹별로 다시 계산한다
    cs = g[TARGET].cumsum().to_numpy()
    fut_s = tot - cs
    fut_rate = np.where(fut_n > 0, fut_s / np.maximum(fut_n, 1), np.nan)
    tr["fut_rate"] = fut_rate
    tr["fut_n"] = fut_n
    d = tr[tr.fut_n >= MINF].copy()
    print(f"남은 투구 {MINF}+ 인 행 {len(d):,} "
          f"({len(d) / len(tr) * 100:.1f}%)")

    FE = ["asof_pitcher_success_rate", "asof_pitcher_n",
          "std_asof_pitcher_success_rate", "std_pitcher_n",
          "asof_pitcher_prev1_game_success_rate",
          "asof_pitcher_prev3_game_success_rate",
          "asof_pitcher_prev5_game_success_rate",
          "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
          "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "season"]
    trn = d[d.season <= 2023]
    val = d[d.season == 2024]
    yv = val["fut_rate"].to_numpy(np.float64)
    print(f"학습 {len(trn):,} | 검증(2024) {len(val):,} | "
          f"타깃 평균 {yv.mean():.4f} 표준편차 {yv.std():.4f}\n")

    def mse(p):
        return float(((p - yv) ** 2).mean())

    base = mse(np.full(len(yv), trn["fut_rate"].mean()))
    print(f"{'추정기':<40}{'MSE':>10}{'설명력':>9}")
    print(f"{'상수 (학습 평균)':<40}{base:>10.5f}{0.0:>9.1%}")
    for nm, col in [("통산 rate (asof)", "asof_pitcher_success_rate"),
                    ("우리 std (k=80 손으로 정함)",
                     "std_asof_pitcher_success_rate")]:
        p = val[col].fillna(trn[col].mean()).to_numpy(np.float64)
        print(f"{nm:<40}{mse(p):>10.5f}{1 - mse(p) / base:>9.1%}")

    # ③ 학습된 선형결합 (같은 입력, 가중치만 학습)
    Xt = trn[FE].fillna(trn[FE].median()).to_numpy(np.float64)
    Xv = val[FE].fillna(trn[FE].median()).to_numpy(np.float64)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    Xt = np.column_stack([np.ones(len(Xt)), Xt])
    Xv = np.column_stack([np.ones(len(Xv)), Xv])
    beta, *_ = np.linalg.lstsq(Xt, trn["fut_rate"].to_numpy(np.float64),
                               rcond=None)
    pl = Xv @ beta
    print(f"{'학습된 선형결합':<40}{mse(pl):>10.5f}{1 - mse(pl) / base:>9.1%}")

    # ④ 학습된 GBDT (비선형 결합까지)
    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                              loss_function="RMSE", verbose=0, random_seed=42)
        m.fit(trn[FE].to_numpy(np.float64),
              trn["fut_rate"].to_numpy(np.float64))
        pg = m.predict(val[FE].to_numpy(np.float64))
        print(f"{'학습된 GBDT':<40}{mse(pg):>10.5f}{1 - mse(pg) / base:>9.1%}")
    except Exception as e:                                   # noqa: BLE001
        print("  GBDT 생략:", e)
        pg = None

    # (5) varying-coefficient (경험 베이즈 형태)
    #    참 명세는 실력 = w(n)*시즌rate + (1-w(n))*통산rate, w(n)=n/(n+k) 다.
    #    즉 **계수가 표본수의 함수**다. 선형은 그 근사, GBDT 는 계단 근사다.
    #    x_j * n/(n+k) 를 여러 k 로 만들어 기저를 넓히면 최소제곱으로 풀 수 있다.
    RATE = ["asof_pitcher_success_rate", "std_asof_pitcher_success_rate",
            "asof_pitcher_prev1_game_success_rate",
            "asof_pitcher_prev3_game_success_rate",
            "asof_pitcher_prev5_game_success_rate"]
    OTH = ["asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
           "asof_batter_success_rate"]
    KS = (10.0, 40.0, 160.0, 640.0)
    medv = {c: float(trn[c].median()) for c in RATE + OTH}

    def vc(dd):
        n = dd["std_pitcher_n"].fillna(0).to_numpy(np.float64)
        nc = dd["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
        parts = [np.ones(len(dd))]
        for c in RATE:
            v = dd[c].fillna(medv[c]).to_numpy(np.float64)
            parts.append(v)
            for k in KS:
                parts.append(v * (n / (n + k)))
        for k in KS:
            parts.append(n / (n + k))
            parts.append(nc / (nc + k * 10))
        for c in OTH:
            parts.append(dd[c].fillna(medv[c]).to_numpy(np.float64))
        return np.column_stack(parts)

    Zt, Zv = vc(trn), vc(val)
    lam = 1.0
    A = Zt.T @ Zt + lam * np.eye(Zt.shape[1])
    A[0, 0] -= lam
    bb = np.linalg.solve(A, Zt.T @ trn["fut_rate"].to_numpy(np.float64))
    pv = Zv @ bb
    print(f"{'varying-coefficient (경험베이즈 형태)':<40}{mse(pv):>10.5f}"
          f"{1 - mse(pv) / base:>9.1%}")

    print("\n읽는 법: 학습된 추정기가 우리 std 보다 설명력이 확실히 높으면,")
    print("  '실력 추정'을 별도 회귀로 만들어 그 예측을 피처로 넣을 가치가 있다.")
    print("  비슷하면 k=80 수축이 이미 최적에 가깝다는 뜻이므로 짓지 말 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

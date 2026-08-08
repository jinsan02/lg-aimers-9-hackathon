"""ABS 충격이 투수마다 다른가? (E105)

지금의 SHIFT는 전 구간 균일 상수다. 근거는 "편향이 std_n 구간별로 균일"이었다.
그런데 그건 **표본수** 축이고, ABS 는 표본수가 아니라 **투구 스타일** 축으로
차등 충격을 준다. 심판이 기계로 바뀌면 손해는 존 경계에 사는 투수가 본다.

측정 설계:
  R리그 ABS 도입 = 2024. 따라서 2023 -> 2024 변화가 곧 ABS 도입 충격이다.
  투수별로 delta = rate2024 - rate2023 를 만들고, **2023 시점 특성**으로 회귀한다.
  (2023 특성만 쓰므로 예측 시점에 알 수 있는 정보다 - 2025 에 그대로 적용 가능)

  F리그는 2023 도입이므로 2022->2023 이 도입 충격, 2023->2024 가 2년차 충격.
  2025 R리그는 **2년차**이므로 F리그 2년차 계수가 직접 아날로그다.

읽는 법:
  계수가 유의하면 -> 균일 SHIFT 대신 특성 가중 SHIFT 를 쓸 근거가 된다.
  계수가 0에 가까우면 -> 균일 SHIFT 가 옳다는 확인이 되고 이 축은 닫는다.

실행: python tools/abs_hetero.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
MINP = 300          # 투수-시즌 최소 투구수 (양쪽 시즌 모두)

# 그 투수가 존 경계에 얼마나 의존하는가를 나타낼 후보 특성 (전부 asof = 사전 관측)
CH = ["asof_pitcher_success_rate", "asof_pitcher_ball_rate",
      "asof_pitcher_strike_rate", "asof_pitcher_middle_rate",
      "asof_pitcher_reverse_rate"]


def season_rates(df, league, seasons):
    """(투수, 시즌) 별 실제 성공률 + 그 시즌 시작 시점 특성."""
    d = df[(df.game_type == league) & (df.season.isin(seasons))]
    g = d.groupby(["pitcher_id", "season"])
    out = g.agg(n=("control_success", "size"),
                rate=("control_success", "mean")).reset_index()
    # 특성은 그 시즌 **첫 투구 시점**의 asof (= 직전까지 통산) 로 잡는다
    first = d.sort_values("row_id").groupby(["pitcher_id", "season"]).head(1)
    out = out.merge(first[["pitcher_id", "season"] + CH],
                    on=["pitcher_id", "season"], how="left")
    return out[out.n >= MINP]


def transition(rates, y0, y1, label):
    a = rates[rates.season == y0]
    b = rates[rates.season == y1][["pitcher_id", "rate", "n"]]
    m = a.merge(b, on="pitcher_id", suffixes=("0", "1"))
    if len(m) < 25:
        print(f"\n[{label}] 표본 {len(m)}명 - 부족, 생략")
        return
    m = m.dropna(subset=CH)
    d = (m["rate1"] - m["rate0"]).to_numpy(np.float64)
    w = np.minimum(m["n0"], m["n1"]).to_numpy(np.float64)
    wm = float((d * w).sum() / w.sum())
    print(f"\n[{label}]  투수 {len(m)}명 | 평균 delta {d.mean():+.4f} | "
          f"투구수가중 {wm:+.4f} | 표준편차 {d.std():.4f}")

    # 특성별 단변량: 상위/하위 3분위 delta 차이 + 상관
    print(f"    {'특성':<34}{'상관':>7}{'하위1/3':>10}{'상위1/3':>10}{'격차':>9}")
    for c in CH:
        x = m[c].to_numpy(np.float64)
        if np.nanstd(x) < 1e-9:
            continue
        r = float(np.corrcoef(x, d)[0, 1])
        lo = np.nanpercentile(x, 33.3)
        hi = np.nanpercentile(x, 66.7)
        dl = d[x <= lo].mean()
        dh = d[x >= hi].mean()
        star = " *" if abs(r) > 2.0 / np.sqrt(len(m)) else ""
        print(f"    {c:<34}{r:>7.3f}{dl:>10.4f}{dh:>10.4f}"
              f"{dh - dl:>+9.4f}{star}")

    # 다변량 최소제곱 (특성을 표준화해 계수 크기를 비교 가능하게)
    X = m[CH].to_numpy(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, d, rcond=None)
    res = d - X @ beta
    dof = max(len(d) - X.shape[1], 1)
    cov = (res @ res / dof) * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (res @ res) / (((d - d.mean()) ** 2).sum() + 1e-12)
    print(f"    -- 다변량 (표준화) R2={r2:.3f}")
    for nm, bt, s in zip(["절편"] + CH, beta, se):
        t = bt / s if s > 0 else 0.0
        mark = " <<" if abs(t) >= 2.0 else ""
        print(f"      {nm:<34}{bt:>+9.4f}  t={t:>6.2f}{mark}")


def main():
    use = ["row_id", "season", "game_type", "pitcher_id",
           "control_success"] + CH
    df = pd.read_csv(f"{DATA}/train.csv", usecols=use)
    print(f"train {len(df):,}행 | game_type {sorted(df.game_type.unique())}")

    for lg, name in [("R", "R리그(1군)"), ("F", "F리그(퓨처스)")]:
        rates = season_rates(df, lg, [2021, 2022, 2023, 2024])
        by = rates.groupby("season").apply(
            lambda s: np.average(s["rate"], weights=s["n"]))
        print(f"\n=== {name} 시즌 성공률 (투수 {MINP}구 이상) ===")
        print("   " + "  ".join(f"{int(s)} {v:.4f}" for s, v in by.items()))
        if lg == "R":
            transition(rates, 2022, 2023, f"{name} 2022->2023  ABS 이전(대조군)")
            transition(rates, 2023, 2024, f"{name} 2023->2024  **ABS 도입**")
        else:
            transition(rates, 2022, 2023, f"{name} 2022->2023  **ABS 도입**")
            transition(rates, 2023, 2024,
                       f"{name} 2023->2024  **ABS 2년차** (2025 R리그 아날로그)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

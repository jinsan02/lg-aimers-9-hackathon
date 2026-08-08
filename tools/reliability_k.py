"""수축 상수 k 를 **격자 탐색이 아니라 구성적으로** 구한다 (E161).

## 왜

std_k 80 -> 40 이 미학습 표면에서 +15.05 (t=11.88) 였다. 80 은 자기검증 표면에서
격자로 고른 값이다. 우리 파이프라인의 k 는 전부 같은 방식으로 골랐다:
  std_k 80 / te_k 50 / feat_k 50 / prof_k 60
하나가 15점이나 틀려 있었다면 나머지도 봐야 한다. 그런데 축마다 격자를 돌리면
GPU 시간이 든다 — 사베매트릭스에는 **데이터에서 바로 계산하는 방법**이 있다.

## 방법 (Tango 의 평균회귀 상수)

관측 분산은 진짜 실력 분산 + 표본 잡음이다:

    var_obs = var_true + E[ p(1-p) / n ]
    var_true = var_obs - E[ p(1-p) / n ]
    k = p_bar (1 - p_bar) / var_true

k 는 "사전확률을 실측치와 같은 무게로 두려면 몇 개의 관측이 필요한가"다.
n = k 에서 수축이 정확히 절반이 된다.

교차확인으로 **반분 상관**도 낸다. 그룹 안의 투구를 홀/짝으로 나눠 두 반쪽의
비율을 상관시키면, Spearman-Brown 으로 전체 신뢰도를 되돌릴 수 있고
신뢰도가 0.5 가 되는 n 이 곧 k 다. 두 추정이 크게 다르면 어느 쪽도 믿지 않는다.

## 무엇에 맞추는가

우리 std_k 는 **시즌내 복원치**(E99)에 걸리는 수축이므로 '한 시즌 안에서 그 투수의
성공률'을 추정하는 문제다. 그래서 시즌별로 계산한다. te_k 는 TE 그룹 단위다.

실행: python tools/reliability_k.py
"""

import sys

import numpy as np
import pandas as pd

MIN_N = 20          # 이보다 적은 그룹은 잡음이 지배해 추정을 흔든다


def moments_k(n, s):
    """적률법. n=그룹별 관측수, s=성공수."""
    p = s / n
    pbar = float(s.sum() / n.sum())
    # 그룹을 동일 가중으로 본다 (n 가중은 큰 그룹에 끌려간다)
    var_obs = float(np.var(p, ddof=1))
    noise = float(np.mean(pbar * (1 - pbar) / n))
    var_true = var_obs - noise
    if var_true <= 0:
        return float("inf"), var_true, var_obs, pbar
    return pbar * (1 - pbar) / var_true, var_true, var_obs, pbar


def splithalf_k(df, keys, rng):
    """홀/짝 분할 상관 → Spearman-Brown → 신뢰도 0.5 가 되는 n."""
    d = df.copy()
    d["_h"] = rng.integers(0, 2, len(d))
    g = d.groupby(keys + ["_h"]).control_success.agg(["sum", "size"])
    g = g.unstack("_h").dropna()
    if len(g) < 50:
        return float("nan"), 0
    n0, n1 = g[("size", 0)].to_numpy(), g[("size", 1)].to_numpy()
    ok = (n0 >= MIN_N / 2) & (n1 >= MIN_N / 2)
    if ok.sum() < 50:
        return float("nan"), int(ok.sum())
    p0 = (g[("sum", 0)].to_numpy() / n0)[ok]
    p1 = (g[("sum", 1)].to_numpy() / n1)[ok]
    r_half = float(np.corrcoef(p0, p1)[0, 1])
    if r_half <= 0:
        return float("inf"), int(ok.sum())
    # 반쪽 평균 표본수에서의 신뢰도가 r_half. 신뢰도 0.5 가 되는 n:
    #   rel(n) = n / (n + k)  →  k = n_half (1 - r_half) / r_half
    n_half = float(np.mean((n0[ok] + n1[ok]) / 2))
    return n_half * (1 - r_half) / r_half, int(ok.sum())


def main():
    cols = ["season", "game_type", "pitcher_id", "batter_id",
            "control_success", "balls_before", "strikes_before",
            "batter_hand", "pitcher_hand"]
    df = pd.read_csv("./data/train.csv", usecols=cols)
    df = df[df.game_type == "R"]
    rng = np.random.default_rng(0)

    AXES = {
        "투수 (std_k / feat_k 대상)": ["pitcher_id"],
        "투수×볼카운트 (te_k 'pc')": ["pitcher_id", "balls_before", "strikes_before"],
        "투수×타자손 (te_k 'ph')": ["pitcher_id", "batter_hand"],
        "타자 (te_k 'b')": ["batter_id"],
        "투수×이닝 (참고)": ["pitcher_id", "inning"] if "inning" in df else None,
    }
    print(f"정규리그 {len(df):,}행 | 시즌별로 계산 후 중앙값 보고 "
          f"(그룹 최소 {MIN_N}관측)\n")
    print(f"{'축':<28}{'그룹수':>8}{'평균n':>8}{'적률법 k':>11}"
          f"{'반분법 k':>11}{'현행':>7}")
    current = {"투수 (std_k / feat_k 대상)": "80",
               "투수×볼카운트 (te_k 'pc')": "50",
               "투수×타자손 (te_k 'ph')": "50",
               "타자 (te_k 'b')": "50"}

    for name, keys in AXES.items():
        if keys is None:
            continue
        ks_m, ks_s, gs, ns = [], [], [], []
        for _, g in df.groupby("season"):
            agg = g.groupby(keys).control_success.agg(["sum", "size"])
            agg = agg[agg["size"] >= MIN_N]
            if len(agg) < 50:
                continue
            k, _vt, _vo, _p = moments_k(agg["size"].to_numpy(),
                                        agg["sum"].to_numpy())
            ks_m.append(k)
            ks_s.append(splithalf_k(g, keys, rng)[0])
            gs.append(len(agg))
            ns.append(agg["size"].mean())
        if not ks_m:
            continue
        print(f"{name:<28}{int(np.median(gs)):>8,}{np.median(ns):>8.0f}"
              f"{np.median(ks_m):>11.0f}{np.nanmedian(ks_s):>11.0f}"
              f"{current.get(name, '-'):>7}")

    print("\n※ 두 추정이 서로 가까울 때만 믿는다. 현행값과 크게 다르면 그 축은")
    print("   자기검증 표면 격자탐색이 잘못 고른 것 — std_k 가 80 -> 40 이었듯이.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

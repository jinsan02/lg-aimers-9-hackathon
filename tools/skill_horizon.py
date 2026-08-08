"""실력 추정기의 **앞으로 보는 창(forward horizon)** 최적점을 찾는다.

뒤로 보는 창은 우리가 못 정한다 - 2025 행에서 시퀀스를 못 만들기 때문에
대회가 준 4개 스냅샷(통산 / 당해시즌 / 최근5·3·1경기)이 전부고, 그 가중치는
회귀가 알아서 학습한다.

정할 수 있는 건 **타깃의 시야**뿐이다. 지금은 '남은 시즌 전체'인데,
시즌 초 행이면 앞으로 2,000구를 평균한 값이다. 우리가 예측하는 건 바로 다음
한 구이므로 명세가 어긋나 있다.

  N 작다 -> 타깃이 잡음투성이 (이진에 가까워진다)
  N 크다 -> 매끄럽지만 '다음 한 구'와 멀어진다

중간에 최적점이 있다. **평가는 항상 동일한 잣대로 한다** - 어떤 N 으로 학습했든
'앞으로 300구' 성공률을 맞히는 능력으로 비교해야 공정하다.

실행(4070/A100): python tools/skill_horizon.py
"""

import sys

import numpy as np
import pandas as pd

DATA = "./data"
TARGET = "control_success"
EVAL_H = 300          # 공통 평가 잣대: 앞으로 300구 성공률
HS = [100, 300, 1000, 3000, 0]     # 0 = 남은 시즌 전체 (현재 방식)


def fwd(df, h):
    """(투수, 시즌) 안에서 그 행 이후 h구의 성공률/개수. h=0 이면 남은 전체."""
    d = df.sort_values(["pitcher_id", "season"], kind="stable")
    g = d.groupby(["pitcher_id", "season"], sort=False)
    y = d[TARGET].to_numpy(np.float64)
    cs = g[TARGET].cumsum().to_numpy(np.float64)
    tot = g[TARGET].transform("sum").to_numpy(np.float64)
    cnt = g[TARGET].transform("size").to_numpy(np.float64)
    idx = g.cumcount().to_numpy(np.float64)
    rem = cnt - idx - 1
    if h <= 0:
        n = rem
        s = tot - cs
    else:
        # 앞으로 h구: 누적합을 h칸 앞으로 당겨 뺀다 (그룹 경계는 rem 으로 자름)
        n = np.minimum(rem, h)
        ahead = np.empty_like(cs)
        pos = np.arange(len(cs))
        tgt = np.minimum(pos + n.astype(np.int64), pos + rem.astype(np.int64))
        ahead = cs[tgt]
        s = ahead - cs
    r = np.where(n > 0, s / np.maximum(n, 1.0), np.nan)
    return (pd.Series(n, index=d.index).reindex(df.index),
            pd.Series(r, index=d.index).reindex(df.index))


def main():
    sys.path.insert(0, "src")
    use = ["season", "game_type", "pitcher_id", TARGET,
           "asof_pitcher_n", "asof_pitcher_success_rate",
           "asof_pitcher_prev1_game_success_rate",
           "asof_pitcher_prev3_game_success_rate",
           "asof_pitcher_prev5_game_success_rate",
           "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
           "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
           "asof_batter_success_rate"]
    tr = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig", usecols=use)
    tr = tr[tr.game_type == "R"].reset_index(drop=True)
    import season_std as ss
    anchors = ss.build_anchors(tr)
    pri = {c: float(tr[c].mean()) for c in tr.columns if c.startswith("asof_")}
    sp = ss.season_priors(tr)
    tr, _ = ss.add_std(tr, anchors, k=80.0, priors=pri, to_career=False,
                       season_prior=sp)
    print(f"R리그 {len(tr):,}행")

    import skill as sk
    med = {c: float(tr[c].median()) for c in sk.FEAT if c in tr.columns}

    # 공통 평가 잣대
    en, er = fwd(tr, EVAL_H)
    tr["_evn"], tr["_evr"] = en, er
    val_mask = (tr.season == 2024) & (tr._evn >= EVAL_H)
    trn_mask = tr.season <= 2023
    yv = tr.loc[val_mask, "_evr"].to_numpy(np.float64)
    Xv = sk._design(tr.loc[val_mask], med)
    print(f"평가: 2024 중 앞으로 {EVAL_H}구가 있는 {int(val_mask.sum()):,}행 "
          f"| 타깃 sd {yv.std():.4f}\n")

    def mse(p):
        return float(((p - yv) ** 2).mean())

    base = mse(np.full(len(yv), tr.loc[trn_mask, "_evr"].mean()))
    cur = tr.loc[val_mask, "std_asof_pitcher_success_rate"].to_numpy(np.float64)
    print(f"{'학습 타깃 시야':<32}{'MSE':>10}{'설명력':>9}")
    print(f"{'(참고) 우리 std k=80':<32}{mse(cur):>10.5f}"
          f"{1 - mse(cur) / base:>9.1%}")

    for h in HS:
        n, r = fwd(tr, h)
        m = trn_mask & (n >= max(50, min(h if h else 150, 150)))
        d = tr.loc[m]
        X = sk._design(d, med)
        y = r[m].to_numpy(np.float64)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        A = X.T @ X + 1.0 * np.eye(X.shape[1])
        A[0, 0] -= 1.0
        b = np.linalg.solve(A, X.T @ y)
        p = Xv @ b
        nm = "남은 시즌 전체 (현재)" if h == 0 else f"앞으로 {h}구"
        print(f"{nm:<32}{mse(p):>10.5f}{1 - mse(p) / base:>9.1%}"
              f"   학습행 {int(ok.sum()):,}")
    print("\n※ 어떤 N 으로 학습했든 평가는 '앞으로 300구'로 통일했다.")
    print("※ 설명력이 가장 높은 N 이 skill.py 의 타깃이 되어야 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

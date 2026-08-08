"""ABS·R/F 이동·신규/복귀가 현행 예측에 남긴 잔차를 감사한다.

평가 행 독립 규칙에 맞추기 위해 선수 상태는 항상 ``season < S`` 이력으로만 만든다.
2025 추론에서는 2019~2024 train으로 고정 lookup할 수 있는 피처들이다.
"""

from glob import glob
import sys

import numpy as np
import pandas as pd


T = "control_success"
COLS = [
    "row_id", "season", "game_month", "game_type", "top_bottom",
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate", T,
]


def prior_state(df, entity, prefix):
    """(entity, season)에 직전 등장 시즌/리그와 과거 리그별 행수를 붙인다."""
    g = (df.groupby([entity, "season", "game_type"], sort=False)
           .size().rename("n").reset_index())
    wide = (g.pivot_table(index=[entity, "season"], columns="game_type",
                          values="n", fill_value=0, aggfunc="sum")
              .reset_index())
    for c in ("R", "F"):
        if c not in wide:
            wide[c] = 0
    wide = wide.sort_values([entity, "season"])
    wide[f"{prefix}_prior_R_n"] = wide.groupby(entity)["R"].cumsum() - wide["R"]
    wide[f"{prefix}_prior_F_n"] = wide.groupby(entity)["F"].cumsum() - wide["F"]

    # 직전 시즌의 마지막 행 league. 같은 시즌 내부 행은 절대 보지 않는다.
    last = (df.sort_values("row_id", kind="stable")
              .groupby([entity, "season"], as_index=False).tail(1)
              [[entity, "season", "game_type"]]
              .sort_values([entity, "season"]))
    last[f"{prefix}_prev_season"] = last.groupby(entity)["season"].shift(1)
    last[f"{prefix}_prev_league"] = last.groupby(entity)["game_type"].shift(1)
    last = last[[entity, "season", f"{prefix}_prev_season",
                 f"{prefix}_prev_league"]]
    tab = wide.merge(last, on=[entity, "season"], how="left")
    return tab.drop(columns=["R", "F"])


def add_status(df, entity, prefix):
    tab = prior_state(df, entity, prefix)
    out = df.merge(tab, on=[entity, "season"], how="left", validate="many_to_one")
    pr = out[f"{prefix}_prior_R_n"].fillna(0)
    pf = out[f"{prefix}_prior_F_n"].fillna(0)
    prev = out[f"{prefix}_prev_league"]
    gap = out["season"] - out[f"{prefix}_prev_season"]
    cur = out["game_type"]
    unseen = (pr + pf) == 0
    status = np.full(len(out), "other", dtype=object)
    status[unseen] = "unseen"
    status[(~unseen) & (prev == cur) & (gap == 1)] = "same_cont"
    status[(~unseen) & (prev == cur) & (gap >= 2)] = "same_return"
    status[(~unseen) & (prev == "F") & (cur == "R")] = "F_to_R"
    status[(~unseen) & (prev == "R") & (cur == "F")] = "R_to_F"
    out[f"{prefix}_status"] = status
    out[f"{prefix}_gap"] = gap.fillna(-1)
    out[f"{prefix}_same_league_n"] = np.where(cur == "R", pr, pf)
    out[f"{prefix}_other_league_n"] = np.where(cur == "R", pf, pr)
    return out


def print_rates(df, key, min_n=500):
    z = (df.groupby(["season", "game_type", key], observed=True)
           .agg(n=(T, "size"), rate=(T, "mean")))
    base = df.groupby(["season", "game_type"])[T].mean().rename("base")
    z = z.join(base).assign(delta=lambda x: x.rate - x.base)
    print(f"\n=== {key}: 시즌·리그 대비 효과 (n>={min_n}) ===")
    print(z[z.n >= min_n].round(4).to_string())


def attach_pred_2024(df):
    paths = sorted(glob("./out/cat_VB2_base_s*_val_preds.npz"))
    d = df[df.season == 2024].copy()
    if not paths:
        print("\nVB2 예측 없음: 잔차 감사 생략")
        return None
    ps = []
    for p in paths:
        a = np.load(p)["pred"]
        if len(a) != len(d):
            raise ValueError(f"{p}: {len(a)} != 2024 rows {len(d)}")
        ps.append(a)
    d["pred"] = np.mean(ps, axis=0)
    d["resid"] = d[T] - d.pred
    d["sq"] = (d.pred - d[T]) ** 2
    print(f"\nVB2 2024 예측 {len(paths)}시드 결합 | n={len(d):,} | "
          f"bias={d.resid.mean():+.5f} mse={d.sq.mean():.6f}")
    return d


def print_residual(d, key, min_n=500):
    if d is None:
        return
    z = (d.groupby(["game_type", key], observed=True)
           .agg(n=(T, "size"), rate=(T, "mean"), pred=("pred", "mean"),
                bias=("resid", "mean"), mse=("sq", "mean")))
    print(f"\n=== VB2 2024 residual by {key} (n>={min_n}) ===")
    print(z[z.n >= min_n].round(6).to_string())


def print_failure(d):
    if d is None:
        return
    cols = [c for c in ("middle", "ball", "reverse") if c in d]
    z = d.groupby(["game_type", "p_status"])[cols].agg(["size", "mean"])
    print("\n=== 2024 투수 이동 상태별 실패모드 ===")
    print(z.round(4).to_string())


def home_team_regime(df, min_n=1000):
    # 초(T)에는 홈팀이 수비하므로 pitcher_team, 말(B)에는 batter_team이 홈팀이다.
    d = df.copy()
    d["home_team"] = np.where(d.top_bottom == "T", d.pitcher_team_id,
                              d.batter_team_id)
    f = (d[(d.game_type == "F") & (d.season >= 2020)]
           .groupby(["home_team", "season"])
           .agg(n=(T, "size"), rate=(T, "mean")).reset_index())
    f = f[f.n >= min_n]
    p = f.pivot(index="home_team", columns="season", values="rate")
    print("\n=== F 홈팀 proxy별 성공률 (ABS 설치/구장 이질성 후보) ===")
    print(p.round(4).to_string())
    if {2022, 2023, 2024} <= set(p.columns):
        a = p[[2022, 2023, 2024]].dropna()
        if len(a):
            d1, d2 = a[2023] - a[2022], a[2024] - a[2023]
            print(f"teams={len(a)} | 22→23 sd={d1.std():.4f} | "
                  f"23→24 sd={d2.std():.4f} | corr={d1.corr(d2):+.3f}")


def bss(y, p):
    r = float(np.mean(y))
    return 1e5 * (1 - float(np.mean((np.asarray(p) - y) ** 2)) / (r * (1 - r)))


def transfer_f_to_r(df, target):
    """2023 R에서 F→R 잔차를 고정해 2024 R에 적용하는 정직한 전이 게이트."""
    paths = sorted(glob("./out/cat_H3_base_s*_val_preds.npz"))
    src = df[(df.season == 2023) & (df.game_type == "R")].copy()
    if not paths or target is None:
        print("\nH3 source/VB2 target 없음: F→R 전이 생략")
        return
    ps = []
    for p in paths:
        z = np.load(p)
        if len(z["pred"]) != len(src) or not np.array_equal(z["y"], src[T].to_numpy()):
            raise ValueError(f"{p}: 2023 R 정렬 불일치")
        ps.append(z["pred"])
    src["pred"] = np.mean(ps, axis=0)
    src["resid"] = src[T] - src.pred
    m = src.p_status == "F_to_R"
    b1, b0, w = src.loc[m, "resid"].mean(), src.loc[~m, "resid"].mean(), m.mean()
    gap = b1 - b0
    # source 구성에서 평균 0인 두 상수. 전역 수준 보정과 직교시킨다.
    a1, a0 = gap * (1 - w), -gap * w
    tar = target[target.game_type == "R"].copy()
    mt = tar.p_status == "F_to_R"
    adj = np.where(mt, a1, a0)
    raw0, raw1 = bss(tar[T].to_numpy(), tar.pred), bss(tar[T].to_numpy(), tar.pred + adj)
    c0 = tar.pred + (tar[T].mean() - tar.pred.mean())
    c1 = tar.pred + adj
    c1 += tar[T].mean() - c1.mean()
    print("\n=== 2023R → 2024R F_to_R frozen residual transfer ===")
    print(f"source H3 {len(paths)} seeds | F_to_R n={m.sum():,}, bias={b1:+.5f}; "
          f"other bias={b0:+.5f}; conditional gap={gap:+.5f}")
    print(f"frozen offsets F_to_R={a1:+.5f}, other={a0:+.5f} | "
          f"target share={mt.mean():.3f}")
    print(f"target raw BSS {raw0:.3f} -> {raw1:.3f} ({raw1-raw0:+.3f})")
    print(f"target centered BSS {bss(tar[T], c0):.3f} -> {bss(tar[T], c1):.3f} "
          f"({bss(tar[T], c1)-bss(tar[T], c0):+.3f})")


def main():
    df = pd.read_csv("./data/train.csv", usecols=COLS)
    sys.path.insert(0, "src")
    import failmode
    lab = failmode._pitch_labels(df)
    df = pd.concat([df, lab], axis=1)
    print("=== 시즌·리그별 성공/실패모드 ===")
    print(df.groupby(["season", "game_type"])[[T, "middle", "ball", "reverse"]]
            .mean().round(4).to_string())
    df = add_status(df, "pitcher_id", "p")
    df = add_status(df, "batter_id", "b")
    print("=== 시즌·리그 기저율 ===")
    print(df.groupby(["season", "game_type"])[T].agg(["size", "mean"])
            .round(4).to_string())
    print_rates(df, "p_status")
    print_rates(df, "b_status")
    home_team_regime(df)
    d = attach_pred_2024(df)
    print_residual(d, "p_status")
    print_residual(d, "b_status")
    print_failure(d)
    if d is not None:
        d["home_team"] = np.where(d.top_bottom == "T", d.pitcher_team_id,
                                  d.batter_team_id)
        print_residual(d[d.game_type == "F"], "home_team", min_n=500)
        d["p_same_bin"] = pd.cut(d.p_same_league_n,
                                  [-1, 0, 50, 200, 1000, np.inf])
        print_residual(d, "p_same_bin", min_n=300)
    transfer_f_to_r(df, d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

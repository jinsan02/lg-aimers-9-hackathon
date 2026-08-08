"""현행 base/cell 블렌드의 행별 조건부 가중이 시간 반분을 넘어가는지 검문한다.

새 모델을 학습하기 전에 이미 있는 미학습 2024 시드 예측만 쓴다. 전반기에 적합한
가중 규칙을 후반기에, 후반기에 적합한 규칙을 전반기에 적용한다. 평가는 각 target
반분의 전역 편향을 제거해 순위/해상도 변화만 본다. 같은 반분 자기적합 수치는
출력하지 않는다.

실행 (4070 프로젝트 루트):
  python tools/orthogonal_gate.py VB2_base ZD5
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd


CELL_SUBMISSION_SEEDS = {42, 7, 13, 3, 4, 5}
CURRENT_W = 0.55


def _seed(path: str) -> int:
    return int(path.rsplit("_s", 1)[1].split("_", 1)[0])


def load(tag: str, cell: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(glob.glob(f"./out/*_{tag}_s*_val_preds.npz"))
    if cell:
        paths = [p for p in paths if _seed(p) in CELL_SUBMISSION_SEEDS]
    if not paths:
        raise FileNotFoundError(f"예측 없음: {tag}")
    z = [np.load(p, allow_pickle=True) for p in paths]
    y = z[0]["y"].astype(np.float64)
    if any(not np.array_equal(y, q["y"]) for q in z[1:]):
        raise ValueError(f"시드 y 불일치: {tag}")
    P = np.stack([q["pred"].astype(np.float64) for q in z])
    return P.mean(0), P.std(0), y


def centered_bss(y: np.ndarray, p: np.ndarray) -> float:
    p = p - np.mean(p - y)
    r = float(y.mean())
    return 1e5 * (1.0 - np.mean((np.clip(p, 0.0, 1.0) - y) ** 2)
                  / (r * (1.0 - r)))


def fit_weight(b: np.ndarray, c: np.ndarray, y: np.ndarray) -> float:
    d = c - b
    dc = d - d.mean()
    target = (y - b) - np.mean(y - b)
    den = float(dc @ dc)
    return float(np.clip((dc @ target) / den, 0.0, 1.0)) if den > 0 else CURRENT_W


def fixed_bins(values: pd.Series, cuts: list[float], labels: list[str]) -> np.ndarray:
    return pd.cut(values.fillna(-1), cuts, labels=labels).astype(str).to_numpy()


def quantile_axis(fit_v: np.ndarray, test_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(fit_v, [0, .2, .4, .6, .8, 1]))
    if len(edges) < 3:
        return np.zeros(len(fit_v), dtype=int), np.zeros(len(test_v), dtype=int)
    return np.digitize(fit_v, edges[1:-1]), np.digitize(test_v, edges[1:-1])


def segmented_pred(bf: np.ndarray, cf: np.ndarray, yf: np.ndarray,
                   bt: np.ndarray, ct: np.ndarray,
                   sf: np.ndarray, st: np.ndarray, k: float = 2000.0
                   ) -> tuple[np.ndarray, float, float, float]:
    wg = fit_weight(bf, cf, yf)
    wt = np.full(len(bt), wg, dtype=np.float64)
    learned = []
    for value in np.unique(sf):
        mf = sf == value
        if mf.sum() < 500:
            continue
        ws = fit_weight(bf[mf], cf[mf], yf[mf])
        shrink = mf.sum() / (mf.sum() + k)
        ws = wg + shrink * (ws - wg)
        wt[st == value] = ws
        learned.append(ws)
    lo = min(learned, default=wg)
    hi = max(learned, default=wg)
    return (1 - wt) * bt + wt * ct, wg, lo, hi


def transfer_segment_pred(bf: np.ndarray, cf: np.ndarray, yf: np.ndarray,
                          bt: np.ndarray, ct: np.ndarray,
                          sf: np.ndarray, st: np.ndarray, k: float = 2000.0
                          ) -> tuple[np.ndarray, float, float]:
    """서로 다른 세대 멤버 사이에서는 전역 w가 아니라 그룹별 w 편차만 옮긴다."""
    wg = fit_weight(bf, cf, yf)
    wt = np.full(len(bt), CURRENT_W, dtype=np.float64)
    deltas = []
    for value in np.unique(sf):
        mf = sf == value
        if mf.sum() < 500:
            continue
        ws = fit_weight(bf[mf], cf[mf], yf[mf])
        shrink = mf.sum() / (mf.sum() + k)
        delta = shrink * (ws - wg)
        wt[st == value] = np.clip(CURRENT_W + delta, 0.0, 1.0)
        deltas.append(delta)
    return (1 - wt) * bt + wt * ct, min(deltas, default=0.0), max(deltas, default=0.0)


def ridge_pred(bf: np.ndarray, cf: np.ndarray, yf: np.ndarray,
               sbf: np.ndarray, scf: np.ndarray, ff: np.ndarray,
               bt: np.ndarray, ct: np.ndarray,
               sbt: np.ndarray, sct: np.ndarray, ft: np.ndarray,
               alpha: float) -> np.ndarray:
    """고정 w=.55 위에 저차원 행별 correction을 학습한다."""
    df, dt = cf - bf, ct - bt
    qf, qt = (1 - CURRENT_W) * bf + CURRENT_W * cf, (1 - CURRENT_W) * bt + CURRENT_W * ct
    Xf = np.column_stack([df, df * np.abs(df), df * sbf, df * scf, df * ff])
    Xt = np.column_stack([dt, dt * np.abs(dt), dt * sbt, dt * sct, dt * ft])
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd < 1e-12] = 1.0
    Xf, Xt = (Xf - mu) / sd, (Xt - mu) / sd
    target = (yf - qf) - np.mean(yf - qf)
    reg = alpha * len(Xf) * np.eye(Xf.shape[1])
    beta = np.linalg.solve(Xf.T @ Xf + reg, Xf.T @ target)
    return qt + Xt @ beta


def main() -> int:
    base_tag = sys.argv[1] if len(sys.argv) > 1 else "VB2_base"
    cell_tag = sys.argv[2] if len(sys.argv) > 2 else "ZD5"
    b, sb, y = load(base_tag)
    c, sc, yc = load(cell_tag, cell=True)
    if not np.array_equal(y, yc):
        raise ValueError("base/cell y 불일치")
    cols = ["season", "game_month", "game_type", "balls_before", "strikes_before",
            "inning", "asof_pitcher_n", "asof_batter_n", "control_success"]
    raw = pd.read_csv("./data/train.csv", encoding="utf-8-sig", usecols=cols)
    raw = raw[raw.season == 2024].reset_index(drop=True)
    if len(raw) != len(y) or not np.array_equal(raw.control_success.to_numpy(), y):
        raise ValueError("2024 원본/예측 정렬 불일치")

    exp_cuts = [-np.inf, 200, 1000, 3000, 10000, np.inf]
    exp_labels = ["~200", "~1k", "~3k", "~10k", "10k+"]
    axes = {
        "league": raw.game_type.astype(str).to_numpy(),
        "count": (raw.balls_before.astype(str) + "-" + raw.strikes_before.astype(str)).to_numpy(),
        "inning": pd.cut(raw.inning, [-np.inf, 2, 5, 8, np.inf],
                          labels=["1-2", "3-5", "6-8", "9+"]).astype(str).to_numpy(),
        "pitcher_exp": fixed_bins(raw.asof_pitcher_n, exp_cuts, exp_labels),
        "batter_exp": fixed_bins(raw.asof_batter_n, exp_cuts, exp_labels),
    }
    first = raw.game_month.to_numpy() <= 6
    current = (1 - CURRENT_W) * b + CURRENT_W * c
    print(f"base={base_tag} cell={cell_tag} rows={len(y):,} current_w={CURRENT_W:.2f}")
    print(f"ensemble rms={np.sqrt(np.mean((b-c)**2)):.6f} | "
          f"seed_sd base={sb.mean():.6f} cell={sc.mean():.6f}")
    print("delta는 target 반분에서 현행 w=.55 대비 centered BSS 변화")

    directions = [("전반→후반", first, ~first), ("후반→전반", ~first, first)]
    totals: dict[str, list[float]] = {}
    for label, fit, test in directions:
        ref = centered_bss(y[test], current[test])
        wg = fit_weight(b[fit], c[fit], y[fit])
        pg = (1 - wg) * b[test] + wg * c[test]
        print(f"\n[{label}] fit={fit.sum():,} test={test.sum():,} "
              f"source_global_w={wg:.3f} delta={centered_bss(y[test], pg)-ref:+.3f}")
        for name, axis in axes.items():
            p, _, lo, hi = segmented_pred(b[fit], c[fit], y[fit], b[test], c[test],
                                           axis[fit], axis[test])
            delta = centered_bss(y[test], p) - ref
            totals.setdefault(name, []).append(delta)
            print(f"  seg {name:<12} {delta:+8.3f}  w_range={lo:.3f}..{hi:.3f}")
        dynamic = {
            "abs_diff": np.abs(c - b),
            "base_seed_sd": sb,
            "cell_seed_sd": sc,
        }
        for name, values in dynamic.items():
            sf, st = quantile_axis(values[fit], values[test])
            p, _, lo, hi = segmented_pred(b[fit], c[fit], y[fit], b[test], c[test], sf, st)
            delta = centered_bss(y[test], p) - ref
            totals.setdefault(name, []).append(delta)
            print(f"  seg {name:<12} {delta:+8.3f}  w_range={lo:.3f}..{hi:.3f}")
        ff = (raw.game_type.to_numpy()[fit] == "F").astype(float)
        ft = (raw.game_type.to_numpy()[test] == "F").astype(float)
        for alpha in (0.01, 0.1, 1.0):
            p = ridge_pred(b[fit], c[fit], y[fit], sb[fit], sc[fit], ff,
                           b[test], c[test], sb[test], sc[test], ft, alpha)
            delta = centered_bss(y[test], p) - ref
            key = f"ridge_a{alpha:g}"
            totals.setdefault(key, []).append(delta)
            print(f"  {key:<16} {delta:+8.3f}")

    print("\n=== 양방향 요약 ===")
    print(f"{'rule':<18}{'first->second':>15}{'second->first':>15}{'min':>10}{'mean':>10}")
    for name, vals in totals.items():
        print(f"{name:<18}{vals[0]:>+15.3f}{vals[1]:>+15.3f}"
              f"{min(vals):>+10.3f}{np.mean(vals):>+10.3f}")
    print("\n승격 게이트: 양방향 모두 양수이고 min +1 이상인 단순 규칙만 후보로 남긴다.")

    # 과거에 남은 2023 R리그 base/cell 예측으로 그룹별 가중 *편차*가 2024 R에도
    # 넘어가는지 본다. W_cell은 현행 ZD5와 세대가 달라 전역 가중 자체는 옮기지 않는다.
    try:
        bs, sbs, ys = load("W_base")
        cs, scs, ysc = load("W_cell", cell=True)
    except FileNotFoundError:
        return 0
    if not np.array_equal(ys, ysc):
        raise ValueError("W_base/W_cell y 불일치")
    source = pd.read_csv("./data/train.csv", encoding="utf-8-sig", usecols=cols)
    source = source[(source.season == 2023) & (source.game_type == "R")].reset_index(drop=True)
    if len(source) != len(ys) or not np.array_equal(source.control_success.to_numpy(), ys):
        raise ValueError("2023R W 예측/원본 정렬 불일치")
    target_mask = raw.game_type.to_numpy() == "R"
    target = raw.loc[target_mask].reset_index(drop=True)
    br, cr, yr = b[target_mask], c[target_mask], y[target_mask]
    ref = centered_bss(yr, (1 - CURRENT_W) * br + CURRENT_W * cr)
    source_axes = {
        "count": (source.balls_before.astype(str) + "-" + source.strikes_before.astype(str)).to_numpy(),
        "inning": pd.cut(source.inning, [-np.inf, 2, 5, 8, np.inf],
                          labels=["1-2", "3-5", "6-8", "9+"]).astype(str).to_numpy(),
        "pitcher_exp": fixed_bins(source.asof_pitcher_n, exp_cuts, exp_labels),
        "batter_exp": fixed_bins(source.asof_batter_n, exp_cuts, exp_labels),
    }
    target_axes = {
        "count": (target.balls_before.astype(str) + "-" + target.strikes_before.astype(str)).to_numpy(),
        "inning": pd.cut(target.inning, [-np.inf, 2, 5, 8, np.inf],
                          labels=["1-2", "3-5", "6-8", "9+"]).astype(str).to_numpy(),
        "pitcher_exp": fixed_bins(target.asof_pitcher_n, exp_cuts, exp_labels),
        "batter_exp": fixed_bins(target.asof_batter_n, exp_cuts, exp_labels),
    }
    print("\n=== 2023 R -> 2024 R 그룹별 가중 편차 전이 (구 W 세대, 보조 근거) ===")
    print(f"source rows={len(ys):,} target rows={len(yr):,} "
          f"source_global_w={fit_weight(bs, cs, ys):.3f}")
    for name in source_axes:
        p, dlo, dhi = transfer_segment_pred(bs, cs, ys, br, cr,
                                             source_axes[name], target_axes[name])
        print(f"  {name:<12} {centered_bss(yr, p)-ref:+8.3f}  "
              f"delta_w={dlo:+.3f}..{dhi:+.3f}")
    dynamic = {
        "abs_diff": (np.abs(cs - bs), np.abs(cr - br)),
        "base_seed_sd": (sbs, sb[target_mask]),
        "cell_seed_sd": (scs, sc[target_mask]),
    }
    for name, (vs, vt) in dynamic.items():
        ss, st = quantile_axis(vs, vt)
        p, dlo, dhi = transfer_segment_pred(bs, cs, ys, br, cr, ss, st)
        print(f"  {name:<12} {centered_bss(yr, p)-ref:+8.3f}  "
              f"delta_w={dlo:+.3f}..{dhi:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

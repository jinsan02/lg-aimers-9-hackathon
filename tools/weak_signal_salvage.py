"""Near-miss 신호의 장점만 현행 K0 위에서 보존할 수 있는지 감사한다.

1) 5K cell이 2023에서 개선한 저카디널리티 그룹만 고정해 2024에 적용.
2) 예측 구종확률 보정이 recent-middle + exact-PB(K0) 뒤에도 남는지 확인.

모든 gate는 source=2023 라벨로만 정하고 target=2024는 평가에만 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbmf_transfer_audit import (PAIR_K, bss, fit_map, middle_keys, pair_keys,
                                post)
from pitch_proba_pair_gate import fit_apply as fit_apply_pair
from pitch_uncertainty_gate import fit_apply as fit_apply_1d


OUT = Path("out")


def load(name, kind):
    z = np.load(OUT/f"cat_{name}_{kind}_preds.npz", allow_pickle=True)
    return z["y"].astype(float), z["pred"].astype(float)


def qbin_fit_apply(xs, xt, q=4):
    good = xs[np.isfinite(xs)]
    e = np.unique(np.quantile(good, np.linspace(0, 1, q+1)))
    if len(e) != q+1:
        return np.zeros(len(xs), int), np.zeros(len(xt), int)
    e[0], e[-1] = -np.inf, np.inf
    a = np.searchsorted(e[1:-1], xs, side="right")
    b = np.searchsorted(e[1:-1], xt, side="right")
    a[~np.isfinite(xs)], b[~np.isfinite(xt)] = -1, -1
    return a, b


def segment_gain(y, p, d, group, scale=.25):
    out = {}
    for v in pd.unique(group):
        m = np.asarray(group == v)
        if m.sum() < 5000:
            continue
        out[str(v)] = bss(y[m], p[m]+scale*d[m])-bss(y[m], p[m])
    return out


def main():
    ys, base_s = load("MVA_native", "val")
    yt, base_t = load("MVA_native", "test")
    ysc, old_s = load("MVCELL_s42", "val")
    ytc, old_t = load("MVCELL_s42", "test")
    _, new_s = load("MVCELL5K_s42", "val")
    _, new_t = load("MVCELL5K_s42", "test")
    assert np.array_equal(ys, ysc) and np.array_equal(yt, ytc)

    use = ["season", "game_month", "game_type", "inning", "balls_before",
           "strikes_before", "num_runners_on", "base_state", "pitcher_hand",
           "batter_hand", "pitcher_id", "batter_id", "asof_pitcher_n",
           "asof_batter_n", "asof_pitcher_prev5_game_middle_rate"]
    d = pd.read_csv("data/train.csv", usecols=use)
    src = d[d.season.eq(2023)].reset_index(drop=True)
    tgt = d[d.season.eq(2024)].reset_index(drop=True)
    assert len(src) == len(ys) and len(tgt) == len(yt)

    core_s = post(.45*base_s+.55*old_s)
    core_t = post(.45*base_t+.55*old_t)
    newcore_s = post(.45*base_s+.55*new_s)
    newcore_t = post(.45*base_t+.55*new_t)
    cell_ds, cell_dt = newcore_s-core_s, newcore_t-core_t

    pexp_s, pexp_t = qbin_fit_apply(src.asof_pitcher_n.to_numpy(float),
                                    tgt.asof_pitcher_n.to_numpy(float))
    bexp_s, bexp_t = qbin_fit_apply(src.asof_batter_n.to_numpy(float),
                                    tgt.asof_batter_n.to_numpy(float))
    conf_s, conf_t = qbin_fit_apply(core_s, core_t)
    dis_s, dis_t = qbin_fit_apply(np.abs(cell_ds), np.abs(cell_dt))
    axes = {
        "league": (src.game_type.to_numpy(), tgt.game_type.to_numpy()),
        "phase": (np.where(src.game_month.le(6), "early", "late"),
                  np.where(tgt.game_month.le(6), "early", "late")),
        "inning3": ((np.minimum((src.inning-1)//3, 2)).to_numpy(),
                    (np.minimum((tgt.inning-1)//3, 2)).to_numpy()),
        "count_family": (np.where(src.balls_before.eq(3), "3ball",
                            np.where(src.strikes_before.eq(2), "2strike", "neutral")),
                         np.where(tgt.balls_before.eq(3), "3ball",
                            np.where(tgt.strikes_before.eq(2), "2strike", "neutral"))),
        "runners": (src.num_runners_on.to_numpy(), tgt.num_runners_on.to_numpy()),
        "hands": ((src.pitcher_hand.astype(str)+src.batter_hand.astype(str)).to_numpy(),
                  (tgt.pitcher_hand.astype(str)+tgt.batter_hand.astype(str)).to_numpy()),
        "pitcher_exp_q4": (pexp_s, pexp_t),
        "batter_exp_q4": (bexp_s, bexp_t),
        "core_conf_q4": (conf_s, conf_t),
        "old_new_disagree_q4": (dis_s, dis_t),
    }

    report = {"cell_segments": {}, "pitch_on_k0": {}}
    for name, (gs, gt) in axes.items():
        per = segment_gain(ys, core_s, cell_ds, gs, .25)
        keep = {v for v, gain in per.items() if gain > 0}
        ms = np.array([str(v) in keep for v in gs])
        mt = np.array([str(v) in keep for v in gt])
        sg = bss(ys, core_s+.25*cell_ds*ms)-bss(ys, core_s)
        tg = bss(yt, core_t+.25*cell_dt*mt)-bss(yt, core_t)
        report["cell_segments"][name] = {
            "kept": sorted(keep), "source_coverage": float(ms.mean()),
            "target_coverage": float(mt.mean()), "source_gain": sg,
            "target_gain": tg, "per_source_group": per}
        print(f"cell {name:<23} keep={sorted(keep)} cov={mt.mean():.1%} "
              f"gain={sg:+.3f}/{tg:+.3f}")

    # 현행 recent-middle q8/k500 + exact-PB k500을 2023에서 fit해 2024에 동결.
    ms, mt = middle_keys(src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
                         tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r0 = ys-core_s
    r0 -= r0.mean()
    ma, mb = fit_map(ms, mt, r0, 500.)
    r1 = ys-(core_s+ma)
    r1 -= r1.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r1, PAIR_K)
    k0s, k0t = core_s+ma+ea, core_t+mb+eb
    print(f"K0 base gains source/target="
          f"{bss(ys,k0s)-bss(ys,core_s):+.3f}/{bss(yt,k0t)-bss(yt,core_t):+.3f}")

    z = np.load(OUT/"pitch_uncertainty_gate.npz")
    q3, q4 = z["q2023"], z["q2024"]
    early = src.game_month.le(6).to_numpy()
    rk = ys-k0s

    def audit(name, full_delta, cv_delta):
        vals = {}
        for w in (.25, .5):
            vals[f"w{w:g}_source_cv"] = bss(ys,k0s+w*cv_delta)-bss(ys,k0s)
            vals[f"w{w:g}_target"] = bss(yt,k0t+w*full_delta)-bss(yt,k0t)
        report["pitch_on_k0"][name] = vals
        print(f"pitch {name:<24} {json.dumps(vals)}")

    # offspeed 1D
    cv = np.zeros(len(ys))
    cv[~early] = fit_apply_1d(q3[early,2], rk[early], q3[~early,2])
    cv[early] = fit_apply_1d(q3[~early,2], rk[~early], q3[early,2])
    td = fit_apply_1d(q3[:,2], rk, q4[:,2])
    audit("offspeed", td, cv)

    # fastball x breaking 2D
    cv2 = np.zeros(len(ys))
    cv2[~early] = fit_apply_pair(q3[early,:2], rk[early], q3[~early,:2])
    cv2[early] = fit_apply_pair(q3[~early,:2], rk[~early], q3[early,:2])
    td2 = fit_apply_pair(q3[:,:2], rk, q4[:,:2])
    audit("fastball_x_breaking", td2, cv2)

    # 사전 고정 scale의 두 약신호를 평균해 중복을 제한.
    combo_cv = .25*cv + .25*cv2
    combo_t = .25*td + .25*td2
    report["pitch_on_k0"]["equal_combo"] = {
        "source_cv": bss(ys,k0s+combo_cv)-bss(ys,k0s),
        "target": bss(yt,k0t+combo_t)-bss(yt,k0t),
    }
    print("pitch equal_combo", report["pitch_on_k0"]["equal_combo"])

    # 다른 출력 기하/분할 알고리즘이 K0 앙상블에 기여하는지 고정 가중 검문.
    for key, stem in (("ft_blend", "ft_FTT1_s42"),
                      ("xgb_blend", "xgb_XCR1")):
        fv = np.load(OUT/f"{stem}_val_preds.npz", allow_pickle=True)
        ft = np.load(OUT/f"{stem}_test_preds.npz", allow_pickle=True)
        if not np.array_equal(fv["y"], ys) or not np.array_equal(ft["y"], yt):
            raise ValueError(f"{stem} target mismatch")
        report[key] = {"rms_target": float(np.sqrt(np.mean((ft["pred"]-k0t)**2)))}
        for w in (.02, .05, .10, .20):
            report[key][f"w{w:g}_source"] = (
                bss(ys,(1-w)*k0s+w*fv["pred"])-bss(ys,k0s))
            report[key][f"w{w:g}_target"] = (
                bss(yt,(1-w)*k0t+w*ft["pred"])-bss(yt,k0t))
        print(f"{stem} on K0", report[key])

    (OUT/"weak_signal_salvage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

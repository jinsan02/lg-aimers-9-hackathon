"""예측 구종분포의 불확실성이 제구 잔차를 설명하는지 rolling gate.

실제 구종이나 test의 다른 행은 쓰지 않는다. Trackman <=S-1로 학습한 행단독
P(pitch_type|x)에서 entropy/max/margin을 만들고, 시즌 S의 core 잔차 관계가 S+1로
전이되는지만 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from pitch_type_control_gate import (PITCH_CATS, PITCH_FEATURES, TYPES,
                                     load_tables)


OUT = Path("out")


def bss(y, p, center=False):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if center:
        p = p + y.mean() - p.mean()
    r = y.mean()
    return float(1e5 * (1 - np.mean((p-y)**2)/(r*(1-r))))


def fit_proba(tm, target, cutoff, seed=42):
    tr = tm[(tm.season <= cutoff) & tm.pitch_type_group.isin(TYPES)].copy()
    xa, xb = tr[PITCH_FEATURES].copy(), target[PITCH_FEATURES].copy()
    for c in PITCH_CATS:
        xa[c] = xa[c].astype(str)
        xb[c] = xb[c].astype(str)
    m = CatBoostClassifier(
        loss_function="MultiClass", iterations=500, depth=7,
        learning_rate=.08, l2_leaf_reg=10, random_seed=seed,
        task_type="GPU", devices="0", max_ctr_complexity=1,
        allow_writing_files=False, verbose=100)
    m.fit(Pool(xa, tr.pitch_type_group.astype(str), cat_features=PITCH_CATS))
    raw = m.predict_proba(xb)
    return raw[:, [list(m.classes_).index(c) for c in TYPES]]


def signals(q):
    s = np.sort(q, axis=1)
    return {
        "entropy": -(q*np.log(np.clip(q, 1e-12, 1))).sum(1)/np.log(q.shape[1]),
        "max_prob": s[:, -1],
        "top_margin": s[:, -1]-s[:, -2],
        "p_fastball": q[:, 0],
        "p_breaking": q[:, 1],
        "p_offspeed": q[:, 2],
    }


def fit_apply(x_fit, resid_fit, x_apply, q=4, k=500.0):
    edges = np.unique(np.quantile(x_fit, np.linspace(0, 1, q+1)))
    if len(edges) < 3:
        return np.zeros(len(x_apply))
    edges[0], edges[-1] = -np.inf, np.inf
    bf = np.searchsorted(edges[1:-1], x_fit, side="right")
    ba = np.searchsorted(edges[1:-1], x_apply, side="right")
    nb = len(edges)-1
    cnt = np.bincount(bf, minlength=nb)
    sm = np.bincount(bf, weights=resid_fit, minlength=nb)
    d = sm/(cnt+k)
    d -= np.average(d[bf])
    return d[ba]


def main():
    main_df, tm = load_tables()
    src = main_df[main_df.season == 2023].reset_index(drop=True)
    tgt = main_df[main_df.season == 2024].reset_index(drop=True)

    bv = np.load(OUT/"cat_MVA_native_val_preds.npz")
    bt = np.load(OUT/"cat_MVA_native_test_preds.npz", allow_pickle=True)
    cv = np.load(OUT/"cat_MVCELL_s42_val_preds.npz")
    ct = np.load(OUT/"cat_MVCELL_s42_test_preds.npz", allow_pickle=True)
    assert np.array_equal(src.control_success.to_numpy(), bv["y"])
    assert np.array_equal(tgt.control_success.to_numpy(), bt["y"])
    psrc = .45*bv["pred"] + .55*cv["pred"]
    ptgt = .45*bt["pred"] + .55*ct["pred"]
    ysrc, ytgt = bv["y"].astype(float), bt["y"].astype(float)

    print("fit pitch head <=2022 -> 2023", flush=True)
    qsrc = fit_proba(tm, src, 2022)
    print("fit pitch head <=2023 -> 2024", flush=True)
    qtgt = fit_proba(tm, tgt, 2023)
    ss, st = signals(qsrc), signals(qtgt)
    early = src.game_month.le(6).to_numpy()
    resid = ysrc-psrc
    report = {"core_2023": bss(ysrc, psrc), "core_2024": bss(ytgt, ptgt),
              "arms": {}}
    for name in ss:
        cvd = np.zeros(len(src))
        cvd[~early] = fit_apply(ss[name][early], resid[early], ss[name][~early])
        cvd[early] = fit_apply(ss[name][~early], resid[~early], ss[name][early])
        td = fit_apply(ss[name], resid, st[name])
        arm = {"source_signal_mean": float(ss[name].mean()),
               "target_signal_mean": float(st[name].mean()),
               "source_delta_sd": float(cvd.std()),
               "target_delta_sd": float(td.std())}
        for scale in (.25, .5, 1.0):
            arm[f"w{scale:g}_cv_gain"] = bss(ysrc, psrc+scale*cvd)-bss(ysrc, psrc)
            arm[f"w{scale:g}_target_gain"] = bss(ytgt, ptgt+scale*td)-bss(ytgt, ptgt)
            arm[f"w{scale:g}_target_centered_gain"] = (
                bss(ytgt, ptgt+scale*td, True)-bss(ytgt, ptgt, True))
        report["arms"][name] = arm
        print(name, json.dumps(arm, ensure_ascii=False), flush=True)

    path = OUT/"pitch_uncertainty_gate.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(OUT/"pitch_uncertainty_gate.npz",
                        q2023=qsrc, q2024=qtgt, y2023=ysrc, y2024=ytgt,
                        core2023=psrc, core2024=ptgt)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

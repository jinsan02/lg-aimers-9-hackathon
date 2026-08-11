"""예측 구종확률 2차원 조합의 강수축 rolling 잔차 gate."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bss(y, p, center=False):
    if center:
        p = p + y.mean()-p.mean()
    r = y.mean()
    return 1e5*(1-np.mean((p-y)**2)/(r*(1-r)))


def bins(x, edges):
    return np.searchsorted(edges[1:-1], x, side="right")


def fit_apply(xf, rf, xa, q=4, k=1000.0):
    ee = []
    for j in range(2):
        e = np.unique(np.quantile(xf[:, j], np.linspace(0, 1, q+1)))
        if len(e) != q+1:
            return np.zeros(len(xa))
        e[0], e[-1] = -np.inf, np.inf
        ee.append(e)
    cf = bins(xf[:, 0], ee[0])*q + bins(xf[:, 1], ee[1])
    ca = bins(xa[:, 0], ee[0])*q + bins(xa[:, 1], ee[1])
    cnt = np.bincount(cf, minlength=q*q)
    sm = np.bincount(cf, weights=rf, minlength=q*q)
    d = sm/(cnt+k)
    d -= np.mean(d[cf])
    return d[ca]


def main():
    z = np.load(Path("out/pitch_uncertainty_gate.npz"))
    q3, q4 = z["q2023"], z["q2024"]
    y3, y4 = z["y2023"], z["y2024"]
    p3, p4 = z["core2023"], z["core2024"]
    d = pd.read_csv("data/train.csv", usecols=["season", "game_month"])
    early = d[d.season.eq(2023)].game_month.le(6).to_numpy()
    r3 = y3-p3
    names = ["fastball", "breaking", "offspeed"]
    report = {}
    for a, b in itertools.combinations(range(3), 2):
        x3, x4 = q3[:, [a,b]], q4[:, [a,b]]
        cv = np.zeros(len(y3))
        cv[~early] = fit_apply(x3[early], r3[early], x3[~early])
        cv[early] = fit_apply(x3[~early], r3[~early], x3[early])
        td = fit_apply(x3, r3, x4)
        arm = {}
        for w in (.25, .5, 1.0):
            arm[f"w{w:g}_cv"] = bss(y3,p3+w*cv)-bss(y3,p3)
            arm[f"w{w:g}_target"] = bss(y4,p4+w*td)-bss(y4,p4)
            arm[f"w{w:g}_target_centered"] = (
                bss(y4,p4+w*td,True)-bss(y4,p4,True))
        key = f"{names[a]}_x_{names[b]}"
        report[key] = arm
        print(key, json.dumps(arm), flush=True)
    Path("out/pitch_proba_pair_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

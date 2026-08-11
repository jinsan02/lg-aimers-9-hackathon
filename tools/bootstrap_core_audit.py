"""Evaluate a bootstrap variant as a partial replacement inside the K0 core."""

from pathlib import Path

import numpy as np
import pandas as pd

from pbmf_transfer_audit import PAIR_K, bss, fit_map, middle_keys, pair_keys, post


ROOT = Path(__file__).resolve().parents[1]


def load(stem, kind):
    z = np.load(ROOT / "out" / f"cat_{stem}_{kind}_preds.npz")
    return z["y"].astype(float), z["pred"].astype(float)


def report(label, frame, y, before, after):
    masks = {"all": np.ones(len(frame), bool),
             "R": frame.game_type.eq("R").to_numpy(),
             "F": frame.game_type.eq("F").to_numpy(),
             "early": frame.game_month.le(6).to_numpy(),
             "late": frame.game_month.gt(6).to_numpy()}
    vals = {k: bss(y[m], after[m])-bss(y[m], before[m]) for k,m in masks.items()}
    print(f"{label:<25} " + " ".join(f"{k}={v:+.3f}" for k,v in vals.items()))


def main():
    ys, bs = load("MVA_native", "val")
    yt, bt = load("MVA_native", "test")
    ysc, cs = load("MVCELL_s42", "val")
    ytc, ct = load("MVCELL_s42", "test")
    ysv, vs = load("BTP1", "val")
    ytv, vt = load("BTP1", "test")
    if not all(np.array_equal(a, b) for a,b in ((ys,ysc),(ys,ysv),(yt,ytc),(yt,ytv))):
        raise ValueError("surface target mismatch")

    cols = ["season","game_month","game_type","pitcher_id","batter_id",
            "asof_pitcher_prev5_game_middle_rate"]
    data = pd.read_csv(ROOT/"data"/"train.csv", usecols=cols)
    src = data[data.season.eq(2023)].reset_index(drop=True)
    tgt = data[data.season.eq(2024)].reset_index(drop=True)
    core_s, core_t = post(.45*bs+.55*cs), post(.45*bt+.55*ct)
    ms, mt = middle_keys(src.asof_pitcher_prev5_game_middle_rate.to_numpy(float),
                         tgt.asof_pitcher_prev5_game_middle_rate.to_numpy(float))
    r = ys-core_s; r -= r.mean()
    ma, mb = fit_map(ms, mt, r, 500.)
    r = ys-(core_s+ma); r -= r.mean()
    ea, eb = fit_map(pair_keys(src), pair_keys(tgt), r, PAIR_K)
    k0s, k0t = core_s+ma+ea, core_t+mb+eb

    print("=== BTP1 Bernoulli bootstrap core replacement ===")
    report("standalone", tgt, yt, bt, vt)
    for w in (.10,.25,.50,1.0):
        cand_s = post(.45*((1-w)*bs+w*vs)+.55*cs)+ma+ea
        cand_t = post(.45*((1-w)*bt+w*vt)+.55*ct)+mb+eb
        report(f"base replace {w:g} source", src, ys, k0s, cand_s)
        report(f"base replace {w:g} target", tgt, yt, k0t, cand_t)


if __name__ == "__main__":
    main()

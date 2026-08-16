"""Time-honest audit of player plate-appearance depth proxies.

The released rows do not contain a game/PA id or the actual pitch number, so
test-row sequencing is forbidden and unavailable.  Completed prior seasons can
still estimate a player's tendency to reach two strikes or a deep count.  This
script checks whether those frozen, label-free player summaries explain a
champion's residual in two rolling transitions before any GPU feature run.
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from invalidated import guard as _guard_invalidated                # noqa: E402

import numpy as np
import pandas as pd

from full_dataset_audit import raw_bss
from full_residual_audit import correction, load_frames


def table(all_rows: pd.DataFrame, cutoff: int, role: str, k: float = 500.0):
    hist = all_rows.loc[all_rows.season < cutoff].copy()
    hist["two"] = (hist.strikes_before == 2).astype(float)
    hist["deep"] = ((hist.balls_before + hist.strikes_before) >= 4).astype(float)
    hist["full"] = ((hist.balls_before == 3) &
                    (hist.strikes_before == 2)).astype(float)
    key = f"{role}_id"
    g = hist.groupby(key)[["two", "deep", "full"]].agg(["sum", "size"])
    out = pd.DataFrame(index=g.index)
    for metric in ("two", "deep", "full"):
        n = g[(metric, "size")].to_numpy(float)
        s = g[(metric, "sum")].to_numpy(float)
        prior = float(hist[metric].mean())
        out[f"{role}_{metric}"] = (s + k * prior) / (n + k)
    return out


def attach(rows, all_rows, cutoff):
    out = pd.DataFrame(index=rows.index)
    for role in ("pitcher", "batter"):
        tab = table(all_rows, cutoff, role)
        ids = rows[f"{role}_id"]
        for c in tab.columns:
            out[c] = ids.map(tab[c]).fillna(float(tab[c].mean())).to_numpy()
    out["two_gap"] = out["batter_two"] - out["pitcher_two"]
    out["deep_gap"] = out["batter_deep"] - out["pitcher_deep"]
    return out


def quantile_code(source, target, q=10):
    edges = np.unique(np.quantile(source, np.linspace(0, 1, q + 1)[1:-1]))
    return (np.searchsorted(edges, source, side="right"),
            np.searchsorted(edges, target, side="right"), len(edges) + 1)


def run(tag, source_year, target_year, all_rows):
    src, tgt, _, ps, pt = load_frames(tag, source_year, target_year)
    fs = attach(src, all_rows, source_year)
    ft = attach(tgt, all_rows, target_year)
    resid = src.control_success.to_numpy(float) - ps
    y = tgt.control_success.to_numpy(float)
    base = raw_bss(y, pt)
    rows = []
    current_two = (tgt.strikes_before.to_numpy() == 2).astype(float)
    current_deep = ((tgt.balls_before + tgt.strikes_before).to_numpy() >= 4).astype(float)
    for c in fs.columns:
        cs, ct, n = quantile_code(fs[c].to_numpy(), ft[c].to_numpy())
        add = correction(resid, cs, ct, n, k=500.0)
        rows.append((c, raw_bss(y, np.clip(pt + add, .01, .99)) - base))
        # The historical style should matter most after the PA actually reaches
        # the corresponding state.  The interaction remains row-local.
        gate = current_two if "two" in c else current_deep
        rows.append((c + "_current_gate",
                     raw_bss(y, np.clip(pt + add * gate, .01, .99)) - base))
    return pd.DataFrame(rows, columns=["feature", f"gain_{target_year}"])


def main():
    # These runs predate 5b61fbd (2026-08-13 03:23:39): their fit
    # partitions held later seasons, so the "next season" transitions
    # below were never next-season transitions. Refuse rather than
    # reproduce the numbers. See docs/INVALIDATED.tsv.
    _guard_invalidated(['MVB22_native'])
    cols = ["season", "pitcher_id", "batter_id", "balls_before",
            "strikes_before"]
    all_rows = pd.read_csv("data/train.csv", encoding="utf-8-sig", usecols=cols)
    a = run("MVB22_native", 2022, 2023, all_rows)
    b = run("MVA_native", 2023, 2024, all_rows)
    out = a.merge(b, on="feature")
    out["min_gain"] = out[["gain_2023", "gain_2024"]].min(axis=1)
    out = out.sort_values("min_gain", ascending=False)
    out.to_csv("out/full_audit/pa_depth_proxy_transfer.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

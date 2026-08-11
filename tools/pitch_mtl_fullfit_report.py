"""Audit the v13 failure under submission-faithful preprocessing refits."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRANSITIONS = {
    2023: ("out/pitch_mtl_diag22_base_test2023.npz",
           "out/pitch_mtl_diag22_aux_test2023.npz"),
    2024: ("out/pitch_mtl_diag23_base_test2024.npz",
           "out/pitch_mtl_diag23_aux_test2024.npz"),
}


def bss(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - y) ** 2) / (r * (1 - r)))


def centered_bss(y, p):
    return bss(y, np.clip(p + y.mean() - p.mean(), 0, 1))


def paired(path0, path1):
    z0, z1 = np.load(path0), np.load(path1)
    if not np.array_equal(z0["y"], z1["y"]):
        raise ValueError("target mismatch")
    y, base, aux = z0["y"], z0["members"], z1["members"]
    delta = np.asarray([bss(y, aux[i]) - bss(y, base[i])
                        for i in range(len(base))])
    se = delta.std(ddof=1) / np.sqrt(len(delta))
    return {
        "y": y, "base": base.mean(0), "aux": aux.mean(0),
        "delta": delta, "mean": delta.mean(), "se": se,
        "t": delta.mean() / se,
        "ensemble": bss(y, aux.mean(0)) - bss(y, base.mean(0)),
        "centered": (centered_bss(y, aux.mean(0)) -
                     centered_bss(y, base.mean(0))),
    }


def simple_segments(frame):
    out = frame.copy()
    out["period"] = np.where(out.game_month <= 6, "early", "late")
    out["count"] = (out.balls_before.astype(str) + "-" +
                    out.strikes_before.astype(str))
    out["inn_bin"] = pd.cut(out.inning, [-99, 3, 6, 9, 99],
                             labels=["1-3", "4-6", "7-9", "10+"]).astype(str)
    out["exp_p"] = pd.cut(out.asof_pitcher_n,
                           [-np.inf, 20, 100, 500, np.inf],
                           labels=["p0-20", "p21-100", "p101-500", "p500+"]).astype(str)
    out["exp_b"] = pd.cut(out.asof_batter_n,
                           [-np.inf, 20, 100, 500, np.inf],
                           labels=["b0-20", "b21-100", "b101-500", "b500+"]).astype(str)
    return out


def main():
    reports = {season: paired(*paths) for season, paths in TRANSITIONS.items()}
    for season, report in reports.items():
        print(f"{season-1}->{season}: mean={report['mean']:+.3f} "
              f"SE={report['se']:.3f} t={report['t']:+.3f} "
              f"ensemble={report['ensemble']:+.3f} "
              f"centered={report['centered']:+.3f}")
        print("  seeds " + " ".join(f"{x:+.3f}" for x in report["delta"]))

    qt = paired("out/pitch_mtl_diag23_qt22_base_test2024.npz",
                "out/pitch_mtl_diag23_qt22_aux_test2024.npz")
    print(f"2023->2024 QT<=2022: ensemble={qt['ensemble']:+.3f}, "
          f"paired={qt['mean']:+.3f}, t={qt['t']:+.3f}")

    cols = ["season", "game_type", "game_month", "balls_before",
            "strikes_before", "pitcher_hand", "batter_hand", "inning",
            "num_runners_on", "asof_pitcher_n", "asof_batter_n"]
    raw = pd.read_csv("data/train.csv", usecols=cols)
    segment_tables = []
    axes = ["game_type", "period", "balls_before", "strikes_before", "count",
            "pitcher_hand", "batter_hand", "inn_bin", "num_runners_on",
            "exp_p", "exp_b"]
    for season, report in reports.items():
        frame = simple_segments(raw[raw.season == season].reset_index(drop=True))
        frame["improvement"] = ((report["base"] - report["y"]) ** 2 -
                                (report["aux"] - report["y"]) ** 2)
        rows = []
        for axis in axes:
            for value, group in frame.groupby(axis, dropna=False):
                if len(group) >= 1000:
                    rows.append((axis, str(value), len(group),
                                 group.improvement.mean()))
        segment_tables.append(pd.DataFrame(
            rows, columns=["axis", "value", f"n{season}", f"imp{season}"]))
    merged = segment_tables[0].merge(segment_tables[1], on=["axis", "value"])
    stable = merged[(merged.imp2023 > 0) & (merged.imp2024 > 0)]
    print(f"simple row-local segments positive in both transitions: {len(stable)}")
    if len(stable):
        print(stable.to_string(index=False))


if __name__ == "__main__":
    main()

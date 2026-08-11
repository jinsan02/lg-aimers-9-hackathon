"""Full, time-honest audit of every released competition column.

This is deliberately model-light.  It screens columns and interactions by
learning a smoothed lookup on seasons < S and applying it to season S.  The
same code is therefore usable before CatBoost and cannot accidentally select
on an in-season random split.  Expensive stages are separate so partial work
is preserved::

    python tools/full_dataset_audit.py profiles
    python tools/full_dataset_audit.py singles
    python tools/full_dataset_audit.py pairs
    python tools/full_dataset_audit.py trackman

Outputs are written below out/full_audit/.  ``centered_bss`` is diagnostic
resolution after removing the target-season mean error; only ``raw_bss`` is a
legal deployable score.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from itertools import combinations

import numpy as np
import pandas as pd


DATA = "data"
OUT = "out/full_audit"
TARGET = "control_success"
ID = "row_id"
YEARS = (2022, 2023, 2024)


def raw_bss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    den = float(y.mean() * (1.0 - y.mean()))
    return 100000.0 * (1.0 - float(np.mean((p - y) ** 2)) / den)


def load_main() -> tuple[pd.DataFrame, list[str]]:
    test_cols = list(pd.read_csv(f"{DATA}/test.csv", nrows=0,
                                 encoding="utf-8-sig").columns)
    feats = [c for c in test_cols if c != ID]
    df = pd.read_csv(f"{DATA}/train.csv", encoding="utf-8-sig",
                     usecols=feats + [TARGET])
    return df, feats


def _feature_code(src: pd.Series, tgt: pd.Series) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Source-fitted categorical/quantile code; zero is always missing/unknown."""
    nun = int(src.nunique(dropna=True))
    name = src.name or ""
    is_cat = (src.dtype == object or nun <= 32 or name.endswith("_id"))
    if is_cat:
        vals = pd.Index(src.dropna().unique())
        mapper = pd.Series(np.arange(1, len(vals) + 1, dtype=np.int32), index=vals)
        a = src.map(mapper).fillna(0).to_numpy(np.int32)
        b = tgt.map(mapper).fillna(0).to_numpy(np.int32)
        return a, b, len(vals) + 1, "categorical"

    x = pd.to_numeric(src, errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tgt, errors="coerce").to_numpy(np.float64)
    finite = x[np.isfinite(x)]
    if not len(finite):
        return np.zeros(len(x), np.int32), np.zeros(len(z), np.int32), 1, "empty"
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 17)[1:-1]))
    a = np.where(np.isfinite(x), np.searchsorted(edges, x, side="right") + 1, 0)
    b = np.where(np.isfinite(z), np.searchsorted(edges, z, side="right") + 1, 0)
    return a.astype(np.int32), b.astype(np.int32), len(edges) + 2, "qbin16"


def _lookup(y_src: np.ndarray, c_src: np.ndarray, c_tgt: np.ndarray,
            n_groups: int, k: float = 100.0) -> np.ndarray:
    prior = float(y_src.mean())
    n = np.bincount(c_src, minlength=n_groups).astype(np.float64)
    s = np.bincount(c_src, weights=y_src, minlength=n_groups).astype(np.float64)
    rate = (s + k * prior) / (n + k)
    return rate[c_tgt]


def _scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    raw = raw_bss(y, p)
    pc = np.clip(p + (float(y.mean()) - float(p.mean())), 1e-5, 1 - 1e-5)
    return raw, raw_bss(y, pc), float(np.sqrt(np.mean((p - p.mean()) ** 2)))


def profiles() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, feats = load_main()
    seasons = [int(x) for x in sorted(df.season.unique())]
    rows = []
    for c in feats:
        s = df[c]
        row = {
            "column": c, "dtype": str(s.dtype),
            "nunique": int(s.nunique(dropna=True)),
            "missing": float(s.isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0, .01, .5, .99, 1])
            row.update(min=float(q.iloc[0]), p01=float(q.iloc[1]),
                       median=float(q.iloc[2]), p99=float(q.iloc[3]),
                       max=float(q.iloc[4]), mean=float(s.mean()),
                       std=float(s.std()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/main_profile.csv", index=False)

    miss = df.groupby("season")[feats].apply(lambda x: x.isna().mean())
    miss.to_csv(f"{OUT}/main_missing_by_season.csv")
    base = (df.groupby(["season", "game_type"])[TARGET]
            .agg(["size", "mean"]).reset_index())
    base.to_csv(f"{OUT}/target_by_season_league.csv", index=False)

    nums = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    corr_rows = []
    for c in nums:
        for year in seasons:
            d = df.loc[df.season == year, [c, TARGET]].dropna()
            corr = d[c].corr(d[TARGET]) if d[c].nunique() > 1 else np.nan
            corr_rows.append({"column": c, "season": year, "pearson": corr})
    pd.DataFrame(corr_rows).to_csv(f"{OUT}/target_corr_by_season.csv", index=False)

    sample = df[nums].sample(min(300_000, len(df)), random_state=20260811)
    C = sample.corr(numeric_only=True)
    red = []
    for a, b in combinations(nums, 2):
        v = C.loc[a, b]
        if np.isfinite(v) and abs(v) >= .70:
            red.append({"a": a, "b": b, "corr": float(v), "abs_corr": abs(float(v))})
    pd.DataFrame(red).sort_values("abs_corr", ascending=False).to_csv(
        f"{OUT}/redundant_numeric_pairs.csv", index=False)

    summary = {
        "rows": len(df), "features": len(feats), "seasons": seasons,
        "target_mean": float(df[TARGET].mean()),
        "missing_columns": int(sum(df[c].isna().any() for c in feats)),
    }
    with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


def singles() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, feats = load_main()
    rows = []
    for year in YEARS:
        sm = df.season.to_numpy() < year
        tm = df.season.to_numpy() == year
        ys = df.loc[sm, TARGET].to_numpy(np.float64)
        yt = df.loc[tm, TARGET].to_numpy(np.float64)
        for c in feats:
            a, b, n, kind = _feature_code(df.loc[sm, c], df.loc[tm, c])
            p = _lookup(ys, a, b, n)
            raw, cen, spread = _scores(yt, p)
            rows.append({"season": year, "column": c, "kind": kind,
                         "groups": n, "raw_bss": raw, "centered_bss": cen,
                         "pred_sd": spread, "coverage": float((b != 0).mean())})
        print(f"singletons {year}: {len(feats)} complete", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/singleton_transfer.csv", index=False)
    wide = out.pivot(index="column", columns="season", values="raw_bss")
    wide["mean"] = wide.mean(axis=1)
    wide["min"] = wide[list(YEARS)].min(axis=1)
    wide.sort_values(["min", "mean"], ascending=False).to_csv(
        f"{OUT}/singleton_transfer_ranked.csv")


def _encode_year(df: pd.DataFrame, feats: list[str], year: int):
    sm = df.season.to_numpy() < year
    tm = df.season.to_numpy() == year
    codes = {}
    for c in feats:
        a, b, n, kind = _feature_code(df.loc[sm, c], df.loc[tm, c])
        codes[c] = (a, b, n, kind)
    return sm, tm, codes


def _pair_one(ys, yt, ca, cb, na, nb):
    sa, ta = ca
    sb, tb = cb
    cs = sa.astype(np.int64) * nb + sb
    ct = ta.astype(np.int64) * nb + tb
    p = _lookup(ys, cs, ct, na * nb)
    return _scores(yt, p)


def pairs() -> None:
    os.makedirs(OUT, exist_ok=True)
    df, feats = load_main()
    all_rows = []
    selected: set[tuple[str, str]] = set()
    for year in (2024,):
        sm, tm, codes = _encode_year(df, feats, year)
        ys = df.loc[sm, TARGET].to_numpy(np.float64)
        yt = df.loc[tm, TARGET].to_numpy(np.float64)
        uni = {}
        for c in feats:
            a, b, n, _ = codes[c]
            uni[c] = _scores(yt, _lookup(ys, a, b, n))[0]
        for ix, (a, b) in enumerate(combinations(feats, 2), 1):
            aa, at, na, _ = codes[a]
            ba, bt, nb, _ = codes[b]
            raw, cen, spread = _pair_one(ys, yt, (aa, at), (ba, bt), na, nb)
            all_rows.append({"season": year, "a": a, "b": b,
                             "groups": na * nb, "raw_bss": raw,
                             "centered_bss": cen, "pred_sd": spread,
                             "synergy_raw": raw - max(uni[a], uni[b])})
            if ix % 100 == 0:
                print(f"pairs {year}: {ix}/1081", flush=True)
        latest = pd.DataFrame(all_rows)
        chosen = pd.concat([
            latest.nlargest(180, "synergy_raw")[["a", "b"]],
            latest.nlargest(80, "raw_bss")[["a", "b"]],
        ]).drop_duplicates()
        selected = set(map(tuple, chosen.to_numpy()))
        del codes
        gc.collect()

    for year in (2022, 2023):
        sm, tm, codes = _encode_year(df, feats, year)
        ys = df.loc[sm, TARGET].to_numpy(np.float64)
        yt = df.loc[tm, TARGET].to_numpy(np.float64)
        uni = {}
        for c in feats:
            a, b, n, _ = codes[c]
            uni[c] = _scores(yt, _lookup(ys, a, b, n))[0]
        for ix, (a, b) in enumerate(sorted(selected), 1):
            aa, at, na, _ = codes[a]
            ba, bt, nb, _ = codes[b]
            raw, cen, spread = _pair_one(ys, yt, (aa, at), (ba, bt), na, nb)
            all_rows.append({"season": year, "a": a, "b": b,
                             "groups": na * nb, "raw_bss": raw,
                             "centered_bss": cen, "pred_sd": spread,
                             "synergy_raw": raw - max(uni[a], uni[b])})
        print(f"selected pairs {year}: {len(selected)} complete", flush=True)
        del codes
        gc.collect()
    out = pd.DataFrame(all_rows)
    out.to_csv(f"{OUT}/pair_transfer.csv", index=False)
    stable = out[out[["a", "b"]].apply(tuple, axis=1).isin(selected)]
    rank = (stable.groupby(["a", "b"])
            .agg(years=("season", "nunique"), mean_raw=("raw_bss", "mean"),
                 min_raw=("raw_bss", "min"), mean_synergy=("synergy_raw", "mean"),
                 min_synergy=("synergy_raw", "min"),
                 mean_centered=("centered_bss", "mean"))
            .reset_index())
    rank.sort_values(["min_synergy", "mean_synergy"], ascending=False).to_csv(
        f"{OUT}/pair_transfer_ranked.csv", index=False)


def triples() -> None:
    """Condition the stable pair relations on a compact baseball context."""
    os.makedirs(OUT, exist_ok=True)
    df, feats = load_main()
    pr = pd.read_csv(f"{OUT}/pair_transfer_ranked.csv")
    stable = pr[(pr.years == 3) & (pr.min_raw > 0) & (pr.min_synergy > 0)]
    pair_list = list(stable[["a", "b"]].itertuples(index=False, name=None))
    contexts = [c for c in (
        "balls_before", "strikes_before", "outs_before", "inning",
        "game_type", "pitcher_hand", "batter_hand", "base_state",
        "num_runners_on", "game_month", "asof_pitcher_n",
    ) if c in feats]
    rows = []
    for year in YEARS:
        needed = sorted(set(contexts + [x for p in pair_list for x in p]))
        sm, tm, codes = _encode_year(df, needed, year)
        ys = df.loc[sm, TARGET].to_numpy(np.float64)
        yt = df.loc[tm, TARGET].to_numpy(np.float64)
        for a, b in pair_list:
            aa, at, na, _ = codes[a]
            ba, bt, nb, _ = codes[b]
            pair_raw, _, _ = _pair_one(ys, yt, (aa, at), (ba, bt), na, nb)
            ps = aa.astype(np.int64) * nb + ba
            pt = at.astype(np.int64) * nb + bt
            for c in contexts:
                if c in (a, b):
                    continue
                ca, ct, nc, _ = codes[c]
                groups = na * nb * nc
                if groups > 5_000_000:
                    continue
                ts = ps * nc + ca
                tt = pt * nc + ct
                pred = _lookup(ys, ts, tt, groups)
                raw, cen, spread = _scores(yt, pred)
                rows.append({"season": year, "a": a, "b": b, "c": c,
                             "groups": groups, "raw_bss": raw,
                             "centered_bss": cen, "pred_sd": spread,
                             "gain_over_pair": raw - pair_raw})
        print(f"triples {year}: {len(pair_list)} pairs x contexts complete", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/triple_transfer.csv", index=False)
    rank = (out.groupby(["a", "b", "c"])
            .agg(years=("season", "nunique"), mean_raw=("raw_bss", "mean"),
                 min_raw=("raw_bss", "min"),
                 mean_gain=("gain_over_pair", "mean"),
                 min_gain=("gain_over_pair", "min"),
                 mean_centered=("centered_bss", "mean")).reset_index())
    rank.sort_values(["min_gain", "mean_gain"], ascending=False).to_csv(
        f"{OUT}/triple_transfer_ranked.csv", index=False)


def trackman() -> None:
    os.makedirs(OUT, exist_ok=True)
    tm = pd.read_csv(f"{DATA}/trackman_history.csv", encoding="utf-8-sig")
    rows = []
    for c in tm.columns:
        s = tm[c]
        row = {"column": c, "dtype": str(s.dtype),
               "nunique": int(s.nunique(dropna=True)),
               "missing": float(s.isna().mean())}
        if pd.api.types.is_numeric_dtype(s):
            q = s.quantile([0, .01, .5, .99, 1])
            row.update(min=float(q.iloc[0]), p01=float(q.iloc[1]),
                       median=float(q.iloc[2]), p99=float(q.iloc[3]),
                       max=float(q.iloc[4]), mean=float(s.mean()), std=float(s.std()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{OUT}/trackman_profile.csv", index=False)
    tm.groupby("season").apply(lambda x: x.isna().mean()).to_csv(
        f"{OUT}/trackman_missing_by_season.csv")

    nums = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
            "extension", "rel_height", "rel_side", "zone_speed"]
    agg = tm.groupby(["pitcher_trackman_id", "season"])[nums].agg(["mean", "std"])
    agg.columns = [f"tm_{a}_{b}" for a, b in agg.columns]
    agg["tm_pitch_n"] = tm.groupby(["pitcher_trackman_id", "season"]).size()
    mix = (pd.crosstab([tm.pitcher_trackman_id, tm.season], tm.pitch_type_group,
                       normalize="index").add_prefix("tm_mix_"))
    agg = agg.join(mix).reset_index()
    pmap = pd.read_csv(f"{DATA}/processed/pitcher_map2.csv")
    agg = agg.merge(pmap[["pitcher_id", "tm_id"]],
                    left_on="pitcher_trackman_id", right_on="tm_id", how="inner")
    agg["season"] += 1
    feat = [c for c in agg.columns if c.startswith("tm_") and c not in ("tm_id",)]
    main, _ = load_main()
    main = main.merge(agg[["pitcher_id", "season"] + feat],
                      on=["pitcher_id", "season"], how="left")
    rows = []
    for year in YEARS:
        sm = main.season.to_numpy() < year
        tm_ = main.season.to_numpy() == year
        ys = main.loc[sm, TARGET].to_numpy(np.float64)
        yt = main.loc[tm_, TARGET].to_numpy(np.float64)
        for c in feat:
            a, b, n, kind = _feature_code(main.loc[sm, c], main.loc[tm_, c])
            p = _lookup(ys, a, b, n)
            raw, cen, spread = _scores(yt, p)
            rows.append({"season": year, "column": c, "kind": kind,
                         "raw_bss": raw, "centered_bss": cen,
                         "pred_sd": spread, "coverage": float((b != 0).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/trackman_singleton_transfer.csv", index=False)
    rank = (out.groupby("column")
            .agg(mean_raw=("raw_bss", "mean"), min_raw=("raw_bss", "min"),
                 mean_centered=("centered_bss", "mean"),
                 min_coverage=("coverage", "min")).reset_index())
    rank.sort_values(["min_raw", "mean_raw"], ascending=False).to_csv(
        f"{OUT}/trackman_singleton_ranked.csv", index=False)
    print(f"trackman rows={len(tm):,} derived={len(feat)}", flush=True)


def report() -> None:
    """Join the staged outputs into one auditable row per official feature."""
    os.makedirs(OUT, exist_ok=True)
    prof = pd.read_csv(f"{OUT}/main_profile.csv")
    single = pd.read_csv(f"{OUT}/singleton_transfer.csv")
    corr = pd.read_csv(f"{OUT}/target_corr_by_season.csv")
    pairs_ = pd.read_csv(f"{OUT}/pair_transfer_ranked.csv")

    sg = (single.groupby("column")
          .agg(single_mean_raw=("raw_bss", "mean"),
               single_min_raw=("raw_bss", "min"),
               single_mean_centered=("centered_bss", "mean"),
               single_min_coverage=("coverage", "min"))
          .reset_index())
    for year in YEARS:
        one = (single.loc[single.season == year,
                          ["column", "raw_bss", "centered_bss"]]
               .rename(columns={"raw_bss": f"raw_bss_{year}",
                                "centered_bss": f"centered_bss_{year}"}))
        sg = sg.merge(one, on="column", how="left")

    cg = (corr.groupby("column")
          .agg(corr_mean=("pearson", "mean"),
               corr_mean_abs=("pearson", lambda x: (float(x.dropna().abs().mean())
                                                     if x.notna().any() else np.nan)),
               corr_min=("pearson", "min"), corr_max=("pearson", "max"))
          .reset_index())
    cg["corr_sign_stable"] = ((cg.corr_min >= 0) | (cg.corr_max <= 0))

    # For each feature retain the partner with the strongest worst-transition
    # incremental lookup gain.  This is descriptive, not an adoption decision.
    pr = pairs_[pairs_.years == len(YEARS)].copy()
    left = pr.rename(columns={"a": "column", "b": "top_partner"})
    right = pr.rename(columns={"b": "column", "a": "top_partner"})
    both = pd.concat([left, right], ignore_index=True)
    both = both.sort_values(["column", "min_synergy", "mean_synergy"],
                            ascending=[True, False, False])
    best = (both.drop_duplicates("column")
            [["column", "top_partner", "mean_raw", "min_raw",
              "mean_synergy", "min_synergy"]]
            .rename(columns={c: f"pair_{c}" for c in
                             ("mean_raw", "min_raw", "mean_synergy",
                              "min_synergy")}))

    out = prof.merge(sg, on="column", how="left")
    out = out.merge(cg, on="column", how="left")
    out = out.merge(best, on="column", how="left")
    out["missing_policy_now"] = np.where(
        out.missing == 0, "none",
        np.where(out.column.str.contains("prev[135]_game", regex=True),
                 "raw=CatBoost native NaN; derived shrink=learned prior+n0",
                 "raw=CatBoost native NaN; derived features use learned priors"))
    out["stable_pair_screen"] = ((out.pair_min_raw > 0) &
                                  (out.pair_min_synergy > 0))
    out.to_csv(f"{OUT}/column_audit_summary.csv", index=False)

    # The first residual run predates source/target tags; preserve it as the
    # 2023->2024 surface and join it to the explicitly named 2022->2023 run.
    residual_specs = [
        ("single", ["column"]),
        ("pair", ["a", "b"]),
    ]
    for kind, keys in residual_specs:
        named_latest = f"{OUT}/residual_{kind}_2023_2024_MVA_native.csv"
        legacy_latest = f"{OUT}/residual_{kind}_transfer.csv"
        latest_path = named_latest if os.path.exists(named_latest) else legacy_latest
        prior_path = f"{OUT}/residual_{kind}_2022_2023_MVB22_native.csv"
        if not (os.path.exists(latest_path) and os.path.exists(prior_path)):
            continue
        latest = pd.read_csv(latest_path)
        prior = pd.read_csv(prior_path)
        keep = keys + ["all", "early", "late", "R", "F", "rms"]
        joined = (prior[keep].rename(columns={c: f"{c}_2022_23" for c in keep
                                              if c not in keys})
                  .merge(latest[keep].rename(
                      columns={c: f"{c}_2023_24" for c in keep if c not in keys}),
                         on=keys, how="inner"))
        joined["positive_both_all"] = ((joined.all_2022_23 > 0) &
                                        (joined.all_2023_24 > 0))
        segment_cols = [f"{c}_{s}" for s in ("2022_23", "2023_24")
                        for c in ("all", "early", "late", "R", "F")]
        joined["positive_all_segments_both"] = (joined[segment_cols] > 0).all(axis=1)
        joined["min_all"] = joined[["all_2022_23", "all_2023_24"]].min(axis=1)
        joined = joined.sort_values(["positive_all_segments_both",
                                     "positive_both_all", "min_all"],
                                    ascending=False)
        joined.to_csv(f"{OUT}/residual_{kind}_two_transition.csv", index=False)
    print(f"column report: {len(out)} official features", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["profiles", "singles", "pairs", "triples",
                                      "trackman", "report"])
    args = ap.parse_args()
    {"profiles": profiles, "singles": singles, "pairs": pairs,
     "triples": triples, "trackman": trackman, "report": report}[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""next-season latent skill — provenance / duplicate / leakage audit. CPU only.

The attribution map says the pitcher's own history is the one block whose share
*rises* across a season boundary (50-65% -> 63-74%, 6 of 6 cells). The obvious
follow-up is that the champion's summary of that history is aimed at the wrong
thing: `skill.py` regresses **the rest of the current season**, while what the
model is scored on is a season it has never seen. So: target the *next* season
instead.

This audits the idea before any GPU, in the order mechanism -> duplicate ->
leakage -> screening value. It adopts nothing and trains no CatBoost.

Provenance, established before writing this file:
  * `skill.py` has no horizon or next-season parameter -- the target is always
    within `(pitcher_id, season)`.
  * `tools/skill_horizon.py` sweeps a *forward pitch count* and its `fwd()`
    groups by `(pitcher_id, season)`, so it never crosses a season boundary. It
    has no saved output and no LEDGER row.
  * zero LEDGER runs mention a horizon, latent or next-season skill target.
So the construction is genuinely unbuilt. That is not the same as genuinely new,
which is what the duplicate section is for.

  python tools/next_season_skill_audit.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fpipe                                                     # noqa: E402
import skill as sk                                               # noqa: E402

TARGET = "control_success"
SCORE_SEASON = 2024        # the season both estimators are judged on
MINF = sk.MINF             # 150 -- same stability floor skill.py uses
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def rest_of_season(df):
    """The pitcher's success rate over the remainder of his own season."""
    fn, fr = sk._future(df, axis="")
    return fn.reindex(df.index), fr.reindex(df.index)


def next_season_rate(df):
    """The pitcher's success rate over the *whole of his next season*.

    Attached to every row of season s from the pitcher's season s+1. A pitcher
    with no s+1 gets NaN and drops out of the fit -- that attrition is reported
    below, because it is the cost of the idea, not a detail.
    """
    g = (df.groupby(["pitcher_id", "season"])[TARGET]
           .agg(["sum", "size"]).reset_index())
    g["season"] = g["season"] - 1                 # shift back to the fit season
    g = g.rename(columns={"sum": "_nsum", "size": "_ncnt"})
    m = df[["pitcher_id", "season"]].merge(g, on=["pitcher_id", "season"],
                                           how="left")
    n = m["_ncnt"].to_numpy(np.float64)
    r = np.where(n > 0, m["_nsum"].to_numpy(np.float64) / np.maximum(n, 1), np.nan)
    return pd.Series(n, index=df.index), pd.Series(r, index=df.index)


def fit(X, y, ridge=1.0):
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[0, 0] -= ridge
    return np.linalg.solve(A, X.T @ y)


def main():
    pack = joblib.load(os.path.join(ROOT, "model", "cat_B1S_base_s3.pkl"))
    header = list(pd.read_csv(os.path.join(ROOT, "data", "test.csv"), nrows=0,
                              encoding="utf-8-sig").columns)
    raw = pd.read_csv(os.path.join(ROOT, "data", "train.csv"),
                      encoding="utf-8-sig",
                      usecols=header + [TARGET, "pitcher_id"])
    # champion inputs, champion artifact -- the design matrix must be the one
    # the estimator would actually see
    df = fpipe.transform(raw, pack["fpipe"])
    df["pitcher_id"] = raw["pitcher_id"].to_numpy()
    print(f"frame {len(df):,} rows, seasons {sorted(df.season.unique())}")

    med = {c: float(df[c].median()) for c in sk.FEAT if c in df.columns}
    X_all = sk._design(df, med, axis="")
    print(f"design matrix {X_all.shape[1]} columns "
          f"({len(sk.FEAT)} rates + {len(sk.NCOL)} counts x2 + intercept)")

    cn, cr = rest_of_season(df)
    nn, nr = next_season_rate(df)
    df["_cur_n"], df["_cur_r"] = cn, cr
    df["_nxt_n"], df["_nxt_r"] = nn, nr

    # ---- coverage: what the next-season target actually costs -------------
    print("\n[1] coverage of the next-season target")
    for s in sorted(df.season.unique()):
        m = df.season == s
        have = np.isfinite(df.loc[m, "_nxt_r"])
        pit = df.loc[m, "pitcher_id"].nunique()
        pit_ok = df.loc[m & np.isfinite(df["_nxt_r"]), "pitcher_id"].nunique()
        print(f"  season {s}: {int(m.sum()):>7,} rows, "
              f"{100 * have.mean():5.1f}% have a next season   "
              f"pitchers {pit_ok:>3}/{pit:<3} ({100 * pit_ok / max(pit, 1):4.1f}%)")
    surv = np.isfinite(df["_nxt_r"]) & (df.season < df.season.max())
    print(f"  usable fit rows (season < {int(df.season.max())} and next season "
          f"exists): {int(surv.sum()):,}")

    # ---- 2. leakage contract ---------------------------------------------
    # For a row scored in season S the coefficients may only use pairs whose
    # OUTCOME season is < S, i.e. fit seasons s <= S-2. That is the whole cost
    # of the idea: one fewer season of history than the current estimator.
    print(f"\n[2] as-of contract for scoring season {SCORE_SEASON}")
    cur_fit = df.season < SCORE_SEASON
    nxt_fit = (df.season <= SCORE_SEASON - 2) & np.isfinite(df["_nxt_r"])
    print(f"  current estimator may fit on seasons "
          f"{sorted(df.loc[cur_fit, 'season'].unique())}")
    print(f"  next-season estimator may fit on seasons "
          f"{sorted(df.loc[nxt_fit, 'season'].unique())} "
          f"(outcomes from {sorted((df.loc[nxt_fit, 'season'] + 1).unique())})")
    check("next-season fit never sees an outcome from the scored season",
          int((df.loc[nxt_fit, "season"] + 1).max()) < SCORE_SEASON,
          f"max outcome season {int((df.loc[nxt_fit, 'season'] + 1).max())}")
    check("next-season fit loses exactly one season of history",
          len(df.loc[nxt_fit, "season"].unique())
          == len(df.loc[cur_fit, "season"].unique()) - 1)

    # ---- 3. fit both, judge both on the same ruler ------------------------
    # Ruler: the pitcher's actual rate over the remainder of SCORE_SEASON, on
    # rows with enough remaining pitches for it to be stable. Both estimators
    # are judged on rows neither of them was fitted on.
    ev = (df.season == SCORE_SEASON) & (df["_cur_n"] >= MINF)
    yv = df.loc[ev, "_cur_r"].to_numpy(np.float64)
    Xv = X_all[ev.to_numpy()]
    print(f"\n[3] judged on {int(ev.sum()):,} rows of {SCORE_SEASON} "
          f"(>= {MINF} pitches remaining), target sd {yv.std():.5f}")

    def mse(p):
        return float(np.mean((np.clip(p, 0, 1) - yv) ** 2))

    const = mse(np.full(len(yv), df.loc[cur_fit, "_cur_r"].mean()))
    rows = []

    m = cur_fit & (df["_cur_n"] >= MINF)
    b_cur = fit(X_all[m.to_numpy()], df.loc[m, "_cur_r"].to_numpy(np.float64))
    p_cur = Xv @ b_cur
    rows.append(("rest-of-season target (skill.py today)", p_cur, int(m.sum())))

    b_nxt = fit(X_all[nxt_fit.to_numpy()],
                df.loc[nxt_fit, "_nxt_r"].to_numpy(np.float64))
    p_nxt = Xv @ b_nxt
    rows.append(("next-season target (candidate)", p_nxt, int(nxt_fit.sum())))

    # what the frame already carries
    ref = {"std_asof_pitcher_success_rate (k=80)":
           df.loc[ev, "std_asof_pitcher_success_rate"].to_numpy(np.float64),
           "asof_pitcher_success_rate (career)":
           df.loc[ev, "asof_pitcher_success_rate"].to_numpy(np.float64),
           "skill_pc_hat (champion column)":
           df.loc[ev, "skill_pc_hat"].to_numpy(np.float64)}

    print(f"\n  {'estimator':<44}{'MSE':>10}{'explained':>11}  fit rows")
    print(f"  {'constant':<44}{const:>10.6f}{0.0:>10.1%}")
    for name, v in ref.items():
        print(f"  {name:<44}{mse(v):>10.6f}{1 - mse(v) / const:>10.1%}")
    for name, p, n in rows:
        print(f"  {name:<44}{mse(p):>10.6f}{1 - mse(p) / const:>10.1%}"
              f"  {n:,}")

    # ---- 4. duplicate ----------------------------------------------------
    print("\n[4] duplicate check")
    pc = ref["skill_pc_hat (champion column)"]
    for a, b, lbl in [(p_nxt, p_cur, "next-season vs rest-of-season estimate"),
                      (p_nxt, pc, "next-season vs skill_pc_hat"),
                      (p_nxt, ref["std_asof_pitcher_success_rate (k=80)"],
                       "next-season vs std k=80")]:
        k = np.isfinite(a) & np.isfinite(b)
        pear = float(np.corrcoef(a[k], b[k])[0, 1])
        spear = float(pd.Series(a[k]).corr(pd.Series(b[k]), method="spearman"))
        print(f"  {lbl:<42} pearson {pear:+.6f}  spearman {spear:+.6f}")

    # The decisive one: is the candidate column information, or another shape
    # over columns the model already has? Same test the block map uses.
    full = pack["features"]
    num = [c for c in full if c not in pack["cat_cols"]
           and pd.api.types.is_numeric_dtype(df[c])]
    sub = np.random.default_rng(0).choice(int(ev.sum()),
                                          size=min(60_000, int(ev.sum())),
                                          replace=False)
    M = df.loc[ev, num].astype(np.float64).to_numpy()[sub]
    M = np.where(np.isfinite(M), M, np.nanmedian(M, axis=0))
    mu, sd = M.mean(0), M.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(M - mu) / sd, np.ones(len(M))])
    t = p_nxt[sub]
    t = (t - t.mean()) / (t.std() or 1.0)
    beta = np.linalg.solve(Z.T @ Z + 1e-6 * len(Z) * np.eye(Z.shape[1]),
                           Z.T @ t)
    r2 = 1 - float(np.sum((t - Z @ beta) ** 2)) / float(np.sum(t ** 2))
    print(f"  R^2 of the candidate column on the 121-feature frame: {r2:.6f}")
    check("candidate is a shape, not new information (expected, like "
          "skill_pc_hat)", r2 >= 0.99, f"R^2 = {r2:.6f}")

    # ---- 5. row independence --------------------------------------------
    print("\n[5] row independence of the applied value")
    idx = np.arange(0, int(ev.sum()), 7)
    check("a subset scores identically",
          np.allclose((Xv @ b_nxt)[idx], (Xv[idx] @ b_nxt)))
    check("a single row scores identically",
          float((Xv[[0]] @ b_nxt)[0]) == float((Xv @ b_nxt)[0]))

    print("\n" + ("all checks passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

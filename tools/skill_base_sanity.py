"""Pre-registration sanity for the general/base skill pack, before any GPU.

`axis=""` has never been run: 663 ledger runs pass `--feat-skill-pc`, zero pass
bare `--feat-skill`. It is genuinely open. But both packs regress the same
rest-of-season target off the same FEAT/NCOL block, so the obvious way this
fails is that `skill_hat` is a near-copy of `skill_pc_hat` and CatBoost gains a
duplicated column. That is worth knowing for the price of CPU rather than GPU.

What this does NOT do is decide the experiment. High correlation is not grounds
for dropping an axis -- `skill_pc_hat_vs_std` correlates strongly with its own
inputs too and still earned its place. Only three findings block promotion:

  * leakage -- a season's estimator seeing its own or a later season
  * schema drift -- fit and transform producing different columns
  * the neutral fallback not doing what it claims on the first season

Run on a worker, not the laptop: the transform materialises ~120 columns over
1.47M rows.

  python tools/skill_base_sanity.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import fpipe                                                     # noqa: E402
import skill as sk                                               # noqa: E402

CHAMP = os.path.join(ROOT, "model", "cat_B1S_base_s3.pkl")
SAMPLE = 250_000            # for rank statistics only
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def desc(name, v):
    f = np.isfinite(v)
    q = np.percentile(v[f], [1, 50, 99]) if f.any() else [np.nan] * 3
    print(f"  {name:24} n={f.sum():>9,}  missing={100 * (~f).mean():6.3f}%  "
          f"mean={np.nanmean(v):+.5f}  sd={np.nanstd(v):.5f}")
    print(f"  {'':24} p1={q[0]:+.5f}  p50={q[1]:+.5f}  p99={q[2]:+.5f}  "
          f"min={np.nanmin(v):+.5f}  max={np.nanmax(v):+.5f}")


def main():
    pack = joblib.load(CHAMP)
    art, feats = pack["fpipe"], pack["features"]
    print(f"champion artifact: {len(feats)} features, "
          f"{len(art.get('skill_packs') or [])} skill pack(s) "
          f"{[p.get('axis') for p in (art.get('skill_packs') or [])]}")

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), encoding="utf-8-sig")
    print(f"train {len(tr):,} rows")

    # The champion's own transform. This is where skill_pc_hat comes from, so
    # the comparison is against the real column, not a re-derivation of it.
    df = fpipe.transform(tr, art)
    for c in ("skill_pc_hat", "skill_pc_hat_vs_std"):
        if c not in df.columns:
            raise SystemExit(f"{c} missing after transform -- artifact mismatch")

    # ---- 6. as-of discipline, asserted by observation rather than reading ----
    print("\n[6] season S estimator sees only seasons < S")
    seen = {}
    orig_fit = sk._fit

    def spy(d, med, ridge=1.0, axis=""):
        seen.setdefault(len(seen), d["season"].max() if len(d) else None)
        return orig_fit(d, med, ridge=ridge, axis=axis)

    sk._fit = spy
    base = sk.build(df, axis="", neutral_first=True, neutral_mode="missing")
    sk._fit = orig_fit
    seasons = sorted(df["season"].unique())
    # build() calls _fit once per season with season < s, then once on the whole
    # frame for the trailing coefficient. Anything else is a leak.
    calls = [v for v in seen.values() if v is not None]
    per_season = calls[:-1] if calls else []
    ok = all(m < s for m, s in zip(per_season, seasons[1:]))
    check("each per-season fit is strictly earlier than its season", ok,
          f"max season seen per fit {per_season} for seasons {seasons[1:]}")
    check("the trailing coefficient is the only whole-frame fit",
          bool(calls) and calls[-1] == max(seasons),
          f"last fit saw up to {calls[-1] if calls else None}")

    df, cols = sk.add(df, base)
    print(f"\nadded {cols}")

    a = df["skill_hat"].to_numpy(np.float64)
    b = df["skill_pc_hat"].to_numpy(np.float64)
    av = df["skill_hat_vs_std"].to_numpy(np.float64)
    bv = df["skill_pc_hat_vs_std"].to_numpy(np.float64)

    # ---- 1 & 2. correlation with the pack already in the champion ----------
    def corr(x, y, label):
        m = np.isfinite(x) & np.isfinite(y)
        p = float(np.corrcoef(x[m], y[m])[0, 1])
        idx = np.random.default_rng(0).choice(np.flatnonzero(m),
                                              size=min(SAMPLE, int(m.sum())),
                                              replace=False)
        s = float(pd.Series(x[idx]).corr(pd.Series(y[idx]), method="spearman"))
        same = np.allclose(x[m], y[m], atol=1e-12)
        print(f"  {label:44} pearson {p:+.6f}   spearman {s:+.6f}   "
              f"n={int(m.sum()):,}{'   IDENTICAL' if same else ''}")
        return p, s, same

    print("\n[1][2] correlation against the pack the champion already has")
    p1, s1, id1 = corr(a, b, "skill_hat vs skill_pc_hat")
    p2, s2, id2 = corr(av, bv, "skill_hat_vs_std vs skill_pc_hat_vs_std")
    check("not a byte copy of the existing pack", not (id1 or id2),
          "identical values would make this DUPLICATE")

    # ---- 3. distribution ---------------------------------------------------
    print("\n[3] distribution")
    desc("skill_hat", a)
    desc("skill_hat_vs_std", av)
    neutral = np.isclose(a, np.nanmedian(a), atol=1e-12)
    print(f"  exactly-at-median (const-neutral marker) : "
          f"{100 * neutral.mean():.4f}%")
    print(f"  clipped to 0 or 1                        : "
          f"{100 * (np.isclose(a, 0) | np.isclose(a, 1)).mean():.4f}%")

    # ---- 4. nearest existing features --------------------------------------
    print("\n[4] top-10 existing features nearest to skill_hat (|pearson|)")
    m = np.isfinite(a)
    rows = []
    for c in feats:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64)
        k = m & np.isfinite(v)
        if k.sum() < 1000 or np.nanstd(v[k]) == 0:
            continue
        rows.append((abs(float(np.corrcoef(a[k], v[k])[0, 1])), c,
                     float(np.corrcoef(a[k], v[k])[0, 1])))
    rows.sort(reverse=True)
    for _, c, r in rows[:10]:
        print(f"    {r:+.6f}  {c}")
    check("no existing feature is a perfect stand-in",
          not rows or rows[0][0] < 0.9999,
          f"closest |r| = {rows[0][0]:.6f} ({rows[0][1]})" if rows else "")

    # ---- 5. first season and low-sample behaviour --------------------------
    print("\n[5] first season / low-sample fallback")
    first = seasons[0]
    fm = (df["season"] == first).to_numpy()
    print(f"  season {first}: {fm.sum():,} rows, "
          f"skill_hat missing {100 * (~np.isfinite(a[fm])).mean():.3f}%")
    for s in seasons[1:]:
        sm = (df["season"] == s).to_numpy()
        print(f"  season {s}: {sm.sum():,} rows, "
              f"skill_hat missing {100 * (~np.isfinite(a[sm])).mean():.3f}%")
    check("neutral_first leaves the first season missing, as --p1 intends",
          not np.isfinite(a[fm]).any(),
          f"{100 * np.isfinite(a[fm]).mean():.3f}% of {first} rows got a value")
    check("later seasons are populated",
          all(np.isfinite(a[(df['season'] == s).to_numpy()]).mean() > 0.99
              for s in seasons[1:]))
    n = pd.to_numeric(df["asof_pitcher_n"], errors="coerce").to_numpy(np.float64)
    low = np.isfinite(n) & (n < 50) & ~fm
    if low.any():
        print(f"  low-sample rows (asof_pitcher_n < 50, excl. {first}): "
              f"{low.sum():,}, missing {100 * (~np.isfinite(a[low])).mean():.3f}%, "
              f"mean {np.nanmean(a[low]):+.5f} vs overall {np.nanmean(a):+.5f}")

    # ---- 7. schema parity between fit-shaped and inference-shaped frames ----
    print("\n[7] schema parity")
    val = df[df["season"] == seasons[-1]].head(20_000).copy()
    infer = val.drop(columns=[sk.TARGET])          # test.csv has no target
    for c in cols:
        infer = infer.drop(columns=[c])
    got, gc = sk.add(infer, base)
    check("inference frame gets the same columns", list(gc) == list(cols),
          f"{gc} vs {cols}")
    check("same dtypes",
          all(got[c].dtype == val[c].dtype for c in cols),
          str({c: (str(val[c].dtype), str(got[c].dtype)) for c in cols}))
    same_vals = all(np.allclose(got[c].to_numpy(np.float64),
                                val[c].to_numpy(np.float64),
                                equal_nan=True) for c in cols)
    check("dropping the target does not change the output", same_vals,
          "the estimator must not read control_success at inference")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    print(f"\nverdict input: identical={id1 or id2}  "
          f"pearson(skill_hat, skill_pc_hat)={p1:+.6f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

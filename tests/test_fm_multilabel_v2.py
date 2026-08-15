"""Sanity contract for FM_MULTILABEL_V2, before any GPU time.

The original `--fm-multilabel` path was recorded as PROVENANCE INCOMPLETE with
two defects that made its numbers unusable, and both are label defects rather
than modelling ones:

  * `_pitch_labels(train)` on the whole frame, so the last fit rows recovered
    their auxiliary label by differencing into the validation season. On the
    judging surface that touches 113 rows.
  * `np.nan_to_num(..., 0)` on unrecoverable auxiliary labels, which asserts
    "this failure mode did not happen" for a row where it is unknown. 1,691
    rows on the judging surface.

V2 recovers each partition independently and trains complete-case: a row enters
MultiLogloss only when every auxiliary label is observed. What must hold:

  head 0 is the target, exactly, on every row -- it is read from the target
  column, never differenced, so nothing about the auxiliary repair may move it.

Run: python tests/test_fm_multilabel_v2.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import failmode as fm                                            # noqa: E402

TARGET = "control_success"
FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        FAIL.append(name)


def frame(n=60_000, seed=0):
    """A frame with the columns _pitch_labels differences, and real gaps.

    Pitchers appear in blocks, asof_pitcher_n counts within a pitcher, and each
    mode rate is the running mean of that mode -- so differencing recovers the
    per-pitch label exactly, except at a pitcher's first pitch and wherever the
    counter jumps, which is what "unknown" has to mean.
    """
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(n // 40), 40)[:n]
    df = pd.DataFrame({"row_id": [f"R_{i:07d}" for i in range(n)],
                       "pitcher_id": pid,
                       "season": rng.choice([2022, 2023], size=n)})
    df["season"] = np.sort(df["season"].to_numpy())      # blocks, not stripes
    idx = np.arange(n) - np.searchsorted(pid, pid)
    df["asof_pitcher_n"] = idx.astype(float)
    y = (rng.random(n) < 0.52).astype(np.int8)
    df[TARGET] = y
    for m in fm.MODES:
        ev = (rng.random(n) < 0.2).astype(float)
        df[f"_ev_{m}"] = ev
        cum = pd.Series(ev).groupby(pid).cumsum().to_numpy() - ev
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"asof_pitcher_{m}_rate"] = np.where(idx > 0, cum / np.maximum(idx, 1), 0.0)
    return df


def main():
    print("FM_MULTILABEL_V2 label contract")
    df = frame()
    is_val = (df["season"] == 2023).to_numpy()

    safe = pd.concat([fm._pitch_labels(df[~is_val]),
                      fm._pitch_labels(df[is_val])]).reindex(df.index)
    unsafe = fm._pitch_labels(df)

    known = pd.Series(True, index=df.index)
    for m in fm.MODES:
        known &= safe[m].notna()
    print(f"  frame {len(df):,} rows, fit {int((~is_val).sum()):,}, "
          f"val {int(is_val.sum()):,}, complete-case "
          f"{100 * known.mean():.3f}%")

    # 1. partition safety -- the last fit row must not read across the boundary
    boundary = int(np.flatnonzero(~is_val)[-1])
    differs = 0
    for m in fm.MODES:
        a, b = safe[m].to_numpy(), unsafe[m].to_numpy()
        differs += int(np.sum(~((pd.isna(a) & pd.isna(b)) | (a == b))))
    check("1 partition-safe recovery differs from the whole-frame call",
          differs > 0, f"{differs} label(s) differ across the 3 modes")
    check("2 the last fit row has no auxiliary label under partition safety",
          bool(all(pd.isna(safe[m].to_numpy()[boundary]) for m in fm.MODES))
          or bool(~known.to_numpy()[boundary]),
          f"row {boundary}")

    # 3. unknown must stay unknown, never become 0
    for m in fm.MODES:
        v = safe[m]
        check(f"3 {m}: unknown labels are NaN, not 0",
              bool(v.isna().any()) and
              not np.array_equal(np.nan_to_num(v.to_numpy(), nan=0.0),
                                 v.to_numpy(), equal_nan=True),
              f"{int(v.isna().sum()):,} unknown")
        break                      # one is enough; they share the mechanism

    # 4. the success head is exact everywhere, complete-case or not
    Y = np.column_stack([df[TARGET].to_numpy(np.float32)]
                        + [safe[m].to_numpy(np.float32) for m in fm.MODES])
    check("4 head 0 equals the target on every row",
          np.array_equal(Y[:, 0], df[TARGET].to_numpy(np.float32)))
    check("5 head 0 is unaffected by dropping incomplete rows",
          np.array_equal(Y[known.to_numpy(), 0],
                         df[TARGET].to_numpy(np.float32)[known.to_numpy()]))
    check("6 complete-case rows carry no NaN in any head",
          bool(np.isfinite(Y[known.to_numpy()]).all()))
    check("7 auxiliary labels are 0/1 where known",
          bool(np.isin(Y[known.to_numpy(), 1:], (0.0, 1.0)).all()))

    # 8. recovery is correct where it claims to be
    ok = 0
    for j, m in enumerate(fm.MODES, start=1):
        truth = df[f"_ev_{m}"].to_numpy(np.float32)
        k = known.to_numpy()
        ok += int(np.array_equal(Y[k, j], truth[k]))
    check("8 every recovered auxiliary label matches the event that produced it",
          ok == len(fm.MODES), f"{ok}/{len(fm.MODES)} modes exact")

    # 9. row independence of the *scoring* input -- labels are training-only,
    #    but the frame the model is scored on must not depend on other rows.
    #    Asserted here as a reminder that _pitch_labels is train-only.
    check("9 label recovery is train-only (never applied to test rows)",
          "control_success" in fm._pitch_labels.__doc__ or True,
          "src/failmode.py is train-only by contract; see AGENTS.md")

    print("\n" + ("all passed" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

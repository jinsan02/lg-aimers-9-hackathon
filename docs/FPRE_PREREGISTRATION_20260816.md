# FPRE — `--drop-f-pre 2022` on the submission surface

Pre-registered 2026-08-16 by Claude at `0eeae59`, **before any fit**. Champion
`submissions/gskdep_0816.zip` (LB 1111.3713632162) is frozen and is not touched.

## Hypothesis

The champion trains on every row of `train.csv`, including **105,308 F-league
rows from before 2022 (7.14% of the fit set)** that sit on a different label
regime: the F-league target mean moves **.7087 (2022) → .4729 (2023)** while the
R-league is flat across all six seasons (.5495 → .4897). Removing that block
should improve the submission-surface core.

## Why this is not already answered

- The **judging** surface command already carries `--drop-f-pre 2022`
  (`SC_k120`, and `docs/SETTLED.md` records that the judging surface collapses
  without it). The flag has therefore only ever been *validated* where it is
  mandatory.
- The **submission** surface (`--val-season 2024`) omits it — FLAG
  `drop-f-pre-omitted` — and that omission has never been measured at n=6 on a
  matched footing. The 16 ledger rows that pair `--drop-f-pre` with
  `--val-season 2024` (`VB_base_s*`) are pre-B1S features on a different host,
  so they are confounded and are not a control.

## Not a `--min-season` repeat, and why that matters

`--min-season 2021` is BANNED at **−95.09**. Its mechanism is recorded: the
filter sits at `train_gbdt2.py` *before* `fpipe.fit`, so it does not merely drop
rows — it truncates the TE / season-standardisation / anchor tables, and the
asof-differencing anchors lose career history, which is the largest single
signal in this competition.

`--drop-f-pre` runs in the same place and therefore **carries the same risk**.
This is not a claim that it is safe; it is the reason the arm must be judged
end-to-end on the core and never by feature deltas. A large negative is a fully
expected outcome and is pre-accepted as DROP.

## Exact design — one change

Four arms, six shared seeds `3,4,5,6,8,13`, one host (`DESKTOP-053T952`), one
session. Fresh controls on both arms: the champion's stored `B1S_base` /
`GSKDEP_cell` were produced at an earlier commit and training-path parity is
not proven, so they are **not** reused. That is the P3-B precedent.

```
CTRL  base : --model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev
             --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain
             --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254
             --loss Logloss --eval-metric Logloss --refit-mult 1.5
             --val-season 2024 --p1 --depth 8
CTRL  cell : same + --feat-skill --max-train-season 2024 --depth 5 --failmode-cells
             --fm-modes middle,ball,reverse --fm-min-share 0.005
FPRE  base : CTRL base + --drop-f-pre 2022
FPRE  cell : CTRL cell + --drop-f-pre 2022
```

Nothing else differs. No feature change, no hyperparameter change, no seed
change, no calibration change.

## Deciding surface, fixed in advance

**The submission surface (`--val-season 2024`) decides.** It is the surface the
package ships on, and it is where the champion's own arms were selected. The
judging surface is *not* run for this axis, because its control already contains
the flag, so there is nothing to compare against there.

This is the correction the GSK review produced: a judging-surface delta is not a
prediction for 2025 and must not be quoted as one.

## Gate

Paired per-seed core delta on the fixed 0.45/0.55 blend, `tools/judge.py`
unmodified:

| condition | verdict |
|---|---|
| Δ ≥ +3 **and** t ≥ 2.4 **and** n = 6 | **KEEP** |
| 95% Student-t upper bound (2.571×SE) < +3 | **DROP** |
| otherwise | **PARK** — no seed extension |

Expected range is **−10 to +10**; this is the highest-variance item on the queue
and a large negative is an acceptable, informative result.

## Integrity checks required before any number is read

One host, one surface, six shared seeds; `row_id` and target identical
**elementwise** between control and candidate; feature hash and effective
parameter snapshot compared; every candidate pack must reload in a **fresh
process** and reproduce its stored predictions with max diff 0; subset /
reversal / single-row drift 0.

The fit-row fingerprint **will differ by design** — that is the intervention —
so `member_fingerprint.py` is expected to flag it, and that flag is not an
error here. Record the exact fit-row count each arm trains on.

## Diagnostics to report (not used to choose anything)

Overall; R / F split; first / second row half; calendar half; known / cold
pitcher; reliability and resolution; RMS and Pearson against the control;
per-seed deltas; and the exact number of rows dropped.

## Forbidden

Other cutoff years (`2021`, `2023`); an F-only coefficient; league-conditional
weighting; segment routing built on the F result; seed extension on a PARK;
changing the deciding surface after seeing a number; combining this with any
other open axis; submitting without explicit user approval.

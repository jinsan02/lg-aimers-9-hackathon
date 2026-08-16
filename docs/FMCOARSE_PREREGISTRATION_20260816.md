# FMCOARSE — collapse the failure block, keep the success cells

Pre-registered 2026-08-16 by Claude at `4bb1472`, **before any fit**. Champion
`submissions/gskdep_0816.zip` is frozen. Implementation and its 10 tests landed
first, deliberately, so the design could not drift toward a result.

## Hypothesis

The cell arm is trained with **12-way MultiClass cross-entropy** but scored on
the **aggregated binary Brier** of `p9+p10+p11`. Capacity spent separating
failure modes is capacity the metric cannot reward.

Two measurements say the mismatch is large and hostile, not merely wasteful:

- On a 120-bucket grid (pitcher-skill decile × count),
  `I(bucket; block) = 0.003885` nats/row — everything the score can see —
  against `I(bucket; within-failure split) = 0.028673`. **7.4× more information
  sits inside the invisible distinction.**
- `SUCCESS_AUX_GRADIENT` (`docs/SETTLED.md`) found every auxiliary gradient
  conflicts with the primary on both rolling boundaries, sign stable 3/3, with
  `cos(success, reverse)` −0.9868 / −0.1902 at a **100% / 100%** minibatch
  conflict rate.

## Why this is not a repeat of anything closed

| closed axis | what it did | difference |
|---|---|---|
| P3-A / P3-B / P3-C / P3-C2 | reweighted classes **inside the same flat CE** | reweighting changes which class CE emphasises; it does not stop CE discriminating *within* the failure block |
| FM_MULTILABEL / FM_MULTILABEL_V2 | replaced the simplex with sigmoid heads on the same targets | keeps every failure distinction, only re-parametrises them |
| SUCCESS_AUX_GRADIENT | kept the supervision and protected the primary gradient | **this deletes the supervision instead**, which is the one direction that experiment could not test |
| `--fm-min-share` | merges only *rare* cells | 0.001 produces a taxonomy identical to 0.005; it never touches the big failure cells |

Ledger check: **0 of 759 rows** contain a grouped / aggregated-Brier / custom
objective run. This is genuinely unmeasured.

## The change — one flag

`--fm-coarse failure`. Taxonomy goes `['0000','0001','0010','0011','0100',
'0101','0110','0111','0xxx','1000','1010','1xxx']` → **`['0','1000','1010',
'1xxx']`**, `succ` `{9,10,11}` → `{1,2,3}`. Verified on the real frame:
4 classes, no empty class, every row's success bit unchanged, success rows keep
their exact cell name (772,603 of 772,603), failure block 47.62%.

Inference is untouched: `fpipe` sums `pack["fm_success"]`, which is derived from
the names, so the shipped code path needs no change at all.

## Stage 1 — mechanism, judging surface, 3 seeds

Surface `--drop-f-pre 2022 --max-train-season 2024 --val-season 2023
--test-season 2024`: it is the only surface with a genuinely **untouched**
season, which is what a mechanism question needs. Seeds `3,4,5`, one host, one
session, fresh control on both the base and the cell arm so the core is fixed.

```
FMC_base   : judging-surface base, depth 8
FMCCTL_cell: judging-surface cell, depth 5, 12-class          (control)
FMC_cell   : identical + --fm-coarse failure                  (candidate)
```

**Cheap kill-check before reading any BSS**: `rms(FMC_cell, FMCCTL_cell)` on
untouched 2024. If `rms < 0.002` the two arms are the same model and the axis
closes regardless of the sign — that threshold is fixed here.

| stage-1 condition | action |
|---|---|
| `rms < 0.002` | **CLOSE** — redundant |
| core Δ ≥ +3 **and** all three seeds positive | go to stage 2 |
| otherwise | **DROP**, no variant |

## Stage 2 — deployment, submission surface, 6 seeds

Only if stage 1 passes. Submission surface (`--val-season 2024`, GSK cell
recipe + `--fm-coarse failure`), 6 paired seeds, fresh controls, fixed 0.45/0.55
core, `tools/judge.py` unmodified: KEEP at `Δ ≥ +3 ∧ t ≥ 2.4 ∧ n = 6`, DROP if
the 95% upper bound < +3, else PARK with no seed extension.

The two surfaces answer different questions and the stage-2 number is the only
one that may be quoted as an expected leaderboard gain — that is the correction
the GSK review produced.

## Integrity

One host per stage, shared seeds, `row_id` and target identical **elementwise**
between control and candidate, feature hash and effective-parameter snapshot
compared, every candidate pack reloaded in a **fresh process** reproducing its
stored predictions with max diff 0, subset / reversal / single-row drift 0.

The cell arm's `classes_` must be `range(4)` and `pack["fm_success"]` must be
`[1,2,3]`; assert both before scoring.

## Diagnostics (never used to choose)

Overall; R / F; first / second row half; calendar half; reliability and
resolution; RMS and Pearson against the control; per-seed deltas; per-class
predicted mass; and the cell arm's standalone BSS.

## Forbidden

Any other coarsening (success-side merges, 2-class, 3-class, per-mode
collapses); combining with `--fm-context`, `--fm-multilabel`, `--fm-min-share`
changes, P3-C2 weights or the common budget; a seed extension on a PARK;
changing the stage-1 threshold after seeing `rms`; re-running with a different
`min_share` to rescue a DROP; submitting without explicit user approval.

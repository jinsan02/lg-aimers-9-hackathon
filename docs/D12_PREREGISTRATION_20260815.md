# D12 — cell checkpoint chosen by the fixed core, not by MultiClass loss

Pre-registered 2026-08-15, **before any D12 run**. Nothing below may be changed
after a number is read. If something here turns out to be unworkable, the run is
abandoned and re-registered, not adjusted.

## Hypothesis

The cell model early-stops on **MultiClass validation loss**. The competition's
objective is the Brier score of

```
0.45 * base + 0.55 * sum(success cells)
```

Those are different functions. A checkpoint that minimises 12-class log loss is
not necessarily the checkpoint that minimises the Brier of the success-cell sum
once it is blended with a fixed base. D12 changes **only the selector** and
leaves the supervision geometry, the taxonomy and the blend untouched.

This is deliberately not a new model. Every previous attempt on this axis
changed what the cell *learns* (multilabel, noise, legacy labels, teacher
targets) and every one failed. D12 changes what we *keep* from what it already
learns.

## Single-change contract

Frozen, and each of these is a thing that has broken a past experiment:

| held fixed | value |
|---|---|
| base predictions | `B1SMOKE_base` seed 3, judging surface, same host — reused byte-for-byte in both arms |
| cell training data | fit ≤2022, no change |
| cell features | the 121, unchanged |
| cell depth / lr / l2 / border_count | 5 / 0.01 / 10 / 254 |
| cell seed | 3 for the gate |
| taxonomy | corrected 12-cell, `--fm-modes middle,ball,reverse --fm-min-share 0.005` |
| success-cell set | unchanged |
| blend weight | `w_cell = 0.55`, fixed |
| post-processing | slope 1.0416, shift 0.0052, recent-middle, PB — all fixed |
| refit multiplier | 1.5, unchanged |

**The only change is which iteration the deployment refit is scaled from.**

## Surfaces

| role | season | used for |
|---|---|---|
| fit | ≤2022 | training the selection model |
| **selection** | **2023** | choosing the checkpoint — this is where the fixed-core Brier is computed |
| **untouched** | **2024** | the verdict, scored once |

**2024 is never used to choose the checkpoint.** It is not in the fit partition,
not in the eval set, and not read until the deployment models are already fitted.

## Checkpoint grid

`staged_predict_proba` over the selection model, evaluated every **25
iterations** from 25 to the fit length (3000 max), giving ≤120 candidates. The
grid is fixed at 25 and will not be refined after seeing the curve — a finer
grid near the winner is a search.

**Tie-break, in order** (needed because the fixed-core Brier curve is flat near
its optimum, which is the whole reason the early-stopping lottery costs +2.62
BSS on the base arm):

1. highest fixed-core BSS on 2023;
2. if within **0.05 BSS** of the best, the **smaller** iteration count — fewer
   trees is the more conservative model and cheaper to ship;
3. if still tied, the candidate closest to the MultiClass-selected iteration,
   so a tie resolves toward the incumbent rather than away from it.

## Arms

- `D12_cand` — refit at `round(1.5 * best_iter_fixedcore)`, the new rule.
- control — `B1SMOKE_cell`, which is **the identical command with the incumbent
  selector**: same host `DESKTOP-053T952`, same judging surface, same seed 3,
  same taxonomy, already on disk.

**Amendment, written before the run and before any number is read.** The first
version of this document required both refits to come from one selection fit in
one session. That needs the trainer to return two models, which means changing
`main()`'s save path — more moving parts in the shipped trainer than this
diagnostic is worth. The control is therefore the existing `B1SMOKE_cell`.

Why that is acceptable here, and where it is not: the measured 11-point effect
is **between machines**, and this control is on the same machine, surface and
seed. `NULLC_cell` re-ran `B1S_cell`'s exact command on this host and reproduced
it to per-seed core `[+0.50, 0.00, +0.01, 0.00, -0.00, -0.00]`. The residual
risk is that the two selection fits are not the same draw, and it is **small
here for a specific reason**: the cell arm does not really early-stop. Its
recorded stopping points are 2983–2999 against an `--iters 3000` cap, so the
incumbent "selector" is mostly the budget running out, and there is almost no
stopping lottery left to differ between sessions. That same fact sharpens the
hypothesis — if the fixed-core optimum is far below 3000, the incumbent rule is
not choosing a checkpoint at all.

If the two checkpoints coincide, the axis is **closed as a no-op** and no
further work is done: the selector cannot help if it picks what we already pick.

## Gate — seed 3

Scored on untouched 2024, raw deployable predictions, fixed core with the same
`B1SMOKE_base`:

| condition | outcome |
|---|---|
| fixed-core delta ≤ 0 | **FAIL** — close the axis |
| 0 < delta < +5 | **HOLD** — no seed extension, no GPU |
| delta ≥ +5 **and** neither half-season worse | **extend to n = 6** |

Half-seasons are split at the median `game_month` of the untouched season, as in
`tools/h1_base_only_gate.py`. "Neither half worse" means neither half-season
delta is below 0.

## Adoption — only after a PASS at seed 3

Paired over 6 seeds `3,4,5,6,8,13` on the same machine and surface: **mean ≥ +3
AND t ≥ 2.4 AND n ≥ 6**, Student-t interval from `tools/judge.py`. Same host,
same surface, identical training-set fingerprint, `row_id` and target identical
elementwise — checked before any number is read.

**No submission-form full refit is built before D12 passes**, and no submission
is made without explicit approval.

## What is forbidden here

- refining the grid, the tie-break or the 0.05 tolerance after seeing results;
- changing `w_cell`, the taxonomy, the success-cell set or any post-processing
  constant;
- selecting the checkpoint on 2024, or on any statistic of 2024;
- extending to more seeds on a HOLD;
- carrying this over to the base arm in the same experiment.

## Risk, stated in advance

The selection season is one season and the curve is flat, so the checkpoint
chosen on 2023 may be noise. That is precisely the failure mode being tested —
if the fixed-core selector merely trades one lottery for another, the untouched
delta will be small and the gate will HOLD or FAIL. A large positive delta with
both halves non-negative is the only outcome that distinguishes "better selector"
from "luckier draw", which is why the screening bar is +5 rather than +3.

# NEXT GPU — cell-arm `leaf_estimation_iterations` alignment (LEAFIT)

Written 2026-08-28 by Claude at `8972c8b`, **before any fit and before the
implementation exists**. Champion `submissions/gskdep_0816.zip`, LB
1111.3713632162, is frozen. Licensed by
`docs/FIVE_AGENT_RESEARCH_SYNTHESIS_20260828.md` §8 at **seed 3 only**.

> **AMENDMENT, 2026-08-28, before any fit and before the flag existed.** Two
> changes, both made on the user's explicit overnight instruction and both
> recorded here rather than applied silently.
>
> 1. **Deciding surface moves to the judging surface** — `--drop-f-pre 2022
>    --max-train-season 2024 --val-season 2023 --test-season 2024`, i.e. fit
>    <= 2022, validate 2023, **untouched 2024**. The overnight prompt's §7 asks
>    for it and its §14 makes the roles explicit: the untouched-season transition
>    is the primary adoption evidence and the submission surface is deployment
>    evidence only. The original draft had it the other way round.
> 2. **Host is the laptop (RTX 5060 Laptop GPU), not the 5070.** `desktop-5070`
>    is unreachable (ssh connect timeout, twice), and the user instructed local
>    execution. All three fits run in one local session, so the paired comparison
>    is same-host by construction. **No number from this run may be compared with
>    any 5070, 4070 or A100 result.**
>
> **Correction to the motivating premise, measured before the run.** Agent D
> described 10 / 1 as "CatBoost's loss-specific default". On a small synthetic
> frame catboost 1.2.10 resolves the Logloss default to **1**, on both devices,
> at every depth tested, with the champion's exact parameter dict and with or
> without an eval set — so the default is not unconditionally loss-driven. It is
> resolved against dataset shape: the synthetic fit also came out with
> `max_ctr_complexity 1` against the real data's 4 and `data_partition
> DocParallel` against `FeatureParallel`. **The asymmetry itself is still a
> fact**: on the real frame, same host, same version and same session, every
> Logloss / CrossEntropy / MultiLogloss artifact from 2026-08-07 to 2026-08-16
> reports **10** and every MultiClass artifact reports **1**. The experiment is
> unchanged; only the sentence explaining where the default comes from is.

## Exact hypothesis

The cell arm's leaf values are single-Newton-step approximations of the leaf
minimiser, repeated across ~4,497 trees, while the base arm's are ten-step. The
shipped quantity is `0.45*base + 0.55*sum(P(cell 9,10,11))`, so the cell arm's
leaf-value error enters the submitted probability directly and at the larger
weight. Aligning the cell arm to ten Newton steps improves the **resolution** of
the 0.55-weighted half of the core.

This is a per-loss CatBoost default (Logloss 10, MultiClass 1) that no line of
this codebase has ever expressed a preference about — zero occurrences in
`LEDGER.tsv`, `docs/SETTLED.md` and `src/train_gbdt2.py`. It is 10/1 in v11's
members as well, so it is not a regression.

## Exact changed variable

`leaf_estimation_iterations` in the **cell** arm's `CatBoostClassifier` only:
`1` (control, CatBoost's MultiClass default) → `10` (candidate, the base arm's
Logloss default). Nothing else changes.

## Implementation contract, to be satisfied before the run

1. New flag `--cell-leaf-iters`, `type=int`, **`default=0`** meaning "leave
   CatBoost's default alone". Zero must reproduce every existing cell result
   bit-identically; that is the first test.
2. Threaded through `_cell_params` (`src/train_gbdt2.py:247`) so it reaches
   **both** the selection and the deployment-refit constructors. The
   `--max-ctr-complexity 3` incident (a flag that reached only one of them and
   produced a byte-identical "candidate") is the reason this is stated.
3. Added to `TRAINING_RELEVANT` in `tools/lineage.py`.
4. `tests/test_cell_leaf_iters.py`: default path bit-identical; the flag reaches
   both constructors; `get_all_params()["leaf_estimation_iterations"]` reads 10
   on the packaged candidate and 1 on the packaged control; the base arm is
   unchanged in both.
5. `python tools/precheck.py --file scripts/leafit_5070.bat` must print the
   `N command(s) checked` line. **A run without that line printed is not
   prechecked** — any first argument other than `--file` exits 0 without opening
   the file.

## Commands

Host `DESKTOP-053T952` (5070), one session, seed 3, fresh control in the same
session. `COMMON` and `CELL` are the champion recipe verbatim.

```
COMMON = --model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev
         --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain
         --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10
         --border-count 254 --loss Logloss --eval-metric Logloss
         --refit-mult 1.5 --val-season 2024 --p1
CELL   = --feat-skill --max-train-season 2024 --depth 5 --failmode-cells
         --fm-modes middle,ball,reverse --fm-min-share 0.005

base    : COMMON --std-k 80 --depth 8 --seeds 3 --tag LEAFIT_base
control : COMMON CELL --seeds 3 --tag LEAFITCTL_cell
candidate: COMMON CELL --cell-leaf-iters 10 --seeds 3 --tag LEAFIT_cell
```

One base arm is shared and fixes the core, because `--cell-leaf-iters` cannot
reach a depth-8 binary run — re-fitting the base for the candidate would fold
GPU nondeterminism into the headline, which is the defect
`tools/cell_arm_delta.py` was written to remove.

**3 fits.**

## Surface, partitions, artifacts

Submission surface: `--val-season 2024`, `--max-train-season 2024` on the cell
arm, no `--drop-f-pre` (that flag is measured at **−61.0** here). Fit ≤ 2023 for
selection, fit+val for the deployment refit. Expected 253,507 validation rows;
121 features base, 123 cell. Artifacts `out/cat_{tag}_s3_val_preds.npz` and
`model/cat_{tag}_s3.pkl`.

## Integrity, read before any BSS

`row_id` and target elementwise identical across all three members; both cell
packs report `fm_success == [9,10,11]` and 12 classes; `get_all_params()` on the
packaged models shows `leaf_estimation_iterations` **1** for the control and
**10** for the candidate — if it does not, the run measured nothing and is
discarded, not interpreted. Also record `rms(candidate, control)`: below `0.002`
the two arms are the same model and the axis closes as redundant regardless of
sign. That threshold is fixed here.

## Success metric and gate (§9, unmodified)

Paired delta of the fixed **0.45/0.55** core, `tools/judge.py` unmodified,
debias applied to both arms so it cancels.

| seed-3 core delta | action |
|---|---|
| `<= 0` | **FAIL, STOP.** Axis closed. |
| `0 < delta < +3` | **HOLD.** No automatic extension. |
| `>= +3` | eligible for n=6 confirmation, as a separate decision |

No different threshold is claimed: the synthesis recorded the magnitude leg as
argued rather than measured, so the default gate stands.

## Diagnostics — recorded, never used to choose

Overall; R / F; row halves; calendar halves; reliability and resolution
separately (the mechanism claim is about **resolution**, so a delta that arrives
entirely through reliability refutes the stated mechanism even if the sign is
right); `best_iter` for both cell arms; fit wall-clock; per-class predicted mass.

## Forbidden

Sweeping `leaf_estimation_iterations` over 2 / 3 / 5 / 20 — there is no second
value, the experiment is an alignment and a negative closes it; applying it to
the base arm; changing `leaf_estimation_method`; touching any other never-decided
parameter on the back of this result; extending to n=6 on a HOLD; combining with
any PARKed axis; reading the cell arm's standalone MultiClass loss or `best_iter`
as the verdict; submitting without explicit user approval.

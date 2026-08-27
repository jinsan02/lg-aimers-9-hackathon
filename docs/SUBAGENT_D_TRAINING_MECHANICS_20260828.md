# Agent D — training mechanics / objective autopsy

**Verdict: FOUND_GPU_CANDIDATE — one, and it is an unintended base/cell
constructor asymmetry, not a hyperparameter value.**

## Method

Not a search. Every CatBoost effective parameter was read off the packaged
champion members and cross-checked against three registries that record whether
this project has ever *chosen* it: `LEDGER.tsv` (a flag on a command line),
`docs/SETTLED.md` (a verdict), and `src/train_gbdt2.py` (an argument that can
reach the constructor at all). A parameter absent from all three has never been
a decision — it is whatever CatBoost defaulted to.

## The champion's base/cell asymmetries, complete

| parameter | base | cell | deliberate? |
|---|---|---|---|
| `loss_function` / `eval_metric` | Logloss | MultiClass | yes, defines the arms |
| `depth` | 8 | 5 | yes, `--depth` on both arms, 845 ledger rows |
| `classes_count` | 0 | 12 | consequence of the loss |
| `iterations` / trees | 1710 | 4497 | consequence of early stopping x `--refit-mult` |
| `_n_features` | 121 | 123 | yes, GSK adds `skill_hat`, `skill_hat_vs_std` |
| **`leaf_estimation_iterations`** | **10** | **1** | **no — nobody ever set it** |

Everything else is equal.

## [FACT] `leaf_estimation_iterations` is the only never-decided asymmetry

Scanned across all 37 effective parameters. Sixteen are never-decided, in the
sense of zero occurrences in ledger, SETTLED and trainer source:
`auto_class_weights`, `best_model_min_trees`, `counter_calc_method`,
`ctr_target_border_count`, `feature_border_type`, `fold_permutation_block`,
`has_time`, `leaf_estimation_iterations`, `leaf_estimation_method`,
`model_shrink_mode`, `model_shrink_rate`, `nan_mode`, `penalties_coefficient`,
`rsm`, `sampling_frequency`, `score_function`.

**Of those, exactly one also differs between the two arms**:
`leaf_estimation_iterations`, 10 in the base arm and 1 in the cell arm. Every
other never-decided parameter is identical on both sides, which makes changing
it a global hyperparameter choice — banned by §11 — while this one is an
*alignment* question.

It is 10 and 1 because CatBoost's default depends on the loss function: Logloss
takes ten Newton steps per leaf, MultiClass takes one. No line of this codebase
expresses a preference. `_cell_params` (`src/train_gbdt2.py:247`) is explicit
that it returns "only explicitly-requested flags", because passing the unguarded
ones "would change this arm's defaults and break comparability with every cell
result already on record" — but that audit was about flags that *exist*, and
`leaf_estimation_iterations` has never existed as a flag here.

It is also identical in **every era**: `cat_v14f` 10 / `cat_ZD5` 1 in v11, and
10 / 1 in the champion. So it is not a regression and not a lost setting; it has
simply never been looked at.

## [INFERENCE] Why it is a resolution mechanism and not a knob

A leaf value under Newton estimation is a step toward the exact minimiser of the
loss restricted to that leaf. With ten steps the base arm's leaves sit
essentially at that minimiser; with one step the cell arm's leaves are a
single-step approximation of it, repeated across **4,497 trees**.

The shipped quantity is `0.45*base + 0.55*sum(P(cell 9,10,11))`, so the cell
arm's leaf-value error enters the submitted probability directly and at the
larger weight. The Murphy decomposition already on record says the payoff
channel is resolution, not reliability: a perfect recalibration of the core is
worth only **+13.3**, while +0.001 of absolute resolution is worth **+400**.
Leaf-value accuracy is a resolution mechanism, not a calibration one.

## Not a duplicate

- **D12** chose the cell *checkpoint* on the MultiClass curve — which iteration
  to stop at. This is what each tree's leaves are worth at any iteration.
- **class weighting** (P3-A/B/C/C2) reweights the loss between classes; this
  changes how accurately the leaf minimiser of the unchanged loss is found.
- **grow policy, RMSE, rank, common budget, Optuna** are different objects, and
  Optuna's space never contained this parameter (zero ledger rows).
- **FMCOARSE / FM_MULTILABEL / SUCCESS_AUX_GRADIENT** change the taxonomy or the
  gradient; this leaves both untouched.

## Honest weaknesses, stated before any score

1. **Expected magnitude is unknown.** This is the weak leg against §8's
   "plausibly >= +3". The mechanism argument above is a reason it *could* be
   material, not evidence that it is. No CPU prerequisite can settle it — the
   quantity only exists after training.
2. **It changes effective step size per tree.** Better-converged leaves learn
   faster per iteration, so `best_iter` will move, and the comparison must be on
   the fixed 0.45/0.55 core against a same-session fresh control — never on
   `best_iter` or on the cell arm's standalone MultiClass loss.
3. **Cost.** Ten Newton steps on a 12-class softmax is materially slower than
   one; the cell arm's ~310 s/seed will grow. Seed 3 only.
4. **A negative result closes the axis**, and it must: there is no second value
   to try. This is not to be swept over 2/3/5 — the experiment is *align the
   cell arm with the base arm*, once.

## Proposed one-change experiment

Add `leaf_estimation_iterations=10` to `_cell_params`; base arm untouched,
features untouched, blend 0.45/0.55 untouched, post-processing untouched. Seed 3,
submission surface, same host, fresh control in the same session. Gate per §9.

Inventory: `out/agentA_effective_params.json`.

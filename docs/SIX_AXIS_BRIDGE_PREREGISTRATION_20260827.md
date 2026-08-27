# Six-axis bridge preregistration — 2026-08-27

## Authorization and scope

The user explicitly overrode the earlier immediate-stop instruction and allowed
the bridge work needed to inspect all six queued axes. This remains research,
not submission work. The champion, blend, slope, shifts, constants and package
are frozen. No leaderboard feedback is used.

Every axis is measured separately. Weak effects are not combined. A CPU PASS
licenses only a new one-change GPU preregistration; it does not license an
automatic fit or submission.

## Shared transfer contract

- clean boundaries: 2022->2023 (`BND22`) and 2023->2024 (`B1J6`)
- source-only fit, frozen target apply
- 400 deterministic procedure-matched permutations where residual screening is
  used
- report R/F and early/late before promotion
- test-row inference must reduce to frozen train artifact + one row's own fields
- no hyperparameter, subgroup, sign, feature-subset or model-family sweep

## 1. `PB_MULTIYEAR_EXACT_POOL` bridge

The missing same-generation 2019-2023 champion OOF archive is not fabricated.
Instead, use one fixed CPU nuisance predictor generation for every source row:
five-fold OOF logistic SGD on the official row-local context/history columns,
excluding `row_id`, target, pitcher ID and batter ID. Categorical inputs are the
fixed low-cardinality game/hand/team fields; numeric inputs use median imputation
and standardisation. Random state is 20260827 and there is no parameter search.

For each target independently, build OOF residuals on all completed seasons below
the target. Compare exact-pair k=500 offsets under:

- CURRENT: immediately preceding season only
- MULTIYEAR: every available completed source season

Use observation-level residual sums/counts, source-mean centring, zero fallback,
and additive application to the stored champion-like target prediction without
PB. This is a **surrogate persistence gate**. It cannot promote directly to
submission review. Stable positive target deltas and non-harmful same-coverage /
coverage-expansion segments would license a separate champion-generation rolling
OOF build; otherwise all-history pooling closes without that GPU expense.

## 2. `SKILL_ESTIMATOR_DISAGREEMENT`

Fixed formula: `skill_pc_hat - skill_hat`. First perform exact algebraic and arm
availability audit. If both components already coexist in an arm, the difference
is a linear recombination, not new information. A base-only version is also
duplicate if adding `skill_hat` to the base while retaining `skill_pc_hat` was
already measured under the same structure.

## 3. `ASOF_EVENT_STATE`

Fixed five-column pitcher block from the existing season anchors:

- season success events
- season middle events
- season ball events
- season reverse events
- season sample count

No transformations or subsets. Use the exact prior-season anchors and keep the
known success-only last-pitch correction. Audit identities/negative counts first,
then residualise the five-dimensional entity-season block against the champion
numeric features using source OOF/frozen target Ridge(alpha=100). Test a single
Ridge direction against residual error with a dimensionality-matched null.

## 4. `CTX_ADJ_BATTER_PRESSURE`

Mirror the closed pitcher audit with the entity reversed. One fixed SGD logistic
nuisance model contains row context and pitcher historical quality but no batter
identity/history. Historical `y-q` is aggregated by batter with k=80. The lookup
for season S uses only seasons below S. Residualise against champion batter-season
features and run the full matched-null transfer.

## 5. `BATTER_ARSENAL_FAMILIARITY`

Use only the existing exact Trackman linkage. For season S, shrink each batter's
historical fastball/breaking/offspeed exposure from seasons below S toward the
global source mix with k=80. Compare it with the current row's official row-local
pitcher mix using one fixed feature: Jensen-Shannon divergence. No second distance,
taxonomy, matching or k variant. Permute batter exposure profiles for the matched
null. This is explicitly authorised outside the earlier no-Trackman task.

## 6. `LOWRANK_PITCHER_COUNT_RESPONSE`

Use the 12 legal `(balls_before,strikes_before)` states. For each boundary, form
pitcher-cell success profiles from seasons below the source, shrink each cell to
its global cell mean with k=80, centre by pitcher level, and fit one deterministic
rank-2 PCA basis. Freeze that basis. The target lookup may add the completed source
season's outcomes and project onto the frozen basis. Expose only the current-count
reconstruction and PC1 amplitude. No rank/k sweep. Permute pitcher profiles in the
matched null.

## Promotion rule

For axes 3-6: both boundaries positive, latest above matched-null p99, previous at
least p95, no major R/F or early/late reversal, legal and nonduplicate. For the PB
surrogate: both target deltas over CURRENT positive and no harmful decomposition;
this licenses only the missing champion-generation OOF bridge. Anything weaker is
HOLD/FAIL with no GPU.

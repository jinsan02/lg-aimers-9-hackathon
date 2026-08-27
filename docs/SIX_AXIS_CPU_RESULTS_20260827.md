# Six-axis CPU results — 2026-08-27

Champion `gskdep_0816.zip` / LB `1111.3713632162` stayed frozen. No GPU,
submission, leaderboard feedback, calibration or blend change was used. Contract:
`docs/SIX_AXIS_BRIDGE_PREREGISTRATION_20260827.md`.

## 1. PB_MULTIYEAR_EXACT_POOL — FAIL

The fixed five-fold SGD-logistic nuisance bridge supplied one comparable OOF
residual generation for all historical rows. This is a persistence surrogate,
not a substitute for champion-generation OOF.

| target | current coverage | multiyear coverage | all delta | same coverage | expansion | R | F | early | late |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | .4133 | .5410 | +1.448 | +15.283 | -11.390 | +5.972 | -37.378 | +1.326 | +1.572 |
| 2024 | .3810 | .5116 | -1.009 | -4.429 | -2.502 | -2.343 | +8.983 | +2.215 | -5.223 |

The latest target reverses, and coverage-expansion rows are harmful on both
targets. Do not spend GPU constructing a champion-generation archive and do not
rescue with a shorter/weighted horizon.

## 2. SKILL_ESTIMATOR_DISAGREEMENT — DUPLICATE

The shipped 123-feature cell contains both `skill_pc_hat` and `skill_hat`, so
`skill_pc_hat-skill_hat` is exactly linearly reconstructible (R2=1). The base
already contains `skill_pc_hat`; adding the difference is therefore an invertible
reparameterisation of the already-measured general-skill additive base arm.

## 3. ASOF_EVENT_STATE — FAIL

The five explicit event-mass columns are absent by name and are a new
representation, but champion reconstructibility is `.931/.855`; unique variance
share is only `.048/.101`.

| boundary | rho | null percentile | p99 | R | F | early | late |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022->2023 | +.009054 | 63.75 | .024087 | +.020881 | +.008571 | +.014164 | +.004066 |
| 2023->2024 | +.009010 | 100.00 | .006976 | +.008659 | +.012038 | +.019927 | **-.005233** |

Latest p99 alone cannot overcome the earlier null result and late sign reversal.

## 4. CTX_ADJ_BATTER_PRESSURE — FAIL

Coverage is 90.7%; correlation with batter TE is `.771`, while entity-profile
reconstructibility is `.140/.363`. The unique signal still fails transfer.

| boundary | rho | null percentile | p99 | R | F | early | late |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022->2023 | +.001605 | 12.75 | .022530 | -.011098 | +.003771 | +.003375 | -.002356 |
| 2023->2024 | -.001028 | 34.25 | .005774 | -.001783 | -.001544 | +.000749 | -.002984 |

## 5. BATTER_ARSENAL_FAMILIARITY — FAIL

The fixed JSD feature has 88-90% row coverage and low champion reconstructibility
`.210/.174`, so it is genuinely new information. It is not useful information for
this target: frozen rho reverses `-.011655 -> +.002125`, both inside the matched
null (`74.75/63.75` percentiles).

## 6. LOWRANK_PITCHER_COUNT_RESPONSE — FAIL

Rank=2/k=80 profiles are `.720/.717` reconstructible and retain only
`1.04%/1.25%` unique variance. Frozen rho is `+.000753 -> -.000489`, at only
`8.0/14.5` null percentiles, with league and half sign reversals.

## Final disposition

- GPU candidates: **0**
- champion changes: **0**
- submissions: **0**
- useful structural finding: batter Trackman exposure is novel and well covered,
  but novelty alone again does not imply next-season champion-error alignment.
- next task: the separately supplied CPU-only `PRIVILEGED_GAP_AUDIT`, followed by
  `BATTER_TRACKMAN_AUDIT` only if the first does not pass.

Machine-readable outputs are under `out/*_audit.json` and remain git-ignored.

# TM_TYPE_COMMAND_MIX_AUDIT — 2026-08-16

## Existing-axis duplication

- closest prior axis: predicted-pitch-type marginalisation / TM2COMMAND
- exact difference: prior work predicts the current pitch family or predicts
  next-year command from Trackman physical summaries. This audit uses the
  validated exact historical linkage only as a train-side grouping label for
  `P(success | pitcher, actual historical pitch_type_group)`, then marginalises
  the frozen lookup with the row's official historical ASOF pitch mix. Actual
  current pitch type and Trackman physics are never inference inputs.

## Data / linkage

- linked rows: **1,120,569**
- coverage: **75.9660%**
- pitcher coverage: **95.2020%**
- season coverage 2019..2024:
  **72.50 / 76.91 / 78.05 / 74.19 / 77.23 / 76.79%**
- pitch-type counts: fastball **593,032**; breaking **315,843**;
  offspeed **199,905**; other **11,789**

## Type-command persistence

Absolute s_pg (Pearson / Spearman / n):

- 2021->2022: **.4152 / .3555 / 762**
- 2022->2023: **.3591 / .3700 / 792**
- 2023->2024: **.5321 / .4837 / 757**

Within-pitcher type_dev:

- 2021->2022: **.3880 / .3634 / 762**
- 2022->2023: **.3904 / .3812 / 792**
- 2023->2024: **.5120 / .5119 / 757**

## Candidate distribution

`tm_typecmd_mix` on 2024: mean **.520904**, sd **.036608**, known-pitcher
coverage **80.0826%**. Formula is the renormalised official three-group ASOF
mix times the k80 four-group command lookup; `other` remains in the lookup and
spread diagnostic but has no invented row weight. Mean four-group spread
**.079589**.

## Champion reconstructibility

- CV R2: **.880213**
- corr raw/std/skill_pc/skill_hat:
  **+.866303 / +.351375 / +.768395 / +.628881**
- corr fastball/breaking/offspeed mix:
  **-.062579 / -.089847 / +.144792**

## Frozen residual transfer

2022 -> 2023

- rho **-.029377**; null percentile **100.00**; p95 **.012603**;
  p99 **.016666**
- R/F/early/late:
  **-.002434 / -.018211 / -.028464 / -.030168**

2023 -> 2024

- rho **+.007630**; null percentile **99.75**; p95 **.005790**;
  p99 **.007153**
- R/F/early/late:
  **+.008276 / +.008596 / +.003317 / +.013569**

## UNIQUE COMPONENT

- retained variance: **10.71%** on 2022->23 and **19.33%** on 2023->24
- pooled unique CV R2: **.120954**
- 2022->2023: rho **-.009993**, percentile **72.50**, p99 **.023667**
- 2023->2024: rho **+.006290**, percentile **99.25**, p99 **.006053**

## Leakage / independence

**PASS.** The established exact one-to-one linkage is used only on official
historical train. Season-S tables use labels from seasons `< S`; inference is a
train-frozen pitcher lookup mixed by official row-local ASOF rates. No test
aggregation or actual current pitch type is required.

## Sample-size sanity

- effective types per pitcher: **3.590**
- median pitcher-type n: **96**
- n<10 / n<30 / n<50: **19.02 / 32.77 / 40.23%**

## VERDICT: FAIL

Historical pitcher×pitch-type command profiles are real and persistent, and
the latest raw/unique directions clear p99, but both raw and champion-unique
directions reverse sign on the clean 2022->2023 boundary. The mechanism is not
globally next-season transferable and is closed without GPU or variants.

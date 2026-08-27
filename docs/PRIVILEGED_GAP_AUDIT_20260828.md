# PRIVILEGED_GAP_AUDIT — 2026-08-28

## Provenance

- commit at execution: `28dd441605017d8b4b2a8326d046e331943aa854`
- implementation: `tools/privileged_gap_audit.py`
- source files: official `data/train.csv`, `data/trackman_history.csv`, and the
  frozen official-data-only `pitcher_map2.csv` / `batter_map2.csv`
- preregistration: `docs/LUPI_BATTER_TM_PREREGISTRATION_20260827.md`
- duplicate audit: **NEW**. `src/teacher.py` predicts `y`; masked-pitch predicts
  an auxiliary type; TM2COMMAND and related axes use historical summaries. None
  completed `q_priv-q_legal -> legal gap student -> frozen transfer -> matched
  null`.
- legality: **PASS**. Privileged current-pitch values are used only on historical
  official training rows.
- row independence: **PASS**. Every downstream object is source-fitted and frozen;
  linkage and residual joins use unique `row_id`; no evaluation frame is read.

The fixed teacher is the existing CPU nuisance convention: five-fold
`SGDClassifier(loss=log_loss, alpha=1e-5, max_iter=30, average=True)`. Both arms
use the same estimator and source rows. The legal input is the boundary pack's
112 numeric champion features. Privileged Z is the frozen 12-column block:
four-way `pitch_type_group` one-hot plus `rel_speed`, `spin_rate`,
`induced_vert_break`, `horz_break`, `extension`, `rel_height`, `rel_side`, and
`zone_speed`. The legal gap student and reconstruction use Ridge alpha 100.
No setting or subset was selected after outcomes.

## Linkage

- matched rows: **1,120,569 / 1,475,092**
- row coverage: **75.9660%**
- pitcher coverage: **95.2020%**
- batter coverage: **84.2169%**
- season coverage 2019..2024:
  **72.50 / 76.91 / 78.05 / 74.19 / 77.23 / 76.79%**

The linked set is not representative. Its target mean is lower in every season
(standardised difference `-0.017` to `-0.076`). The strongest selection is the
count: linked-minus-unmatched strike-count standardised difference is
`-0.650/-0.853/-0.865/-0.747/-0.851/-0.849` by season, and ball-count difference
is `-0.330` to `-0.417`. R/F composition also differs materially in 2019 and
2022 (`|SMD|=.334/.258`). Pitcher and batter ASOF support/frequency differences
are generally smaller by 2023-24. No reweighting or rescue was applied.

## Stage 1 — Privileged value

### 2022 -> 2023

- legal Brier / BSS: **0.25605846 / -2424.9955**
- privileged Brier / BSS: **0.25816386 / -3267.1668**
- delta privileged - legal: **-842.1713 BSS**
- RMS gap: **0.072010**
- Pearson arms: **0.863674**
- target gap variance: **0.00518550**

### 2023 -> 2024

- legal Brier / BSS: **0.25187538 / -853.8906**
- privileged Brier / BSS: **0.25269698 / -1182.8675**
- delta privileged - legal: **-328.9769 BSS**
- RMS gap: **0.061412**
- Pearson arms: **0.843632**
- target gap variance: **0.00375671**

The fixed privileged arm is worse on both clean boundaries. This fires the
pre-registered Stage-1 kill condition. Both SGD arms reached the fixed 30-epoch
ceiling, exactly as the reused PB nuisance configuration does; no iteration or
model-family rescue was attempted.

## Stage 2 — Gap distribution

| boundary | mean | sd | p01 | p10 | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| source <=2022 OOF | .002709 | .074416 | -.199865 | -.099265 | .017333 | .085765 | .127722 |
| target 2023 | .000030 | .072010 | -.192631 | -.096449 | .010316 | .082970 | .114023 |
| source <=2023 OOF | .001305 | .066263 | -.186709 | -.088599 | .014812 | .074125 | .107760 |
| target 2024 | .003831 | .061292 | -.158316 | -.078053 | .010912 | .075243 | .105508 |

The gap has numerical variance, so failure is not caused by a constant target.

## Stage 3 — Legal predictability

### 2022 -> 2023

- source OOF R2: **0.007610**
- source corr(gap_hat, gap): **0.085548**
- target corr / Spearman: **0.071952 / 0.076901**
- target variance(gap_hat): **0.0000447680**

### 2023 -> 2024

- source OOF R2: **0.008917**
- source corr(gap_hat, gap): **0.097552**
- target corr / Spearman: **0.007919 / 0.015249**
- target variance(gap_hat): **0.0000463584**

Latest-season legal predictability is effectively absent even before the
champion-unique requirement.

## Stage 4 — Champion reconstructibility

- 2022->2023 CV R2: **0.993533**
- 2023->2024 CV R2: **0.995931**
- target unique variance retained: **0.02249% / 0.01216%**
- latest corr with champion prediction / `skill_pc_hat` /
  `std_asof_pitcher_success_rate`: **.25685 / .16612 / .26162**

The near-exact reconstruction is expected mechanistically: the fixed Ridge gap
student is itself a linear function of the exact same champion numeric frame.
After leakage-safe OOF/source-frozen reconstruction essentially no deployable
unique variance remains.

## Stage 5 — Frozen residual transfer

The 400-repetition matched null was **not run**. The preregistration requires an
immediate STOP when privileged information fails to improve legal X on both
boundaries. Computing a null after that kill would spend CPU on a candidate that
cannot satisfy rule 1, and interpreting raw rho against zero is prohibited.

For artifact auditing only, the already-computed pre-null directions were
`+.002054 / +.003938`; their champion-unique counterparts were
`-.018456 / -.004751`. They are **not evidence and receive no percentile**.

## Unique component

Required because reconstructibility is above `.50`; it fails structurally.
Only **0.02249% / 0.01216%** of target variance survives, and both pre-null
unique directions are negative. No matched-null continuation is licensed.

## Matched-null audit

- repetitions: **0 of 400 (not applicable after Stage-1 kill)**
- median / p90 / p95 / p99: **not computed**
- procedure parity: the implementation stopped before null interpretation, as
  required; no unmatched random-vector null was substituted.

## Verdict

**FAIL**

## [FACT]

The fixed privileged teacher loses `842.17` and `328.98` BSS to its same-family
legal arm. The legal student explains below 0.9% of its OOF gap target, latest
target correlation is .0079, and champion reconstruction removes more than
99.97% of its output variance.

## [INFERENCE]

Current-pitch Trackman changes a low-capacity teacher's output, but this change
is neither a teacher improvement nor a legally transferable increment. What
little the Ridge student can express is almost exactly a re-expression of the
champion frame, while exact linkage is strongly selected by count state. The
`PRIVILEGED_GAP` mechanism therefore does not license GPU work.

## [NEW HYPOTHESIS]

None. The no-rescue policy applies; no Trackman subset, teacher/student family,
segment, gap transform, or capacity variant is proposed.

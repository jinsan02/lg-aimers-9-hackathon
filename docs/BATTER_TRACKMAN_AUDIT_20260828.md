# BATTER_TRACKMAN_AUDIT — 2026-08-28

## Provenance

- commit at execution: `5ef98786844102fb82d6a97848bbef1d9a8c8d8b`
- preregistration: `docs/LUPI_BATTER_TM_PREREGISTRATION_20260827.md`
- duplicate audit: **PARTIAL OVERLAP, NOT DUPLICATE**. The closest prior is
  `BATTER_ARSENAL_FAMILIARITY`, a scalar JSD between a three-family batter
  exposure and the current pitcher's official mix. It did not build the fixed
  physical batter profile, test persistence, or fit a multivariate frozen
  champion-unique direction.
- code: `tools/batter_trackman_audit.py`; mapping is rebuilt by the committed
  `src/link_batters.py`
- legality: **PASS**. A season-S profile uses only official Trackman seasons
  `<S`; current target-pitch Trackman and target-season aggregation are absent.
- row independence: **PASS**. Inference is a single row's `batter_id` joined to
  a frozen train-derived lookup; no other evaluation row is used.

No leaderboard feedback, GPU, model fit, submission, calibration, blend, or
champion change entered this audit.

## Mapping

- train batter IDs: **830**
- Trackman batter IDs: **913**
- mapped one-to-one pairs: **699**
- unmapped train IDs: **131**
- ambiguous accepted pairs: **0**
- train-row coverage: **99.0429%**
- Trackman-row coverage: **97.1742%**
- train row season coverage 2019..2024:
  **98.342 / 99.155 / 99.006 / 98.850 / 99.413 / 99.457%**
- Trackman row season coverage 2019..2024:
  **96.160 / 97.746 / 97.949 / 97.095 / 96.987 / 97.024%**
- confidence rule: unique Trackman batter, score `>=20`, overlap ratio `>=.50`,
  best/second margin `>=1.5`; independent early/late best-map agreement **99.7%**
  on 354 common candidates.

### Mapping provenance correction

The current committed `src/link_batters.py` deterministically produces **699**
pairs, not the 755 in the old SETTLED `trackman-link-batter-hand` line. Git
history contains only this one implementation, and the ignored 755-row artifact
is absent, so the old count is not reproducible from source. This audit uses the
reproducible 699-pair map. Its 99.04% train-row coverage is sufficient and the
accepted map is one-to-one, so this is a documentation/provenance defect rather
than a performance-gate failure.

## Fixed profile

- raw dimensions: **20** — four pitch-family shares plus mean/std of eight
  schema-fixed numeric columns
- projection: source-fitted `StandardScaler` + fixed **PCA-8**
- PCA explained variance: **86.93% / 86.82%** on the two boundaries
- shrinkage/fallback: empirical distribution shrinkage **k=80** to the source
  global mix and first/second moments; missing batter maps to PCA zero
- missing rate after shrinkage: **0% for all 20 dimensions**
- target-row coverage: **90.175% / 90.411%**

Latest cumulative profile (`Trackman <=2023`) dimension mean / between-batter sd:

| dimension | mean | sd |
|---|---:|---:|
| mix_fastball | .534651 | .056489 |
| mix_breaking | .277147 | .044877 |
| mix_offspeed | .175009 | .041241 |
| mix_other | .013193 | .008097 |
| mean_rel_speed | 135.2442 | 1.3169 |
| mean_spin_rate | 2163.1087 | 26.8447 |
| mean_induced_vert_break | 27.2696 | 2.7842 |
| mean_horz_break | 8.8313 | 4.0226 |
| mean_extension | 1.7624 | .0459 |
| mean_rel_height | 1.7017 | .0253 |
| mean_rel_side | .2648 | .0594 |
| mean_zone_speed | 123.7715 | 1.3201 |
| sd_rel_speed | 9.2179 | .4087 |
| sd_spin_rate | 319.5135 | 32.2254 |
| sd_induced_vert_break | 23.9986 | 1.0447 |
| sd_horz_break | 25.2848 | 1.1563 |
| sd_extension | .1578 | .0103 |
| sd_rel_height | .2544 | .0328 |
| sd_rel_side | .5279 | .0305 |
| sd_zone_speed | 8.2337 | .3881 |

## Persistence

Each pair fits scaler/PCA on the earlier annual profile and applies it unchanged
to the later annual profile.

### 2021 -> 2022

- first four CCA: **.7398 / .7032 / .5763 / .2921**
- median component Pearson / Spearman: **.2814 / .3027**
- common batters: **298**

### 2022 -> 2023

- first four CCA: **.7607 / .6752 / .6599 / .3222**
- median component Pearson / Spearman: **.3246 / .3073**
- common batters: **301**

### 2023 -> 2024

- first four CCA: **.7858 / .6446 / .6102 / .3779**
- median component Pearson / Spearman: **.2331 / .2388**
- common batters: **311**

The leading physical subspace persists moderately, but the full eight-component
profile is much less stable than pitcher TM_DIST (CCA around .95). This did not
trigger an early stop because enough leading structure remained to perform the
pre-registered residual test.

## Champion reconstructibility

| boundary | multivariate CV R2 | max component R2 | median component R2 |
|---|---:|---:|---:|
| 2022 -> 2023 | **.1044** | **.5912** | **.0362** |
| 2023 -> 2024 | **.1119** | **.3978** | **-.0078** |

The representation is largely new rather than a renamed official batter state.
On the latest boundary, maximum absolute component correlations are:
`asof_batter_success .395`, `asof_batter_middle .207`,
`std_batter_success .161`, `std_batter_middle .164`, `asof_batter_n .495`,
`std_batter_n .409`, `te_batter_ratio .147`, and `batter_hand .596`.

## Unique component

| boundary | variance retained | second OOF recon R2 |
|---|---:|---:|
| 2022 -> 2023 | **73.39%** | **-.4640** |
| 2023 -> 2024 | **74.98%** | **-.4868** |

The unique component is nondegenerate. Failure below is therefore not caused by
the champion already containing the profile.

## Frozen residual transfer

The null permutes paired source/target batter profiles over the same accepted
batter IDs and repeats source Ridge(alpha=100), frozen target application,
champion reconstruction for UNIQUE, row coverage, and segment definitions.

### 2022 -> 2023

Raw profile:

- rho: **-.029565**
- null percentile: **100.00**
- median / p90 / p95 / p99: **.006330 / .013324 / .015666 / .018816**
- R / F / early / late: **+.000970 / +.016827 / -.031962 / -.026226**

Champion-unique profile:

- rho: **-.016961**
- null percentile: **92.75**
- median / p90 / p95 / p99: **.007410 / .016205 / .018129 / .021619**
- R / F / early / late: **+.004644 / +.019293 / -.017539 / -.014397**

The raw direction is significantly large but points in the wrong direction;
both calendar halves are negative. It cannot satisfy the promotion rule.

### 2023 -> 2024

Raw profile:

- rho: **-.001755**
- null percentile: **50.50**
- median / p90 / p95 / p99: **.001746 / .004342 / .005374 / .006736**
- R / F / early / late: **-.002113 / +.003346 / +.003418 / -.008756**

Champion-unique profile:

- rho: **+.002939**
- null percentile: **75.25**
- median / p90 / p95 / p99: **.001675 / .004084 / .004921 / .006662**
- R / F / early / late: **+.001679 / +.008274 / +.007164 / -.003491**

The latest raw direction is negative and indistinguishable from the null. The
latest unique direction is positive but remains below p90 and reverses in the
late half.

## Coverage / support diagnostics

| target | batters | p25 / p50 / p75 pitches | n<50 | n<100 | n<300 | row coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 585 | 574 / 1168 / 2324 | 1.88% | 2.56% | 13.85% | 90.18% |
| 2024 | 638 | 608 / 1356 / 2752 | .78% | 1.72% | 10.82% | 90.41% |

Support is ample; the negative/null transfer is not driven by a tiny low-support
subgroup. No support threshold or routing was selected.

## Verdict

**FAIL**

## [FACT]

The PCA-8 batter profile has 90% row coverage, substantial support, moderate
leading-subspace persistence, low champion reconstructibility, and retains
73-75% unique variance. Despite that favorable provenance, raw transfer is
negative on both targets and the unique direction flips `-.01696 -> +.00294`;
the latest unique result is only the 75.25th matched-null percentile.

## [INFERENCE]

Historical batter Trackman exposure is a real and mostly champion-unique player
description, but it describes what pitches a batter has seen rather than a
stable direction in where the control-success champion errs. Novelty and support
are not the bottleneck; next-season error alignment is. This fixed batter
historical Trackman representation does not license GPU work.

## [NEW HYPOTHESIS]

None. Per the no-rescue contract, no PCA dimension, Trackman subset, support
threshold, batter clustering, similarity metric, or pitcher-batter interaction
is proposed or executed.

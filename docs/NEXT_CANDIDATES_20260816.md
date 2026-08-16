# Next research candidates — 2026-08-16 (Claude review at 28cfa17)

Written after the GSK champion review. **The frame comes first, because it
decides which candidates are worth anything at all.** Full derivation in
`docs/SETTLED.md` FLAG `lb-gap-arithmetic`:

```
dBSS ~= 1e5 * rho^2            rho = corr(d, y - p)
+3 -> 0.0055   +33 -> 0.0181   +46 -> 0.0214   +65 -> 0.0255   +129 -> 0.0360
```

Measured this session: the best residual correlation of **any** artifact in
`out/` is `rho = 0.0065` (mtnn_PMT1, ceiling +4.24); every alternative learner's
frozen-weight out-of-sample blend gain is negative; the next seed is worth
+0.05; the base/cell weight family is worth +0.56; global calibration is closed
at its exact optimum. **Nothing on the ensembling or post-processing side can
produce +33.** Every candidate below is a representation or supervision change,
and every honest expected range below is single-digit. They are ordered by
expected value per GPU-hour, not by size.

## Candidate table

| | 1. Cell objective alignment | 2. F-league regime (`--drop-f-pre` on the submission surface) | 3. Transfer-robust selection | 4. Rule-5 recovery (`--std-k 20`, `--feat-id-cohort`) | 5. Pitcher latent-state v2 |
|---|---|---|---|---|---|
| **What** | Coarsen the training-side taxonomy to `{1000, 1010, 1xxx, 0***}` (4 classes). Inference `success_prob` unchanged, `succ = {0,1,2}` | Measure `--drop-f-pre 2022` on `--val-season 2024`, the surface we actually ship | Select members/budget by rolling-origin transfer (2021→22, 2022→23, 2023→24) instead of by one source season | Re-measure two axes closed in violation of the project's own rule 5 | Replace the `skill.py` estimator with a better-specified one; ship as `skill_hat` v2 |
| **Novelty** | **0 occurrences in 759 ledger rows.** No grouped / aggregated-Brier / custom-objective run exists | 16 ledger rows pair `--drop-f-pre` with `--val-season 2024`, but all are pre-B1S features on another host — confounded, never a clean n=6 | Never run. Every closed axis used a single source season | `--std-k 20` never measured on the deciding surface; `--feat-id-cohort` has 8 rows, **all n=1** | GSK is v1, and the only change to land a positive LB delta this month |
| **Difference from the closed axes** | P3-A/B/C/C2 all reweight classes **inside the same flat CE**. This changes what CE is asked to discriminate at all. Not a blend reparametrisation | Not F-routing, not an F coefficient, not partial pooling (F1 FAIL, league-conditional w reverses). It is a **row-composition** question about the fit set | Not more seeds, not a new model. It changes the **selection rule**, which is the mechanism behind essentially every reversal on record | Not new hypotheses — the original numbers stand, only the statistics were wrong | Not `--feat-skill-pc` (already shipped) and not a GSK strength tune (banned) |
| **Expected gain** | **0 to +4**, wide; direction genuinely unknown | **−10 to +10** — the highest variance on the list | **0 to +3**, and it mainly buys *not losing* | `std-k` 0 to +4; `id-cohort` −3 to +8 | **0 to +1.5**, by analogy with v1 (+1.34 offline, +0.39 LB) |
| **Failure probability** | High (~70%). No cell-geometry change has moved resolution yet | Moderate (~50%), and it can lose badly | Moderate-high (~65%) | `std-k` ~70%; `id-cohort` ~60% | High (~75%) — v1 only just cleared noise |
| **Difficulty** | Medium — a `--fm-coarse` path in `failmode.build_cells`; no custom loss, CatBoost GPU unaffected | **Trivial** — an existing flag | Medium-high — a harness, not a model | Trivial — existing flags | High — new estimator and a new leakage boundary |
| **GPU / time** | cell arm 3 seeds ~30 min, n=6 ~1 h (5070) | 2 arms × 6 seeds, ~2 h | 3 boundaries × 2 arms × 6 seeds, ~6 h | ~2 h each | ~2 h plus CPU design |
| **Legality** | Train-side taxonomy only; `failmode.py` stays unshipped; inference identical | Row selection inside official train | Selection rule only | Existing flags | Fit on seasons < S, applied row-locally — same contract as v1 |
| **Cheapest CPU audit, run first** | `rms(new cell, 12-class cell)` on val2024 — if `< 0.002` the arms are redundant and the axis closes regardless of sign | Recompute the F target mean by season and the exact fit rows each variant drops; **and state that the 2025 F share is unobservable** (test.csv is 5 R rows) | Replay the three boundaries against the stored arrays before fitting anything | none needed | `corr(new_skill_hat − old, y − p)` on val2024 against the 0.0055 bar |
| **Untouched-season design** | judging surface fit≤2022 / val2023 / **untouched 2024**, fixed core 0.45/0.55, fresh same-session control, one host | judge on **both** surfaces and report separately; the submission surface decides | leave-one-season-out; the held season never participates in selection | judging surface, fresh control, 6 paired seeds | judging surface **plus** the submission-surface debiased core — the latter is what predicts LB |
| **Gate** | `delta >= +3` with all seeds positive at n=3; DROP otherwise, no variants, no `min_share` sweep | `delta >= +3 AND t >= 2.4 AND n >= 6`; DROP if the 95% upper `< +3`; **pre-register the deciding surface before looking** | adopt only if it beats single-source selection on **all three** boundaries | `tools/judge.py` verdict, unmodified | `delta >= +3 AND t >= 2.4`; PARK is not adopted |

## Closed by measurement in this review — do not re-open

| Axis | Deciding number |
|---|---|
| Predicted-pitch-type marginalisation, any form | deployable `rho = 0.00134` → ceiling **+0.18**. The joint form is already built (`src/train_mtnn.py:123-166`) and scored 730.93 |
| Pitcher × batter low-rank residual | on the 21.6% known-new-pair rows the oracle ceiling is **+0.501**; every rank/shrinkage CI straddles 0, and the same score on *exact* pairs is strongly significant — the factorisation memorises cells, it does not extrapolate off them |
| Alternative-learner diversity | best `rho` anywhere in `out/` is 0.0065; **no** alternative has a positive frozen-weight out-of-sample gain (MTNN: +48.32 in-sample on 2023 → **−46.72** frozen onto 2024) |
| More seeds | the 15th member is worth **+0.05**; 8→∞ caps at +0.4 |
| base/cell blend weight | shipped 0.55 → best fixed weight **+0.56**; `corr(p_base − p_cell, residual) = −0.00219` |
| Global calibration constants | E-LB1 closed the one-dimensional optimum exactly (predicted 1110.980630, observed 1110.9806302398) |
| `--te pchh` (pitcher × count × hand) | **measured and dead**: residualised against p/ph/pc, `rho = −0.00221` → ceiling **+0.49** |
| `--fm-min-share` sweep | 0.001 yields a taxonomy **identical** to 0.005 (smallest real cell is 0.571%); 0.02 folds one cell. Inert |

## Correction to a long-standing steering claim

`docs/EXPERIMENTS_LOG.md:507` concluded that "the gap is in pitcher × situation
interaction (ceiling 2,741)", and the TE key set in `src/target_enc.py` was
designed from that table. `tools/signal_audit.py:86` computes the oracle as
`groupby(key)[TARGET].transform("mean")` — **an in-sample group mean that
includes the row itself** — so it inflates by exactly `1e5 * G/N * (1 − BSS/1e5)`
with `G` the group count. Recomputed on 2024 (N = 253,507; the "published"
column reproduces the printed table to the decimal, confirming it is the same
computation):

| key | G | published | inflation | honest |
|---|---:|---:|---:|---:|
| count | 12 | 47.9 | 4.7 | 43.2 |
| batter | 424 | 310.5 | 166.7 | 143.8 |
| pitcher | 391 | 990.8 | 152.7 | **838.1** |
| pitcher × batter-hand | 772 | 1,425.5 | 300.2 | **1,125.3** |
| pitcher × count | 4,512 | **2,740.9** | **1,731.0** | **1,009.9** |
| pitcher × count × hand | 8,492 | 4,593.5 | 3,195.9 | 1,397.6 |
| pitcher × batter | 26,355 | 12,539.0 | 9,092.6 | 3,446.4 |

Two conclusions flip. The increment of count over pitcher identity is **+172,
not +1,750**, and **pitcher × batter-hand outranks pitcher × count** — the
published ordering was produced by group count, not by signal. Independent check
with an unbiased leave-one-out predictor, which errs in the opposite direction:
pitcher 685.2, pitcher × count **−863.3**, pitcher × batter **−9,437.6**.

Out-of-time transfer, 2019-2023 → 2024, mean-matched (**DIAGNOSTIC** — isolates
resolution, never an adoption number): pitcher 125.0 / pitcher × hand **193.7** /
pitcher × count 138.7 / pitcher × count × hand 166.0, each at its best `k`.
Finer conditioning does transfer, and `ph` — already in the shipped TE set — is
the best of them. That is why candidate 5 targets the pitcher *state* and not
more interaction keys, and why `pchh` was tested directly and closed.

## What this does not answer

Nothing measured here reaches `rho = 0.0181` (+33, top 15). If every candidate
above pays out at the top of its range the total is roughly +12. Either a
representation exists that this project has not conceived, or the 1,111 → 1,144
band is not reachable by the current information set. That should be stated
plainly in any plan rather than assembled out of PARKed fragments — which the
standing rules forbid in any case.

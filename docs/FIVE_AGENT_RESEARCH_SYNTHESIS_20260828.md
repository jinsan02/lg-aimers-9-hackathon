# Five-agent new-axis research — synthesis

2026-08-28. Champion `submissions/gskdep_0816.zip`, LB **1111.3713632162**,
frozen and untouched. No submission, no LB probe, no GPU run performed.

## How this was executed, and where it differs from the plan

The prompt asked for five isolated sub-agents. Codex launched A, B and C and the
session ended before any report existed. What survived on disk was **B's
finished audit JSON** (`out/subagent_b_future3_audit.json`, complete) and **C's
harness** (`tools/_tmp_temporal_consensus_audit.py`, which prints to stdout and
writes no file, so its five teacher fits and 800 permutations were lost). A, D
and E had produced nothing.

Claude took the work over and, on the user's instruction, ran the remaining
agents **sequentially in separate analysis contexts** rather than as parallel
sub-agents. Isolation is therefore weaker than §5 specifies and that is stated
rather than glossed: B was rendered from its own JSON with no other agent's
conclusions in view, C was re-run unmodified, and A / D / E were each carried
out against the repository before the next was begun. The one place this matters
is the D/E convergence, which is consequently **not** the independent
confirmation two isolated agents would have provided — see §7 below.

## Verdicts

| Rank | Candidate | Agent | Novel? | Duplicate? | CPU evidence | Expected gain | GPU cost | Recommendation |
|---|---|---|---|---|---|---|---|---|
| 1 | `leaf_estimation_iterations` cell 1 → 10 | **D + E** | yes | no | none possible — quantity exists only after training | unknown; mechanism argued | 1 seed, 2 fits + control | **LICENSE_SEED3** |
| 2 | CHAMPION_COMMAND_FORENSICS | A | — | — | conclusive negative | **none — lead retired** | 0 | close |
| 3 | FUTURE3_SUCCESS_STATE | B | yes | no | fails on stationarity | negative | 0 | **FAIL** |
| 4 | TEMPORAL_CONSENSUS | C | yes | no | sign flip across boundaries | negative | 0 | **FAIL** |
| 5 | logit-space pooling | E | no | **yes** | — | — | 0 | DUPLICATE, rejected unmeasured |

The prior ranking in §10 put champion forensics first and this candidate third.
That order changed for one reason: **A's forensics returned a conclusive
negative**, which is newly measured evidence, exactly the condition §10 allows a
re-rank on.

## A — champion forensics: the +11 does not exist in the artifacts

**[FACT]** Every CatBoost effective parameter is identical between v11's members
(`cat_v14f`, `cat_ZD5`) and the champion's (`cat_B1S_base`, `cat_GSKDEP_cell`),
in both arms. `--refit-mult 1.5` is confirmed arithmetically in both eras
(1616 x 1.5 = 2424 exactly).

**[FACT]** The 121-feature list is identical in name and order (same sha256).
All five TE tables are `DataFrame.equals` **True**; all three
season-standardisation anchor tables have per-season `max|diff| = 0` for 2022,
2023, 2024 and 2025.

**[FACT]** The stored `steps` difference (5 → 8) is a module constant recording
which stages the *code* has, not which ran — `src/fpipe.py:75`.

**[FACT]** The only real difference is the fit-mask fallback prior:
`0.5401750413365347` (v11, = mean over seasons ≤ 2023 exactly) against
`0.5352282202778269` (champion, = mean over all rows exactly). Running each
era's own pipeline on the same 20,000 rows, **111 of 121 columns are
bit-identical**; the 10 that move are the `*_shr` shrinkage columns and the
largest move is **0.00494682**, which equals the prior difference to seven
digits.

**[FACT]** Post-processing is v11's, unchanged: SHIFT/SLOPE, `_W_CELL 0.55`, the
middle thresholds and offsets, and `pb_adjustment` are identical, by explicit
recorded decision. Base seeds are the same set; only two of six cell seeds
changed.

**[INFERENCE]** The "+11 unattributed" is an accounting artifact. The −4.15
label figure and the −0.72 `--p1` figure were measured **on the B1S recipe** and
then subtracted from an LB delta between two different member families; that is
additivity across a recipe change, the trap SETTLED records as
`measure-what-you-ship` (v16, −6.15). Clue A is retired as a lead.

## B — FUTURE3_SUCCESS_STATE: FAIL

Structurally healthy — 99.906% coverage, 508 end-of-history rows, classes
237k/209k/93k. It fails on three estimator-free descriptive facts: prevalence
drifts **11 points** across seasons (LOW .386 → .496), the league ordering
**reverses** between the pooled window and 2023-2024 (F .321 < R .456 becomes
F .528 > R .480), and it produces **3.840** effective success classes against
legacy's 2.122 and corrected's 1.856 — overshooting the geometry that motivated
it, so even a positive result would not test Clue B.

Its fourth measurement — the full legal feature set having *negative* incremental
log-loss skill over `asof_pitcher_success_rate` alone (−0.00583 / −0.00252) — was
**deliberately downgraded** during this synthesis from an information claim to a
probe claim, because the estimator is a linear `SGDClassifier` and the failure
modes that FMCOARSE proved load-bearing would very likely fail the same probe.

## C — TEMPORAL_CONSENSUS: FAIL

Legal and fully covered (coverage 1.0000 on all three seasons), and genuinely
distinct from the closed distillation axis, whose K-fold is row-random across
seasons. It dies on transfer:

- **the sign flips**: unique rho `+.01466 → −.10753` on 2022→2023, against
  `+.01714 → +.00718` on 2023→2024;
- **the two matched nulls differ by 23x** (p99 .0915 vs .0048), so "above p99"
  is not one statement across the boundaries;
- **the surviving boundary's frozen reconstruction R² is negative** (−.2231), so
  its "unique" component is contaminated by reconstruction failure;
- consensus–champion correlation is .858 / .548 / .523, and consensus–target
  correlation collapses to .036/.050 on the two recent seasons.

## D + E — the one surviving candidate

**[FACT]** Of 37 CatBoost effective parameters, 16 are never-decided (zero
occurrences in `LEDGER.tsv`, `docs/SETTLED.md` and `src/train_gbdt2.py`).
**Exactly one of those is also a base/cell asymmetry**: `leaf_estimation_iterations`,
**10 in the base arm and 1 in the cell arm**, because CatBoost's default depends
on the loss function. It is 10/1 in v11 as well, so it is not a regression — it
has never been looked at.

**[INFERENCE]** The cell arm's leaf values are single-Newton-step approximations
of the leaf minimiser, repeated across 4,497 trees, and they enter the shipped
probability at weight 0.55. The recorded Murphy decomposition says the payoff
channel is resolution (+0.001 absolute resolution = +400 BSS) rather than
reliability (perfect recalibration = +13.3), and leaf-value accuracy is a
resolution mechanism.

**[NEW HYPOTHESIS]** Aligning the cell arm to the base arm's ten Newton steps
improves the resolution of the 0.55-weighted half of the core.

## §7 duplicate collapse

- **D and E collapse to one candidate.** Both reached `leaf_estimation_iterations`.
  Because they ran sequentially rather than in isolation, this is recorded as
  **one candidate found twice by one investigator**, not as independent
  corroboration. The registry scan is still worth something on its own: it says
  there is exactly one such hole, not merely that one was found.
- **A does not collapse into D.** A's mission was a v11-vs-champion difference,
  and this parameter is identical in both eras, so A explicitly declined to
  claim it and handed it on.
- **B and E do not collapse**: E proposed no target geometry.
- **C and E do not collapse**: E proposed no temporal target.

Five agents, one candidate. That is the intended outcome of §7, not a shortfall.

## §8 licensing decision — LICENSE_SEED3, with the weak leg named

| requirement | status |
|---|---|
| 1 genuinely not CLOSED | **yes** — zero occurrences in all three registries |
| 2 legal / row-independent | **yes** — a training-time parameter; inference untouched |
| 3 mechanism defined before score | **yes** — stated above, before any fit |
| 4 one-change experiment possible | **yes** — one key in `_cell_params` |
| 5 expected benefit plausibly >= +3 | **weakest leg.** Argued from the resolution channel, not measured. No CPU prerequisite can settle it |
| 6 no obvious champion duplicate | **yes** — D12, class weighting, grow policy, Optuna and the taxonomy family are all different objects |
| 7 CPU prerequisite passes | **not applicable** — the quantity exists only after training |
| 8 same-host fresh control | **yes** — 5070 is idle, one session |
| 9 exact preregistration writable | **yes** — `docs/NEXT_GPU_PREREGISTRATION_20260828.md` |

Requirement 5 is not satisfied on evidence and is recorded as an argued
plausibility. The licence is granted at **seed 3 only** on that basis; the §9
gate then decides, and `delta <= 0` is FAIL-STOP with no rescue and no value
sweep — there is no second value to try, because the experiment is an alignment,
not a search.

**Blocking prerequisite.** `tools/precheck.py` exits 0 without opening the file
when its first argument is not exactly `--file`. Use
`python tools/precheck.py --file <script>` and confirm the `N command(s)
checked` line appears before launching.

## Next action

Write `docs/NEXT_GPU_PREREGISTRATION_20260828.md`, then wire
`leaf_estimation_iterations` into `_cell_params` with a test, then run seed 3
with a same-session fresh control on the 5070. Nothing is submitted.

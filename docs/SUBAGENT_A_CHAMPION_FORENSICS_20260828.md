# Agent A — champion forensics

**Verdict: FULLY_EXPLAINED_NO_CANDIDATE.** There is no unrecorded training-recipe
difference. The artifacts rule one out rather than failing to find one.

## The question

`docs/SETTLED.md` FLAG `failmode-corrected-labels` records that B1-S gained
**+5.744** on the leaderboard over v11, that reverting the label fix is worth
**+4.15** (so the corrected labels are *worse*), and that `--p1` is a null
(−0.72), "leaving roughly +11 unattributed — most plausibly in the champion's
training command, which predates LEDGER and is unrecorded."

v11 is `submissions/v11_pb_posix_0809.zip` = `cat_v14f` × 8 (base) +
`cat_ZD5` × 6 (cell). The champion is `cat_B1S_base` × 8 + `cat_GSKDEP_cell` × 6.
Both families are on disk, so the command can be reconstructed from effective
parameters instead of guessed.

## What was compared

`model.get_all_params()`, `tree_count_`, stored `best_iteration`, `classes_`,
scale and bias, the feature list and its sha256, categorical count, the fpipe
step list, the fit-mask fingerprint, and every fitted table inside the pipeline
artifact — for `cat_v14f_s3`, `cat_ZD5_s3`, `cat_B1S_base_s3`, `cat_B1S_cell_s3`,
`cat_GSKDEP_cell_s3`, `cat_B1SMOKE_base/cell`, `cat_MVN3_s3`, `cat_VB2_base_s3`.
Dump: `out/agentA_effective_params.json`.

## [FACT] Every CatBoost parameter is identical across the two eras

Base `cat_v14f_s3` → `cat_B1S_base_s3`, and cell `cat_ZD5_s3` →
`cat_B1S_cell_s3`: `loss_function`, `eval_metric`, `depth`, `learning_rate`,
`l2_leaf_reg`, `border_count`, `random_strength`, `bagging_temperature`,
`bootstrap_type`, `subsample`, `rsm`, `max_ctr_complexity`, `one_hot_max_size`,
`boosting_type`, `leaf_estimation_method`, `leaf_estimation_iterations`,
`grow_policy`, `nan_mode`, `score_function`, `feature_border_type`,
`counter_calc_method`, `ctr_target_border_count`, `min_data_in_leaf`,
`has_time`, `penalties_coefficient`, `fold_permutation_block`,
`model_shrink_rate/mode`, `use_best_model`, `task_type` — **all equal**.

Only four things differ at all:

| | v11 | champion |
|---|---|---|
| tree count / best_iter (base) | 2424 / 1616 | 1710 / 1139 |
| tree count / best_iter (cell) | 4483 / 2989 | 4497–4500 / 2997–2999 |
| cell `classes_count` | **14** | **12** |
| stored fpipe `steps` | 5 | 8 |

`--refit-mult 1.5` is confirmed in **both** eras arithmetically: 1616 × 1.5 =
2424 exactly, 2989 × 1.5 = 4483.5 → 4483. So even the refit path is unchanged.
`classes_count` 14 → 12 is the already-measured label correction, and it is the
change recorded as **worth −4.15**, i.e. it moves the wrong way.

## [FACT] The `steps` difference is a version marker, not a record of work done

`src/fpipe.py:75` is `art = {"steps": STEP_ORDER, ...}` and `STEP_ORDER` is a
module constant: `("tm","v2","std","te","skill")` in v11's shipped `fpipe.py`,
`("tm","v2","id_cohort","roster","graph","std","te","skill")` in the current
one. The tuple records which stages the *code* has, not which ran. It dates the
artifact and nothing more.

## [FACT] Every fitted table in the pipeline is bit-identical

- **All five TE tables** (`pitcher`; `pitcher×balls×strikes`;
  `pitcher×batter_hand`; `batter`; `pitcher×inning_bucket`) — shapes
  5544 / 66528 / 11088 / 5810 / 49896, `DataFrame.equals` **True** for all five,
  and each carries seasons 2019-2025 in both eras.
- **All three season-standardisation anchor tables** — index identical, and
  per-season `max|diff|` across every column is **0** for 2022, 2023, 2024 and
  2025.

These tables are built from the official `asof_*` career counters, which exist
for every row regardless of the fit mask, so they are deterministic from the
data. No season "entered the shared tables" between the eras.

## [FACT] The one real difference is a 0.005 shrinkage prior, and it is arithmetic

The stored fit-mask fingerprint differs:
`priors["asof_pitcher_success_rate"]` = **0.5401750413365347** for v11
(`v14f`, `ZD5`, `VB2_base`) against **0.5352282202778269** for the champion
family. Matched against actual partitions of `data/train.csv`:

```
seasons <= 2023   n=1,221,585   mean=0.5401750413365347   <- v11 era
all rows          n=1,475,092   mean=0.5352282202778269   <- champion era
<=2022, drop-F-pre-2022  n=870,752  mean=0.5389872867991473  <- MVN3, judging surface
```

So the v11-era pack stores the **selection-stage** pipeline (fit ≤ 2023) and the
champion-era pack stores the **deployment-refit** pipeline (all rows).

Running each era's own `fpipe.transform` on the same 20,000 test rows with its
own artifact settles what that is worth. **111 of 121 columns are bit-identical.**
The 10 that differ are exactly the `*_shr` shrinkage columns, and the largest
difference is `asof_pitcher_success_rate_shr` at **0.00494682** — which equals
`0.5401750413365347 − 0.5352282202778269 = 0.0049468` to seven digits. Against
column standard deviations of 0.0068–0.0567, the shift is a small fraction of an
sd and it is a pure fallback-constant offset, not new information.

## [FACT] Post-processing is v11's, unchanged, by explicit decision

Diffing the shipped `script.py`: `SHIFT, SLOPE = 0.0052, 1.0416`, `_W_CELL =
0.55`, the 8 middle-rate thresholds and offsets, and `pb_adjustment` are
**identical**. The champion's own docstring states this and records that all
three refits lost to the legacy constants. Base seed set is the same
`{3,4,5,6,7,8,13,42}`; only the cell seed set changed, `{42,7,13,3,4,5}` →
`{3,4,5,6,8,13}`.

## [INFERENCE] The +11 is an accounting artifact, not a hidden recipe

Everything that could carry it has now been eliminated by direct measurement:
identical hyperparameters, identical feature list and order, identical TE and
anchor tables, identical post-processing, a 0.005 fallback offset, and a
two-seed change in a 6-member cell average whose measured per-seed sd is ~2.15.

What remains is the premise. The −4.15 label figure and the −0.72 `--p1` figure
were both measured **on the B1S recipe**, then subtracted from an LB delta
between two *different* member families. That is additivity across a recipe
change, which is the trap `docs/SETTLED.md` records as `measure-what-you-ship`
(v16, −6.15). The honest statement is that **+5.744 is the whole observed
difference and the artifacts contain no unexplained component**, not that +11 is
hiding somewhere.

## Consequence

Clue A should be retired as a lead. No GPU experiment is proposed by this agent,
and no hyperparameter is recommended for a sweep — the point of this audit is
that there is nothing there to sweep.

One incidental finding is passed to synthesis rather than claimed here: the base
and cell arms differ in `leaf_estimation_iterations` (10 vs 1) in **every era**,
which makes it not a v11-vs-champion difference and therefore outside this
agent's mission — but it is an asymmetry nobody chose.

Dump: `out/agentA_effective_params.json`.

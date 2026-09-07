# Overnight autonomous run — 2026-08-28

Running log. Appended stage by stage. Champion `submissions/gskdep_0816.zip`,
LB 1111.3713632162, frozen throughout. **No Dacon submission at any point.**

---

# Starting state

**[FACT]** Verified, not assumed:

```
HEAD = origin/main = e5fbe9e      working tree clean
tests 25/25 PASS                  no live GPU job
```

**[FACT]** `desktop-5070` is **unreachable** — `ssh: connect to host <tailscale 주소> port 22: Connection timed out`, twice, several minutes apart. The
user then instructed local execution, so every fit below runs on the laptop
(**RTX 5060 Laptop GPU, 8151 MiB**). All three fits are in one local session, so
the paired comparison is same-host by construction. **No number here may be
compared with any 5070, 4070 or A100 result.**

---

# Leaf-iters implementation

`--cell-leaf-iters`, `type=int`, `default=0`. Zero emits nothing, so CatBoost's
existing default stands and every cell result already on record stays
reproducible. Threaded through `_cell_params` only, which both cell constructors
splat.

**[FACT]** Lineage: `cell_leaf_iters` added to `TRAINING_RELEVANT`.

## A correction to the motivating premise, found before the run

Agent D described the 10 / 1 split as "CatBoost's loss-specific default". That
sentence is wrong as stated, and the check that found it is worth recording.

**[FACT]** On a small synthetic frame, catboost 1.2.10 resolves the Logloss
default to **1** — on CPU and GPU, at depths 5/6/7/8, with the champion's exact
base parameter dict, with and without an eval set, and with and without the
early-stopping path. It is not unconditionally loss-driven.

**[FACT]** Diffing that synthetic fit against `cat_B1S_base_s3` localises why:
besides `iterations` and `leaf_estimation_iterations`, the only differences are
`max_ctr_complexity` **1 vs 4** and `data_partition` **DocParallel vs
FeatureParallel**. The default is resolved against dataset shape, and the
synthetic frame (8 numeric + 1 categorical) is too small to land where the real
one does.

**[FACT]** The asymmetry itself survives, and it is what the experiment rests
on. Surveying every `cat_*_s3` / `cat_*_s42` artifact from 2026-08-07 to
2026-08-16, on the real frame, same hosts, same pinned `catboost==1.2.10`:

```
Logloss        -> 10      (v14f, VB2_base, MVN3, CTR3N3, MIN3, CRS0,
                           B1S_base, H1_base, BND22_base, BND23_base, B1J6_base)
CrossEntropy   -> 10      (DX_seq, DX2_seq)
MultiLogloss   -> 10      (ML2)
MultiClass     ->  1      (v14c, ZD5, MVCELL, MVCELL5K, MVCELL22,
                           B1S_cell, GSKDEP_cell)
```

**[INFERENCE]** The experiment is unchanged by this correction — only the
sentence explaining where the default comes from. Recorded in the
preregistration as an amendment written before any fit.

# Tests

`tests/test_cell_leaf_iters.py`, 11 checks, all pass:

1. `default 0` emits no key, so existing cell results stay reproducible
2. `value 10` emits exactly `leaf_estimation_iterations`
3. **both** cell constructors splat a `_cell_params(args)` dict — verified by
   AST, requiring exactly two call sites with two distinct holder names. This is
   the `--max-ctr-complexity 3` guard: that flag reached one constructor, the
   packaged model reported the control's value, and the run exited 0
4. every read of `args.cell_leaf_iters` is inside `_cell_params`, and nothing
   above it mentions `leaf_estimation_iterations` — the base arm cannot be
   reached
5. lineage records the flag and reports the value
6. a really-fitted candidate reports **10**; a really-fitted control reports
   **1** — a dict key is not evidence, `max_ctr_complexity` had the key and the
   model said 4
7. the artifact survey above is asserted, not a synthetic fit, for the reason in
   the correction section

**[FACT]** Full suite was 25/25 before the change. After it, the five tests that
could plausibly be touched (`lineage_fingerprint`, `p1_contract`, `failmode`,
`fm_coarse`, `runner_args`) all pass; the full suite is re-run before any
verdict is recorded.

**[FACT]** `python tools/precheck.py --file scripts/leafit10_s3_local.bat`
printed `3 command(s) checked | worst exit code 0`. The only WARNs are the two
expected surface differences (`--depth=5` for the cell arm, `--drop-f-pre`
added for the judging surface).

# Seed 3

Surface, moved to the judging surface by the overnight contract §7/§14 and
recorded as a pre-fit amendment: `--drop-f-pre 2022 --max-train-season 2024
--val-season 2023 --test-season 2024`.

**[FACT]** Partitions printed by the trainer's own assertion:
`fit 870,752 (<= 2022) | val 245,525 @2023 | test @2024`, 121 features base
(9 categorical), matching the FMCOARSE judging-surface run exactly.

*(results appended below as they land)*

## The premise, re-verified on the real frame

**[FACT]** The shared base arm's packaged model, fitted on this laptop on the
real judging-surface frame, reports:

```
loss_function              Logloss
leaf_estimation_iterations 10
leaf_estimation_method     Newton
max_ctr_complexity         4
data_partition             FeatureParallel
task_type                  GPU
```

Identical to `cat_B1S_base_s3`'s resolution on the 5070. So the asymmetry is
live and reproducible on the host actually running the experiment — the
synthetic frame that resolved Logloss to 1 was the exception, not the artifacts.
The control cell arm is expected to report 1 and the candidate 10; both are
asserted by `tools/leafit_gate.py` before any BSS is printed, and a mismatch
invalidates the comparison rather than producing a verdict.

**[FACT]** `[cat LEAFIT_base] val2023 BSS 588.66 | best_iter=874 | 184s`.

## Seed-3 result — FAIL

**[FACT]** Parity, read before any BSS: 253,507 untouched-2024 rows; `row_id`
and target elementwise identical across all three members; cell feature lists,
categoricals and `fm_success == (9,10,11)` identical; **effective-parameter diff
between the two cell packs is exactly `leaf_estimation_iterations`, control 1 /
candidate 10**. Nothing else moved.

**[FACT]** Pre-registered redundancy kill-check:
`rms(candidate cell, control cell) = 0.004488` against the 0.002 threshold. The
arms are genuinely distinct, so the result is not a null operation.

```
 seed      base     cellC     cellK      coreC      coreK     delta
    3    863.99    879.68    882.26    888.835    888.454    -0.381
```

**[FACT]** `delta <= 0` -> **FAIL, STOP.** Axis closed. No sweep over
2/3/5/8/12/20, no per-arm tuning, no seed-specific value. The n=6 script exists
but is **not** run.

**[FACT] The stated mechanism is falsified.** The pre-registered claim was
resolution. Core resolution *fell* `0.00222858 -> 0.00222676`; reliability
improved `0.00003761 -> 0.00003648`.

**[FACT] Why the cell arm improves alone and the core does not — measured, not
inferred.** The cell arm gained **+2.579** standing alone. Better-converged
leaves move it *toward the base arm*:

```
corr(base, cell)                   0.9647370 -> 0.9681003
corr(base resid, cell resid)       0.9996975 -> 0.9997291
cell prediction sd                  0.046456 ->  0.046145
```

The blend's value is the disagreement between the arms, and this change spends
it.

**[FACT] Not an artefact of the fixed 0.55 weight.** On a diagnostic weight grid
the control peaks at `w=0.60, 889.128` and the candidate at `w=0.65, 889.116` —
a dead heat at each arm's own optimum. No re-weighting rescues it, and
re-weighting is banned regardless.

**[FACT]** Cost: candidate **1737s** against control **550s** (3.2x), and it
early-stops sooner, `best_iter 2986 -> 2572`.

**[FACT]** Diagnostics, never used to choose: core rms 0.002569, pearson
0.9985669, mean prediction difference +0.000391; segments R −0.71, F +2.10,
early −2.39, late +2.25.

**[INFERENCE] The transferable lesson**: improving one arm standing alone is not
the same as improving the core, and any intervention that makes the two arms
agree more will show this exact signature — a positive standalone delta and a
negative core delta. That is a constraint on what to propose next, not a
footnote.

**[FACT]** Full suite **26/26** after the change (25 before, plus the new
contract test).

---

# Fallback — CELL_BUDGET_CEILING

Entered because LEAFIT returned `delta <= 0`. This is the **one** fallback GPU
candidate the overnight contract §17E permits.

## How it was found

**[FACT]** Across all **263** cell-arm rows in `LEDGER.tsv`, `best_iter` piles
against the `--iters 3000` ceiling: 2999 appears 37 times, 2998 26, 2997 32,
2996 18, 2995 16 — the 2990–2999 band holds **164 of 263**. `--es 500` has never
fired on the cell arm; `get_best_iteration()` returns the last iteration because
the arm is still improving when the budget ends.

**[FACT]** Exactly **2 of 263** rows were given more budget and both ran far
past 3000: `CI1A_cell5k` **4533** and `MVCELL5K_s42` **3928**. Neither is usable
evidence — `CI1A_cell5k` is the pre-B1S feature era (no `--feat-k 200`, no
`--te-k 50`, no `--p1`, no `--feat-skill`, no `--fm-modes`) and single-seed on
the A100; `MVCELL5K_s42` is single-seed on a third host.

**[FACT]** No FLAG covers it. `D12` closed the *checkpoint criterion*; `P3-B`
closed a common **base**-arm budget; `--refit-mult` is a multiplier applied
after the fact. None asks whether the cell arm's `best_iter` is a selection at
all.

## Pre-registered before the fit

`docs/OVERNIGHT_FALLBACK_PREREGISTRATION_20260828.md`. Two things fixed in
advance:

1. **Validity condition** — early stopping must actually fire,
   `best_iter + 500 < 8000`, or the run is uninformative about performance and
   the axis closes as "the cell arm does not stop", not as a verdict.
2. **The LEAFIT failure signature is named as the threat.** LEAFIT lost because
   `corr(base, cell)` **rose**. More trees is a capacity change, not
   convergence-to-the-base, so the prediction is that the correlation holds or
   falls. **If it rises the way LEAFIT's did, the axis closes on that basis even
   if the core delta is positive** — the result would then be noise on a refuted
   mechanism.

Budget 8000 rather than 5000 because `4533 + 500 > 5000`, so 5000 may itself
have been truncating.

Base and control are **reused** from the LEAFIT session — same host, same day,
same commit, identical command except the changed variable. `--cell-leaf-iters`
defaults to 0 and emits nothing, so the code path is byte-equivalent to the
pre-flag trainer. Parity is asserted before scoring: the effective-parameter
diff must be `iterations` only, and `leaf_estimation_iterations` must read 1 on
**both** cell packs so the closed axis cannot leak in. **1 fit.**

## Validity condition — met

**[FACT]** `best_iter = 3813`, shrunk to 3814 trees. `3813 + 500 = 4313 < 8000`,
so early stopping fired and genuinely selected.

**[FACT]** That alone confirms the observation on the current recipe: the
control's `best_iter 2986` was a **truncation**, not a selection. Given the
budget, this arm wants **28% more trees**.

*(score appended when the refit completes)*

## CELLBUD result — FAIL, and the mechanism test passing is the point

**[FACT]** Parity: 253,507 untouched-2024 rows; `row_id` and target elementwise
identical; cell features, categoricals and `fm_success` identical;
`leaf_estimation_iterations` **1 on both**, so the closed axis did not leak in;
the trainer's own lineage records differ in **`iters` only**.
`rms(candidate, control) = 0.002472` against the 0.002 threshold.

```
 seed      base     cellC     cellK      coreC      coreK     delta
    3    863.99    879.68    874.12    888.835    887.779    -1.056
```

**[FACT]** `delta <= 0` -> **FAIL, STOP.** Score search ends per §17E.

**[FACT] The pre-registered diversity mechanism test PASSED — and the candidate
still lost.**

```
                            LEAFIT (failed)        CELLBUD (this run)
corr(base, cell)      0.9647370 -> 0.9681003   0.9647370 -> 0.9617575
corr(base resid, cell resid)  0.9996975 -> 0.9996598 (fell)
cell prediction sd            0.046456  -> 0.047381  (rose)
```

The prediction written before the fit was that a capacity change would hold or
lower the correlation, unlike LEAFIT's convergence-to-the-base. It fell. The
diversity story was right, and the arm got worse regardless.

**[FACT] The mechanism is overfitting to the validation season.** The trainer's
selection-surface score *improves* — `val2023 BSS 589.88 -> 590.70` — while
untouched-2024 cell BSS falls **5.56**. Core resolution falls
`0.00222858 -> 0.00222385`; reliability improves.

**[INFERENCE] The ceiling is load-bearing, which is the opposite of the
intuition that opened the axis.** `--iters 3000` is not a defect the cell arm
suffers; it is implicit regularisation that happens to sit near the right
stopping point for out-of-time generalisation. The 813 extra iterations early
stopping on val2023 endorses do not survive a season boundary.

**[INFERENCE]** Read with **D12** — which closed changing *which metric* picks
the cell checkpoint — the cell arm's checkpoint machinery tracks untouched-season
performance in neither its criterion nor its budget, and the shipped
configuration survives because the ceiling compensates for both.

**[FACT]** Segments (diagnostic only): R −1.62, F +3.20, early +2.42, late
**−5.60**. Cost: **16,584 s (4.6 h)** for the single candidate fit against the
control's 550 s, because the refit runs 1.5 × 3813 = 5,720 trees with no early
stopping.

## A tool defect found and fixed by the parity check

**[FACT]** The gate initially **refused to judge CELLBUD**, reporting an empty
effective-parameter diff. `effective()` was not collecting `iterations`, so an
experiment whose changed variable *is* the budget could never satisfy its own
parity contract. Fixing that then made the **LEAFIT** comparison fail parity,
because its refit length moved too (better leaves → earlier `best_iter` →
shorter refit).

**[INFERENCE]** The rule was wrong in both directions. `iterations`,
`tree_count` and `best_iteration` are **outcomes**, not settings — any change
that moves the stopping point moves them. The parity contract now asserts the
changed variable from each run's own **lineage record** (`resolved_args`), which
holds the flags actually passed. Verified: LEAFIT's two cell runs differ in
`cell_leaf_iters` only; CELLBUD's differ in `iters` only. Both gates then pass
parity and reproduce their verdicts — LEAFIT still **−0.381**.

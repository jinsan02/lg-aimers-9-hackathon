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

**[FACT]** `desktop-5070` is **unreachable** — `ssh: connect to host
100.121.174.83 port 22: Connection timed out`, twice, several minutes apart. The
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

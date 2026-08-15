# P3-C2 handoff — implementation brief for Codex

Written 2026-08-15 by Claude at `128497f`. Everything up to and including the
safety gate is **done and frozen**. What remains is implementation, tests, and
one seed-3 run.

Read first, in order: [AGENTS.md](../AGENTS.md) →
[docs/P3C2_PREREGISTRATION_20260815.md](P3C2_PREREGISTRATION_20260815.md) →
the `P3-C2` and `P3-C` entries at the end of [docs/SETTLED.md](SETTLED.md).

---

## 1. State

| | |
|---|---|
| main | `128497f`, pushed, working tree clean |
| champion | **B1S8, LB 1108.4333490288** — unchanged, not to be touched |
| 5070 | idle: 0 Aimers scheduled tasks, 0 python, GPU ~700 MiB / 1% |
| deployed commit | check `C:\aimers\.deployed_commit` and `.deploy_history`; redeploy before running |
| tests | 16/16 modules pass (`python tests/run_all.py`) |
| GPU spent on P3-C2 so far | **zero** |

Four axes closed earlier today, all recorded: RANK16 FAIL (−219.73), H1ADD
base-only DROP (+0.387), D12 FAIL (−62.96), P3-B DROP (+0.278). **No submission
candidate exists.** Do not submit anything without explicit user approval.

## 2. What P3-C2 is — and the one thing that is easy to get wrong

Change **only** which cells the MultiClass CE weight applies to. Inference
semantics do not change.

```
success set for inference   [9, 10, 11]     UNCHANGED
balanced (weighted) cells   {9, 10}
residual cell               11 → weight exactly 1.0
cell 11 in P(success)       YES, and in the deweight normalisation
```

**Cell 11 (`1xxx`) is not excluded, merged, deleted or reclassified.** It is the
`min_share = 0.005` residual catch-all — 0.161% / 0.069% / 0.193% of success
mass in fit / val / test. P3-C tried to balance against it and its own gate
fired at `w11 = 207.03`. P3-C2 leaves it alone. If you find yourself dropping it
from `fm_success`, you have changed the experiment.

Verified counts (do not re-derive by hand; recompute in code from the training
rows):

| partition | cell 9 | cell 10 | cell 11 |
|---|---:|---:|---:|
| fit ≤2022 | 337,393 | 138,840 | 768 |
| val 2023 | 84,838 | 37,829 | 85 |
| test 2024 | 84,364 | 38,629 | 238 |

Selection-fit weights from the formula: **w9 = 0.70575412, w10 = 1.71504249,
w11 = 1.0**, all failure cells 1.0. Mass preservation is exact (477,001.0 vs
477,001). All eight safety checks passed — re-run them in code anyway and refuse
to train if any fails.

## 3. Where to implement

### 3a. Weights at fit time — `src/train_gbdt2.py`

The cell branch is `if args.failmode_cells:` at **line 527**.

- **Selection fit**: labels are `code`, the fit Pool is built just above
  **line 548** (`clf.fit(tr, eval_set=va)`). Compute the weights from
  `code[~is_val]` — the rows this model actually trains on.
- **Deployment refit**: labels are `rcode` from `fm.build_cells(train_dep, ...)`
  at **line 592**, and the Pool is at **lines 596–597**. Recompute the weights
  independently from `rcode`, because the refit trains on more rows and its
  counts differ. Never reuse the selection weights here.

**Trap — the refit Pool already passes row weights.** Line 597 is
`weight=_refit_weights(args, train_dep)` (defined at line 283). CatBoost
multiplies `class_weights` by per-row `weight`, so the two compose. Decide
explicitly and write down which you use:

- `class_weights=[...]` on the `CatBoostClassifier` (cleanest, 12-vector in
  class order), **or**
- fold the class weight into the row weight vector.

Do not do both. Whichever you pick, assert the resulting effective mass equals
the unweighted success mass to ≤1e-9 and print it.

**Trap — class order.** `class_weights` is positional. Assert
`list(model.classes_) == list(range(12))` and that your weight vector is indexed
the same way. `failmode.success_prob` (`src/failmode.py:278`) sums
`proba[:, sorted(succ)]`, so it assumes column *i* is class *i*. Keep that true.

### 3b. Deweight at inference — `src/fpipe.py`

`fpipe.predict` dispatches at **line 610**:

```python
if pack.get("fm_success"):
    return proba[:, pack["fm_success"]].sum(axis=1)
```

This is the shipped submission path. Add the deweight **before** the sum, gated
on `pack.get("analytic_deweight")`, so a pack without the flag behaves exactly as
today:

```
z_c = q_c / w_c            # elementwise over the 12 columns
p_c = z_c / sum_j(z_j)     # renormalise per row
P(success) = p9 + p10 + p11
```

Cell 11's weight is 1.0 but it is still divided, still renormalised, still
summed.

### 3c. The pack contract

The pack dump is around **line 2433** (`"fm_success": getattr(model, "_fm_success", None)`).
Store these **in the pack dict**, not as CatBoost attributes — custom attributes
do not survive joblib, which cost this project two silent defects already
(`rank_calib`, `fm_multilabel`):

```
class_weights, classes_order, fm_success=[9,10,11],
balanced_cells=[9,10], residual_cells=[11], analytic_deweight=True,
selection_counts, selection_weights, refit_counts, refit_weights
```

Also add the two count/weight vectors to `tools/lineage.py`'s record.

## 4. Tests required before any GPU

Write them in `tests/`, run `python tests/run_all.py` (it fails on zero
collected). The ten from the instruction:

1. all weights 1 → bit-identical to current cell inference
2. per-row class probabilities sum to 1 after deweight
3. all finite, in [0,1]
4. `classes_` order == weight vector order
5. `P(success) == p9 + p10 + p11`
6. cell 11 still in the success sum
7. `w11 == 1.0` exactly
8. save → **new process** → load → predict parity
9. trainer scoring == `fpipe.predict(pack, X)`
10. subset / reversal / half / single-row independence

Plus: `python tools/verify_champion_identical.py` must stay
**14/14 bit-identical, blend 0.5087694207**. If any champion prediction moves,
stop and report — do not run P3-C2.

Two lessons from today's tests, both real:

- A stub that emits a **constant** probability per checkpoint asserts its own
  error — against a balanced target a constant 0.9 scores *worse* than 0.5.
  Vary skill, not level.
- `np.load` on an npz keeps the archive open and locks the file on Windows. Use
  a context manager.

## 5. Running it

Both arms, same session, same host, one fixed base artifact into both cores.
Use a **fresh control** unless host, surface, commit, feature order, training
fingerprint and inference-path parity are *all* proven first —
`tools/training_path_diff.py` (AST, not commit equality) and
`tools/feature_path_parity.py` exist for this, and the latter reports
`INCONCLUSIVE` rather than passing when it cannot drive a function. Today that
inconclusive result is why P3-B fitted 12 models instead of 6.

```bash
python tools/precheck.py --file scripts/<your>.bat     # must exit 0
bash tools/deploy_5070.sh                              # stamps .deploy_history
scp -q scripts/<your>.bat desktop-5070:C:/aimers/scripts/
bash tools/run5070.sh P3C2 'C:\\aimers\\scripts\\<your>.bat' 'C:\\aimers\\out\\p3c2.log'
```

- `run5070.sh` registers under **SYSTEM** so the job survives ssh disconnect.
  Verify `Task To Run:` shows **single** backslashes — `C:aimersscripts...`
  means they were eaten and the task will report `267011`.
- SYSTEM has no user PATH: call `C:\aimers\.conda\python.exe` by absolute path
  inside the batch.
- New tags only. Do not overwrite `P3BCTL_*`, `P3BCAND_*`, `B1J6_*`,
  `B1SMOKE_*`, or anything under `out/scout_recovered/`.
- Never filter a log through `grep` when checking whether a run succeeded — a
  swallowed traceback once made a dead run read as complete.

Expected cost: the cell arm is ~5 min per seed on the 5070, so two arms at seed
3 is roughly 15 minutes including the feature build.

## 6. Gate and report

| condition | outcome |
|---|---|
| artifact or inference parity fails | **INFRA FAIL** |
| untouched fixed-core delta ≤ 0 | **FAIL** |
| 0 < delta < +5 | **HOLD**, no extension |
| delta ≥ +5 and both halves ≥ 0 | extend to n = 6 |

**A FAIL or HOLD at seed 3 closes the whole P3-C / P3-C2 class-weight family.**
Do not then try weighting only cell 9, or only cell 10, or any other variant.

Report: per-partition counts, actual weights and mass-preservation error,
predicted means and BSS before/after deweight, cell-alone BSS, fixed-core BSS
and delta, source 2023 and untouched 2024, halves, R/F, reliability/resolution,
RMS and correlation against the control, per-success-cell calibration and
predicted mass, and artifact hashes/fingerprints/parity.

Before comparing anything, verify — and say so in the report — that both arms
share host, surface, seeds, and that `row_id` and target are identical
**elementwise**, not merely as sets. Every gate this session was checked that
way and one of them (`legacy_failmode_geometry`) was wrong precisely because it
compared by code instead of by name.

## 7. Forbidden

Re-running P3-C; weighting cell 11; any weight sweep; changing the inference
success set or the taxonomy; legacy labels; combining with multilabel, D12 or
the common budget; changing the 0.45/0.55 blend; turning the deweight on or off
after seeing a result; adding a clip, cap, smoothing, power or temperature;
**submitting without explicit user approval**.

Record results in `HANDOFF.md`, `EXPERIMENT.md` and `docs/SETTLED.md`
(**append-only**), then commit and push.

# P3-C2 — structured-success-only balanced cell CE

Pre-registered 2026-08-15 at `87f1cae`, **before any implementation or run**.
A separate experiment from P3-C, not an amendment to it: P3-C stays HOLD and its
`docs/SETTLED.md` entry is untouched (append-only).

## What changed from P3-C, and why it is not a post-hoc rescue

P3-C balanced all three success cells and its own safety gate fired at
`w11 = 207.03`. The taxonomy audit explained why: cell 11 `1xxx` is the
`min_share = 0.005` **residual catch-all**, not a failure mode — 768 of 477,001
success rows in the fit partition, **0.161%**.

P3-C2 therefore narrows **which cells the class weight is applied to**, and
changes nothing about inference:

| | |
|---|---|
| inference success set | **`[9, 10, 11]` — unchanged** |
| balanced cells | `{9, 10}` |
| residual cell | `11`, training weight **exactly 1** |
| cell 11 in `P(success)` | **yes**, and in the deweight normalisation |

Cell 11 is not excluded, merged, deleted or reclassified. It keeps its class,
its probability and its place in the success sum; it simply is not a target for
mass equalisation, because equalising against a 0.161% overflow bin is what the
P3-C gate refused. This is a change of **weighting scope**, not of taxonomy or
of the success set.

## P1 — taxonomy audit (verified 2026-08-15, current code, judging surface)

12 classes `0..11`:
`['0000','0001','0010','0011','0100','0101','0110','0111','0xxx','1000','1010','1xxx']`
Success `[9,10,11]`, matching `cat_B1S_cell_s3.pkl`'s `fm_success` exactly.
`failmode.MIN_SHARE = 0.005`; `failmode.py` is **not** in
`submissions/b1s8_20260813.zip` (22 members, train-only as required).
Labels are rebuilt per partition with `fit_mask` — partition-safe.

| partition | n | n_success | cell 9 | cell 10 | cell 11 | classes |
|---|---:|---:|---:|---:|---:|---|
| fit ≤2022 | 901,200 | 477,001 | 337,393 (70.732%) | 138,840 (29.107%) | 768 (**0.161%**) | 12/12 |
| val 2023 | 245,525 | 122,752 | 84,838 (69.113%) | 37,829 (30.817%) | 85 (**0.069%**) | 12/12 |
| test 2024 | 253,507 | 123,231 | 84,364 (68.460%) | 38,629 (31.347%) | 238 (**0.193%**) | 12/12 |

No class is missing in any partition. Nothing here changed the taxonomy or the
success definition.

## P2 — the formula

On the rows the model is training on, with `S* = {9, 10}`:

```
N* = n9 + n10
w9  = N* / (2 * n9)
w10 = N* / (2 * n10)
w11 = 1
every failure cell = 1
```

Computed from each partition's own training rows, never hard-coded, and never
from validation or test counts or targets. On the selection fit partition this
evaluates to **w9 = 0.70575412, w10 = 1.71504249, w11 = 1.0**, and the
deployment refit recomputes it on its own rows. Both weight vectors and both
count vectors are recorded in the artifact lineage.

Mass preservation, verified exactly: `w9*n9 + w10*n10 + 1*n11 = 477,001.0`
against `n9+n10+n11 = 477,001`, **error 0.000e+00**.

## P3 — safety gate (all PASS, 2026-08-15)

| check | result |
|---|---|
| n9 > 0 and n10 > 0 | PASS |
| max(weight) ≤ 5 | PASS (1.715042) |
| min(weight) ≥ 0.2 | PASS (0.705754) |
| w11 == 1 | PASS |
| all failure weights == 1 | PASS |
| mass error ≤ 1e-9 | PASS (0.000e+00) |
| success set == [9,10,11] | PASS |
| weight vector length == 12 | PASS |

Forbidden: clip, cap, smoothing, power, temperature, any multiplier, merging or
deleting class 11, changing any failure weight, and revising the formula after
seeing a result.

## P4 — analytic deweight

```
z_c = q_c / w_c
p_c = z_c / sum_j(z_j)
P(success) = p9 + p10 + p11
```

Cell 11 has weight 1 but is still divided, still normalised, and still summed.

Stored **explicitly in the pack dict**, never on the CatBoost object:
`class_weights`, `classes_order`, `fm_success=[9,10,11]`,
`balanced_cells=[9,10]`, `residual_cells=[11]`, `analytic_deweight=True`, and
the selection and refit counts and weight vectors.

## P5 — tests required before any GPU

The ten listed in the instruction, plus: **all 14 B1S8 members must stay
bit-identical** through the current `fpipe`
(`tools/verify_champion_identical.py`). If any champion prediction moves,
P3-C2 does not run.

## P6/P7 — seed-3 gate

Fresh control in the same session unless host, surface, commit, feature order,
training fingerprint **and** inference-path parity are all proven before any
number is read. One fixed base artifact feeds both cores.

| condition | outcome |
|---|---|
| artifact or inference parity fails | **INFRA FAIL** |
| untouched fixed-core delta ≤ 0 | **FAIL** |
| 0 < delta < +5 | **HOLD**, no extension |
| delta ≥ +5 and both halves ≥ 0 | extend to n = 6 |

**A FAIL or HOLD at seed 3 closes the entire P3-C / P3-C2 class-weight family.**
No weighting only cell 9 or only cell 10, no other variant.

## P8 — adoption

n = 6 paired, same host/surface/fingerprint, mean ≥ +3, t ≥ 2.4, 95% CI
reported, seed ensemble positive, halves non-worse, no large R/F collapse.
Even on PASS: report the submission-form refit, ZIP structure,
subset-independence and expected LB range, and **request approval before
submitting**.

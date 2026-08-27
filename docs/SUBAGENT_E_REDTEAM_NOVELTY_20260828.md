# Agent E — red-team novelty audit

**Verdict: NO_WORTHWHILE_HOLE that is independent of Agent D.** The one hole the
matrix exposes is the same object D found, so under §7 it collapses rather than
becoming a fifth candidate.

## What was actually scanned

Not a brainstorm. The five-column matrix was populated from registries rather
than from imagination, because SETTLED now carries 183+ FLAG lines and the
failure mode of this exercise is proposing a renamed closed axis.

| column | how it was enumerated | result |
|---|---|---|
| INFORMATION SOURCE | official train columns, Trackman, `asof_*`, derived families in `src/features.py` | every family SHIPPED or CLOSED; the six-axis triage and both 2026-08-28 add-ons closed the last unexplored ones |
| REPRESENTATION | `fpipe` STEP_ORDER stages and their fitted artifacts | all SHIPPED; `id_cohort` CLOSED at −2.780 on 2026-08-16 |
| TRAINING OBJECTIVE | all 37 CatBoost effective parameters x three registries | **16 never-decided; exactly one is also a base/cell asymmetry** |
| OUTPUT GEOMETRY | taxonomy operations | reweight / reparametrise / project / delete all CLOSED — FMCOARSE (−18.6) closed the family |
| DEPLOYMENT CORRECTION | shipped `script.py` | SHIFT/SLOPE, middle offsets, PB offset, blend weight — all CLOSED, every refit lost to the legacy constant |

## Candidates generated, scored, and why they died

**1. Logit-space pooling instead of probability-space pooling.** Same weight,
different operator, so arguably not the closed blend-weight axis.
**DUPLICATE** — `docs/SETTLED.md` FLAG `logit-blend`, CLOSED, centred **+0.021**:
"VB2_base 8 seeds and ZD5 6 seeds at w=.55: probability mean 961.628 vs logit
mean 961.649. The prediction range is too narrow for the operator choice to
matter." Rejected before spending a measurement.
Score: novelty 2, legal 3, gain 0, transfer 1, duplicate **−3**, cost 0.

**2. The fifteen other never-decided CatBoost defaults** (`score_function
Cosine`, `nan_mode Min`, `rsm 1`, `feature_border_type GreedyLogSum`,
`counter_calc_method SkipTest`, `model_shrink_*`, …). All are identical on both
arms, so touching any of them is choosing a global hyperparameter value with no
principle behind the choice — **§11's banned CatBoost grid**, one flag at a time.
Rejected on the rule, not on a number.
Score: novelty 2, legal 3, gain 1, transfer 0, duplicate −1, cost −1.

**3. `leaf_estimation_iterations` base 10 / cell 1.** The only cell in the whole
matrix that is simultaneously never-decided and an unintended asymmetry, which
is what makes the experiment an *alignment* rather than a *value choice* — and
that is precisely what keeps it outside the banned grid.
Score: novelty 3, legal 3, gain 1 (argued, not measured), transfer 2,
duplicate 0, cost −1.

**This is Agent D's candidate.** Two independent routes — D's constructor
autopsy and E's registry matrix — reached the same object from different
directions. Under §7 that is a duplicate collapse, and it is also the strongest
corroboration available that the hole is real: the matrix says there is exactly
one, and D independently says the same one.

## What E did not do

No new feature was proposed. No Trackman summary, TE key, shrinkage k, residual
Ridge, calibration, PB window, NN or feature interaction was put forward; every
one of those is on the §2 closed list and would have been a renamed duplicate.
The honest output of a red-team pass over a search space this thoroughly
explored is usually a null, and inventing a fifth candidate to fill the slot
would be exactly the failure §11's last line names.

## One observation handed to synthesis, not a candidate

The 2026-08-16 finding about `tools/precheck.py` argument dispatch is still
open: any first argument that is not exactly `--file` falls through to the flag
checker, which prints plausible `[OK]`/`[WARN]` lines and **exits 0 without
opening the file**. That is a guard capable of reporting success on a script it
never read, and it gates every GPU launch — including whatever this synthesis
licenses.

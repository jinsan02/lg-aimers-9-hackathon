# GSK2 — GENERAL_SKILL_ADD cell-only transfer gate

Pre-registered 2026-08-16 after RMSE2 closed and before fitting or reading any
GSK2 result. This is the third and final item in the user-approved 1 -> 2 -> 3
queue.

## Question and novelty

The earlier `GENERAL_SKILL_ADD` experiment changed both base and cell on the
submission-selection surface. Its fresh six-seed split was base -0.077 and cell
+2.568 (t=2.18, 6/6 positive), while the combined core was only +1.104 and was
parked. Selecting the cell family after seeing that split is not licensed by the
old experiment. GSK2 is a new, explicit cell-only hypothesis and tests whether
the season-level general-skill representation transfers to an untouched year.

Only the cell member receives `--feat-skill`; `--feat-skill-pc` remains. This
must add exactly `skill_hat` and `skill_hat_vs_std` to the 121-feature control,
for 123 ordered features. No feature subset, coefficient, skill axis, blend
weight, stopping budget or calibration search is allowed.

## Arms

One 5070 session, judging surface fit<=2022 / val2023 / untouched2024, seeds
`3,4,5,6,8,13`:

- fresh control `GSK2CTL_cell`: current corrected 12-cell champion recipe;
- candidate `GSK2CAND_cell`: identical plus `--feat-skill`;
- fixed base in both cores: the just-completed `RMSE2CTL_base`, same host,
  surface, seeds and current Logloss recipe;
- fixed core: `0.45*base + 0.55*cell`.

Cell flags are depth 5, lr .01, l2 10, border 254, 3000 iterations, ES 500,
refit multiplier 1.5, P1, `drop-f-pre 2022`, corrected 12-cell taxonomy,
`fm-modes middle,ball,reverse`, `fm-min-share .005`. All post-processing is
absent from both arms; the paired question is the core model probability only.

The fixed base is allowed because it cancels elementwise in every paired delta.
It still must match the new cell arms on host, surface, seed, row_id, target and
fit-row hash. Any mismatch invalidates the comparison.

## Integrity gate

- all three families contain the same six seeds on `DESKTOP-053T952` and
  `val2023->test2024`;
- row_id and target arrays match elementwise for every seed;
- fit-row hashes match; control and base have the champion 121-feature hash;
- candidate has exactly the control's ordered features plus
  `skill_hat,skill_hat_vs_std`, with no removal or reorder of existing columns;
- every saved candidate and control pack loads in a new process, reproduces its
  stored full prediction, and has reversal/half/single-row drift zero;
- `fm_success == [9,10,11]`, classes are `0..11`, and no multilabel route;
- source and deployment skill statistics are trained only from their allowed
  training partitions; test-frame membership cannot change a prediction.

Failure is INFRA FAIL; no performance verdict is drawn.

## Gate

Primary statistic is the six paired fixed-core untouched-2024 delta.

- KEEP only if mean >= +3, paired t >= 2.4, six-seed ensemble > 0, both row
  halves >= 0 and R/F both >= 0;
- DROP if the Student-t 95% upper bound is below +3;
- otherwise HOLD, with no seed extension or variant.

Report per-seed cell/core deltas, mean/SE/t/Student-t CI, median/sign count,
ensemble, source/target, halves, R/F, reliability/resolution, RMS/correlation,
best iterations, fingerprints, exact feature delta and artifact parity.

No submission without explicit user approval.

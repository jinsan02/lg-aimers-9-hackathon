# GSK3 — independent historical transfer confirmation

Pre-registered after GSK2 returned HOLD and before fitting or reading any GSK3
result. This is a confirmation of the exact GSK2 feature change, not a new
feature or segment correction.

## Why this run

GSK2 added exactly `skill_hat, skill_hat_vs_std` to the corrected 12-cell arm.
On fit<=2022 / val2023 / untouched2024 its fixed-core mean was +3.141 (t=4.168)
and ensemble +3.108, but F was -3.126 and source-season deltas were mostly
negative. It therefore stopped at HOLD. The remaining question is whether the
positive untouched-season transfer repeats at a second honest boundary.

## Frozen arms

One `DESKTOP-053T952` session, fit<=2021 / val2022 / untouched2023, seeds
`3,4,5,6,8,13`, `--drop-f-pre 2021`, `--max-train-season 2023`:

- `GSK3CTL_base`: fresh depth-8 Logloss base;
- `GSK3CTL_cell`: fresh corrected 12-cell control, 121 features;
- `GSK3CAND_cell`: identical cell plus `--feat-skill`, adding exactly
  `skill_hat, skill_hat_vs_std` for 123 features;
- fixed core is `0.45*base + 0.55*cell`.

All other flags, seeds, model parameters and the blend are the GSK2 contract.
No coefficient, F-only routing, segment correction, skill-axis choice, feature
subset, stopping budget or post-processing may change.

## Integrity and decision

Require same host/surface/seed, elementwise row_id and target equality, one
fit-row hash across all 18 members, exact 121->123 ordered feature delta,
new-process replay maxdiff 0, and reversal/half/single-row drift 0.

The untouched-2023 fixed-core gate is deliberately identical to GSK2:

- PASS only if mean >= +3, paired t >= 2.4, ensemble >0, both halves >=0,
  and R/F both >=0;
- DROP if Student-t 95% upper bound < +3;
- otherwise HOLD.

Only PASS licenses training a 2025-deployment GSK cell family and packaging an
unsubmitted candidate. DROP/HOLD ends the overnight model work. No variant or
seed extension is allowed, and nothing is submitted without the user.


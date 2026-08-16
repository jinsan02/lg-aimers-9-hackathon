# Champion recipe — GSKDEP + E-LB1 (2026-08-16)

## Official result

- LB: **1111.3713632162** (`schedule`, 31 seconds)
- artefact: `submissions/gskdep_0816.zip`
- SHA256: `87617a131498b1121c100668b965b57443abbfdbe8bef5c39b28976d4f29e70d`
- previous champion E-LB1: 1110.9806302398
- official gain: **+0.3907329764**
- gain over B1S8: +2.9380141874
- gain over v11 PB: +9.5692960097

The package passed the fresh-process server smoke, completed 245,789 synthetic
rows in 30 seconds, and had exactly zero drift in the strong half/player/
scattered/single-row subset-independence audit. The official server runtime was
31 seconds.

## Exact model recipe

The champion is not a new standalone model. It is the B1S8 ensemble with the
cell arm replaced by the GSK skill-augmented cell family:

1. **Base arm (weight 0.45):** eight B1S binary CatBoost packs, seeds
   `3,4,5,6,8,13,42,7`, 121 features.
2. **Cell arm (weight 0.55):** six `GSKDEP_cell` corrected 12-cell CatBoost
   packs, seeds `3,4,5,6,8,13`, 123 features. Relative to B1S cell, the only
   added inputs are `skill_hat` and `skill_hat_vs_std`.
3. **Cell probability:** sum classes `[9,10,11]`; the train-only failure-mode
   recovery code is not shipped.
4. **Blend calibration:** convert the weighted probability to logit, multiply
   by `SLOPE=1.0416`, invert the logit, then subtract `SHIFT=0.0052`.
5. **Frozen legal lookups:** add the existing recent-middle adjustment and
   exact pitcher-batter adjustment, both fitted only from official training
   data and applied per evaluation row without consulting other test rows.
6. **E-LB1 final correction:** add `+0.002515795361`, then clip to `[0,1]`.

The effective global additive part is not simplified in code: the legacy
`-0.0052` is applied before the two frozen lookups, and the confirmed E-LB1
`+0.002515795361` is applied at the final output. Preserve this order.

## Training and lineage

- host: `DESKTOP-053T952` (5070 Ti, evaluation-server-matched environment)
- cell fit rows: 1,221,585, seasons 2019--2023
- validation: 2024, 253,507 rows
- cell fit hash: `d69792676c50e1cf`
- cell feature hash: `a109f03b48b72da7`
- all six cell packs have 123 ordered features and matching strong lineage
- deployment val2024 cell BSS by seed:
  `927.11/915.54/914.89/922.51/921.13/921.14`
- paired against the matching B1S cell seeds, all six were positive:
  `+7.07/+5.31/+5.97/+11.24/+4.08/+1.86`, mean about `+5.92`

## Why it was promoted

The exact 121-to-123 cell change improved two independent untouched-season
tests before deployment:

| Test | Fixed-core result | Gate |
|---|---:|---:|
| GSK2, untouched 2024 | mean +3.141, t=4.168, ensemble +3.108 | HOLD only because F=-3.126 |
| GSK3, untouched 2023 | mean +19.595, t=5.330, ensemble +19.517 | KEEP |

The official gain confirms that the signal is real, but also confirms the
pre-submission transfer warning. Only about 13% of GSK2's +3.108 offline
ensemble improvement appeared on the 2025 leaderboard (`+0.391`). GSK3 was
heavily F-driven while GSK2's F segment was negative, so the feature transfers
weakly rather than at its historical magnitude.

## Current interpretation and next boundary

- The new champion is valid and should replace E-LB1 as the rollback baseline.
- Do not claim the GSK feature is worth +3 to +20 on 2025; the measured value is
  +0.391 on the frozen champion recipe.
- Do not tune GSK strength, league-specific coefficients, or feature weights
  from this single LB delta. That would fit the public/private answer key and
  conflict with the project's no-LB-constant rule.
- E-LB1's global shift remains closed at its exact one-dimensional optimum.
- E-LB2 (`w_cell=.45/.65`) remains separately pre-registered, but its baseline
  packages were built from E-LB1 rather than this GSK champion. Claude must
  decide whether those probes are still scientifically useful before submission.

## Reproduction sources

- renderer: `tools/prepare_gsk_package.py`
- generated inference script: `out/script_blend_gsk.py` (ignored artefact)
- template: `src/script_blend_b1s.py`
- deployment contract: `docs/GSK_DEPLOYMENT_CONTRACT_20260816.md`
- transfer preregistration: `docs/GSK3_TRANSFER_PREREGISTRATION_20260816.md`
- experiment summary: `EXPERIMENT.md`


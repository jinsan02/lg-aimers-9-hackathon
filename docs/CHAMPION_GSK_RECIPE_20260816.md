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

### Corrections from the 2026-08-16 zip audit (the ZIP is the truth)

Verified member-by-member against `submissions/gskdep_0816.zip`; everything above
matched except these five, which are corrected here:

1. **Two clips, not one.** `script.py:101` is `np.clip(preds - SHIFT, 0, 1)` — the
   legacy shift is followed by a clip to `[0,1]` *before* the two frozen lookups,
   and `script.py:110` clips again after them. Both are part of the recipe.
2. **SHIFT is stored positive.** `script.py:37` holds `SHIFT = 0.0052` and line 101
   applies `preds - SHIFT`. Writing the constant as `-0.0052` describes the effect,
   not the literal.
3. **Only the cell packs carry a lineage record.** All six `GSKDEP_cell` packs have
   schema-1 lineage (host, fit rows, seasons, hashes). The eight base packs carry
   **none** — as they did not in B1S8 either. Their provenance is established more
   strongly instead: all eight `.pkl`, `model/matchup_constants_2024.npz`,
   `features.py`, `season_std.py`, `skill.py` and `target_enc.py` are
   **byte-identical (sha256) to `submissions/b1s8_20260813.zip`**, so the base arm
   is literally the same artifact and its training set is necessarily identical.
4. **`fpipe.py` differs from the B1S8 shipped copy** (explicit `predict` dispatch
   plus `deweight_multiclass`). The base packs are unchanged but the code that
   scores them is not, which is why the whole package — not just the cell arm —
   was re-audited.
5. The "245,789 rows in 30 seconds" figure is the 5070-class host. On the laptop
   the honest extrapolation from a measured 60,000-row run (10.1 s including all
   14 model loads) is **≈41 s**. Both are ~15x inside the 600 s limit.

**Packaging note, not an action item.** `requirements.txt` pins only
`catboost==1.2.10` while `script.py:27` imports `joblib` (and the modules use
numpy/pandas). That is a real gap in principle, but the official 31-second run
already proves the evaluation image supplies them. Do not repackage the champion
to "fix" this.

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

The official gain confirms that the signal is real.

⚠ **The "only 13% of +3.108 reproduced / transfer-attenuated" reading is
retracted** (Claude review, 2026-08-16). It compared the LB against a
**judging-surface** number that was never a prediction for 2025. The
surface-matched estimate was already on record — `GENERAL_SKILL_ADD`
(`docs/SETTLED.md`) gives fresh-vs-fresh core **+1.104, SE 1.086** — and an
independent recomputation from the stored val2024 arrays, swapping only the cell
arm and applying the shipped calibration, gives **956.136 → 957.478 = +1.342**.

The sampling SE of the LB delta itself is **±1.06** (calibrated
`rms(p_new − p_old) = 0.001323` over 245,789 rows). So the observed **+0.391**
is **0.9 SE** below the surface-matched **+1.342** and **0.66 SE** below the
recorded **+1.104**. There is no attenuation to explain: the submission-surface
debiased delta predicted the leaderboard inside one standard error.

The operative consequence is a limit on measurement, not on the feature: **one
submission is one draw with SE ≈ 1, so the leaderboard cannot resolve anything
below roughly +2 to +3 for changes of this magnitude.** The F-segment caveat
(GSK3 F-driven, GSK2 F = −3.126) remains recorded as a segment risk, but it is
not what produced the small official gain.

## Current interpretation and next boundary

- The new champion is valid and should replace E-LB1 as the rollback baseline.
- Do not claim the GSK feature is worth +3 to +20 on 2025; the measured value is
  +0.391 on the frozen champion recipe.
- Do not tune GSK strength, league-specific coefficients, or feature weights
  from this single LB delta. That would fit the public/private answer key and
  conflict with the project's no-LB-constant rule.
- E-LB1's global shift remains closed at its exact one-dimensional optimum.
- E-LB2 (`w_cell=.45/.65`) is **CANCELLED** (Claude, 2026-08-16). The offline
  weight curve on val2024 with the GSK cell peaks at `w* ≈ 0.575–0.60` and is
  worth **+0.07** over the shipped 0.55; the whole .45–.65 span covers 0.78
  points. Against an LB delta SE of ±1.06 the probe's signal-to-noise is 0.066,
  so two submissions would fit a quadratic to noise. `_W_CELL` stays 0.55 and no
  coefficient is transplanted from those packages.

## Reproduction sources

- renderer: `tools/prepare_gsk_package.py`
- generated inference script: `out/script_blend_gsk.py` (ignored artefact)
- template: `src/script_blend_b1s.py`
- deployment contract: `docs/GSK_DEPLOYMENT_CONTRACT_20260816.md`
- transfer preregistration: `docs/GSK3_TRANSFER_PREREGISTRATION_20260816.md`
- experiment summary: `EXPERIMENT.md`


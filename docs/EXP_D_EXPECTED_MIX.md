# EXP-D — Trackman expected pitch mix

## Existing-implementation search

- `src/tm_context.py` already builds linkage-free league context aggregates on
  the same keys (`MIN_N = 200`, coarser-key fallback).
- `src/build_tm_pitchmix.py` (E114) already builds P(group | pitcher, count)
  from Trackman with per-season expanding accumulation.
- The champion already carries **twelve** pitch-mix columns, not three: raw
  `asof_pitcher_{fastball,breaking,offspeed}_rate`, their `_shr` shrunk forms,
  the season-standardised `std_*` forms and their `_delta`. Removing the delta
  family costs −4.99 (`local-targeted-pruning`).
- SETTLED neighbours: `--tm-feats` CLOSED at −6.04 (t −2.14),
  `predicted-pitch-probability-features` CLOSED.

So the only genuinely new element is the context ratio multiplier
`P(g | context) / P(g)`, and the correlation gate has to run against twelve
existing columns rather than three.

## Hypothesis

Multiplying a pitcher's own pitch-group rates by a situational league ratio
adds information the twelve existing pitch-mix columns do not carry.

## Baseline

`B1S` (submission shape, `--p1`, six seeds), 121 features. Candidate is the
same run with three columns added: 124 features.

## Change

`src/build_expected_mix.py` → `data/processed/expected_mix.csv`, merged via a
new `--extra-feats`. Season S uses Trackman seasons **< S** only; 2019 has no
prior Trackman season so its ratio is exactly 1.0.

### One bug worth recording

The first build produced a feature *identical* to the personal rate —
correlation 1.000000, sd of the difference 0.000000. Trackman spells hands
`"Right"`/`"Left"` and `train.csv` codes them `2`/`1`, so every context lookup
missed and the ratio came back 1.0 everywhere. Nothing would have failed: the
run would have trained, scored, and reported a clean null result for a feature
that was never computed.

Both the builder and `--extra-feats` now refuse on a high miss rate (1% and 5%
respectively). After the fix the ratio moves the feature properly —
correlation with the own-rate 0.71 / 0.76 / 0.79 across the three groups.

## Results — 6 paired seeds, val2024

| arm | Δ | SE | t | 95% upper | per seed |
|---|---:|---:|---:|---:|---|
| base | +1.34 | 1.33 | +1.01 | +4.75 | 1.47, 6.51, 0.18, 2.81, −3.32, 0.40 |
| cell | +1.58 | 0.97 | +1.63 | +4.09 | 1.28, 1.74, 5.94, 1.54, −1.08, 0.07 |
| **core** | **+1.64** | 0.87 | **+1.87** | **+3.88** | 1.53, 4.09, 2.99, 2.43, −2.09, 0.87 |

| Metric | Baseline (121) | Experiment (124) | Delta |
|---|---:|---:|---:|
| BSS core, 6-seed mean | — | — | +1.64 |
| BSS core, ensembled | 929.49 | 931.17 | +1.68 |
| best_iter, seed 3 base | 1139 | 1206 | +67 |
| runtime | — | ~51 min, 6 seeds, both arms | — |

## Prediction correlation

Core predictions correlate **0.998999** with the 121-feature baseline; centred
rms 0.00207, so `Dmax = 400320 · rms² = 1.7`. As a *blend member* that would be
worthless. This is a feature addition rather than a member, so the number is
context and not the verdict — but it says the twelve existing pitch-mix columns
already carry nearly all of it.

## Verdict

**PARK.** Five of six seeds positive and the ensemble gains +1.68, but Δ = +1.64
with t = 1.87 clears neither adoption bar (Δ ≥ +3, t ≥ 2.4), and the 95% upper
bound of +3.88 sits just above the rejection line — precisely the 0…+3
borderline the audit says to park rather than rescue with more seeds.

## Reasons

1. Below both adoption thresholds on the surface that matters.
2. Not rejectable either. This is a borderline, and the rule for borderlines is
   to park.
3. The correlation gate explains the size: 0.999 against the baseline. The
   context ratio is mostly re-expressing what the model already has.

## EXP-E — expected pitch physics

**Not run.** The plan gates E on D being KEEP. D is PARK, so E does not start.
`src/build_tm_command.py`, `src/build_mechanics_history.py` and
`src/build_tm_consistency.py` already read `rel_speed`, `spin_rate`,
`extension` and per-pitch release spread, and should be read first if E is ever
opened.

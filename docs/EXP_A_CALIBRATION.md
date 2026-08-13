# EXP-A — legacy calibration re-derivation

## Existing-implementation search

- Same experiment: **no**. `SLOPE = 1.0416` / `SHIFT = 0.0052` are inline
  constants in `src/script_blend_v11.py`; nothing has re-derived them.
- Similar: `tools/analyze_v12_failure.py` studied the recent-middle lookup, not
  the slope/shift pair.
- SETTLED: not closed. `CHAMPION_v11_RECIPE.md` part 6 already lists both as
  weakly evidenced.

## Hypothesis

The two constants can be re-derived from OOF and improved on.

## Baseline

`B1S` core (`.45/.55`, six seeds, submission shape) on val2024, and `B1J6` core
on the judging surface where a *later* season exists to transfer to.
Tool: `tools/calib_refit.py`.

## Change

Calibration only. No retraining, no recent-middle change, no PB change.

## Protocol correction

The plan asks for `a, b` minimising OOF Brier, reported as OOF Brier before and
after. That fits and scores on the same rows and cannot lose. Every fit below
is therefore reported twice: in-sample, and frozen onto a season the fit never
saw. Only the second decides.

## Results

### The form does not transfer — B1J6, fit on val2023, applied to unseen 2024

| | source (2023) | frozen onto 2024 | Δ vs raw |
|---|---:|---:|---:|
| raw | 616.80 | 894.93 | — |
| legacy `1.0416 / 0.0052` | 584.81 | 893.70 | **−1.22** |
| refit `a=0.9532 b=+0.0149` | 624.32 | 875.00 | **−19.93** |
| refit legacy form `a=0.9534 shift=−0.00373` | 624.36 | 874.92 | −20.00 |
| oracle fitted on 2024 itself | — | 898.65 | +3.73 |

**A freshly fitted pair transfers 19 points worse than the constants it was
meant to replace.** And the reason is not noise: the source-optimal slope is
`0.9532` — below 1, flatten the predictions — while the target-optimal slope is
`1.0481` — above 1, sharpen them. The correction the data asks for **changes
direction between seasons**.

The whole axis is also small. Even the oracle, fitted directly on the answer,
is worth +3.73.

### On the submission surface the legacy constants are already near-optimal

`B1S` core, val2024, 253,507 rows:

| | BSS |
|---|---:|
| raw | 929.49 |
| legacy `1.0416 / 0.0052` | **956.14** |
| refit in-sample `a=1.0645 b=−0.0315` | 959.51 |

Legacy is worth **+26.65** here and sits 3.37 below an in-sample optimum. It was
−1.22 on the judging surface, which is not a contradiction: different training
set, different prediction distribution, so the surfaces are not comparable
(SETTLED `drop-f-pre-omitted`).

### What the legacy constants actually do

| | BSS | pred mean |
|---|---:|---:|
| raw | 929.49 | 0.494266 |
| pure centering (subtract the bias) | 956.15 | 0.486105 |
| legacy slope + shift | 956.14 | 0.488829 |

**The legacy calibration is worth exactly what centering is worth, to 0.01
BSS.** All +26.65 of it is bias removal; the slope contributes nothing
measurable on top. That is consistent with the transfer result — a bias
correction in the direction of a monotonically falling base rate is the part
that has any reason to survive a season, and the shape is the part that does
not.

## Prediction correlation

Not applicable: no new feature, and calibration is a monotone map, so ordering
and therefore resolution are unchanged by construction.

## Verdict

**DROP the refit. Keep the legacy constants.**

Not because they are well-founded — they still are not — but because the one
test that can distinguish them says a refit is worse. Refitting on 2024 and
shipping to 2025 is the same operation that lost 19.93 going 2023 → 2024.

## Reasons

1. Fit-and-freeze is the only honest protocol here, and it rejects the refit.
2. The optimal slope flips from 0.9532 to 1.0481 across one season boundary.
3. The in-sample gain over legacy is +3.37 on B1S — below any adoption bar
   before considering that it is in-sample.
4. What the constants buy is centering, and centering is the piece most likely
   to transfer.

## Follow-up this suggests

The slope is doing nothing (−0.01 against pure centering). A candidate worth
one run: **replace slope+shift with an explicit bias correction** — simpler,
one parameter instead of two, and the parameter is the one with a mechanism.
That is a change to the submission script, so it goes through the same
fit-on-source / freeze / apply gate before it can ship, and it is a separate
experiment from this one.

## Next

EXP-C, recent-middle re-verification. Note it is the same class of object: a
lookup fitted on one season's residuals and frozen onto the next. The result
above is the prior to carry into it, and v12 already lost LB −18.271 changing
exactly that lookup.

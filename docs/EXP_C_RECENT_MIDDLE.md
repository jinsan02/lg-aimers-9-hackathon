# EXP-C — recent-middle correction re-verification

## Existing-implementation search

- Same experiment: **no**. The 8 offsets are inline constants in
  `src/script_blend_v11.py`; nothing has recomputed them from OOF.
- Similar: `tools/make_matchup_constants.py::middle_adjustment` generates them
  (same 8 quantile bins, `MID_K = 500`), so the plan's C-2 shrinkage already
  exists in code and only needed refitting. `tools/analyze_v12_failure.py`
  studied this axis when v12 replaced it with career-middle and lost LB −18.271.
- SETTLED: `frozen-career-middle-q8-k200` is closed; the shipped recent-middle
  lookup itself is not.

## Hypothesis

The shipped lookup does not reproduce on new OOF, and a refit improves it.

## Baseline

Both surfaces, `.45/.55` core, six seeds, **after** slope/shift — the lookup
sits downstream of them. Tool: `tools/exp_c_middle.py`.

## C-1 — the shipped lookup reproduces, and on the submission surface it is near-exact

`B1S` (submission shape, val2024, the surface whose training set matches the
champion's):

| bin | n | OOF offset | shipped | diff |
|---|---:|---:|---:|---:|
| NaN | 3,638 | −0.00794834 | −0.00810269 | +0.00015 |
| 1 | 31,217 | +0.00932022 | +0.00912860 | +0.00019 |
| 2 | 31,244 | +0.00152163 | +0.00145823 | +0.00006 |
| 3 | 31,068 | +0.00158138 | +0.00166458 | −0.00008 |
| 4 | 31,349 | +0.00310681 | +0.00313052 | −0.00002 |
| 5 | 31,103 | +0.00027077 | +0.00028883 | −0.00002 |
| 6 | 31,419 | −0.00548524 | −0.00547501 | −0.00001 |
| 7 | 31,119 | −0.00423198 | −0.00413855 | −0.00009 |
| 8 | 31,350 | −0.00511920 | −0.00507436 | −0.00004 |

**Sign agreement 8/8, and every offset matches to about 1e−4.** In-sample
refitting is worth +0.01 BSS over the shipped numbers (965.55 vs 965.54).

This is the strongest reproduction result so far and it says two things. The
lookup is genuinely re-derivable — it was not a tuned constant. And our B1-S
rebuild produces residuals nearly identical to the champion's on this axis,
independent confirmation that the rebuild is the same object.

On the judging surface (`B1J6`, val2023) sign agreement is 7/8; bin 4 flips.
That surface has a different training set, so a bin disagreeing there is not a
contradiction.

**The lookup is non-monotone in the new OOF too.** Bin 1 +0.0114 → bin 2
+0.0077 → bin 3 +0.0033 → bin 4 −0.0027 → bin 5 +0.0005. So the shape is a
property of the data, not an artifact of how the shipped constants were fitted.

## C-2 — refitting transfers worse than keeping it

`B1J6`, offsets and bin edges fitted on val2023, frozen onto unseen 2024:

| | BSS | Δ |
|---|---:|---:|
| slope/shift only | 893.70 | — |
| **+ shipped lookup** | **901.73** | **+8.03** |
| + OOF lookup refitted on 2023, frozen | 899.42 | +5.72 |
| + oracle fitted on 2024 itself | 911.10 | +17.40 |

Shrinkage sweep on the frozen refit: k=100 +5.58, k=500 +5.72, k=2000 +6.17,
k=5000 +6.82. More shrinkage transfers better, monotonically, and still never
reaches the shipped lookup's +8.03.

Unlike the calibration slope in EXP-A, **the shape is stable across the season
boundary** — all 8 bins plus NaN keep their sign from 2023 to 2024. The axis is
real; the refit is just noisier than the constant it would replace.

## C-3 — tiny residual model

Not run. C-2's frozen refit already underperforms the constant it replaces on a
lookup with 30k rows per bin. A model with seven inputs on the same residual
has more freedom to fit the source season, and the ranking above is ordered by
exactly that.

## Results summary

| Metric | Baseline (shipped) | Experiment (OOF refit) | Delta |
|---|---:|---:|---:|
| BSS, frozen onto 2024 | 901.73 | 899.42 | **−2.31** |
| BSS, in-sample on source | 597.26 | 602.73 | +5.47 (not evidence) |
| BSS, B1S in-sample | 965.54 | 965.55 | +0.01 |
| sign agreement vs shipped (B1S) | — | 8/8 | — |
| runtime | — | CPU seconds | — |

## Prediction correlation

Not applicable — an additive per-bin offset on eight bins. It can change
ordering only within the ±0.01 band it moves, and the C-1 table shows the new
offsets sit on top of the old ones.

## Verdict

**KEEP the shipped lookup. DROP the refit.**

## Reasons

1. It reproduces to 1e−4 on the surface that matters, so there is nothing to
   correct.
2. A refit from the prior season transfers 2.31 worse, and every shrinkage
   setting tried stays below it.
3. The axis is worth +8.03 transferred against a +17.40 ceiling, so it is one
   of the larger surviving levers — which is a reason to leave it alone, not to
   tune it.

Together with EXP-A this is the same finding twice: on this data, constants
fitted on one season and frozen onto the next lose to the constants already in
the script. v12's LB −18.271 on this exact lookup is the third instance.

## Next

EXP-D, expected pitch mix.

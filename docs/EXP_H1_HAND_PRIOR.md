# H1 — pitcher x batter-hand hierarchical prior

## Existing-implementation search

Specified in `H1_HAND_MATCHUP_METHOD.md` (supplied outside the repo). Searched
`src/` and `tools/` for `hand_dev`, `ph_hier`, `H1_delta`, `personalized_prior`,
`hier_delta`: **zero hits**. No SETTLED line. `H1B22` in the ledger is an
unrelated plain 2022 baseline; `TH1_hl2` is te-halflife. Never built, never run.

## Hypothesis

CatBoost holds all three inputs — `te_pitcher_batter_hand_ratio_dev` (= the
method's `hand_dev`), `std_season_prior`, `std_pitcher_n` — and must rebuild
`(clip(prior·hand_dev, 0, 1) − prior)·80/(n+80)` out of splits. `src/skill.py`
exists on exactly that argument and measured it: a learned linear combination
explained 59.0% of the pitcher-skill target where a GBDT on the same inputs
managed 46.5%. Handing the model the composed quantity should help.

## Baseline

`B1S` (submission surface, `--p1`), the current champion's member family.
`--feat-h1` replaces `std_asof_pitcher_success_rate_delta` 1:1, per the method.
Feature count stays 121 (`H1 hand-matchup prior: +1 / -1`).

Pre-run sanity, computed independently on all 1,475,092 train rows:

```
hand_dev   finite 68.40%   mean 0.9989  sd 0.0404  p1/p99 0.893/1.104
H1_delta   mean -0.000139  sd 0.005516  p1/p99 -0.0189/+0.0179  zeros 31.60%
corr(H1_delta, hand_dev)  +0.7322      <- not a copy of the existing column
```

The 31.6% zeros are the method's own rule: no left/right history, no invented
lean.

## Results — 6 paired seeds, val2024

| arm | Δ | SE | t | 95% upper | per seed |
|---|---:|---:|---:|---:|---|
| **base** | **+3.88** | 2.82 | +1.38 | +11.12 | +9.36, −2.21, −5.64, **+12.43**, +6.40, +2.94 |
| cell | **−1.55** | 1.28 | −1.21 | +1.74 | −2.09, −3.42, +3.52, +0.94, −4.44, −3.84 |
| core (H1 on both) | +0.63 | 1.35 | +0.47 | +4.09 | — |

**The sign splits by family.** Largest single-feature effect measured on `base`
so far, and negative on `cell`. Mechanically consistent: the cell family
reconstructs P(success) from 14 cells, so a scalar prior adjustment buys it
less than the `std_..._delta` column it displaces costs — and dropping delta
families is already known to cost (`local-targeted-pruning`, −4.99).

## The mixed blend, and why it is not adopted

The two families are separate models, so H1 can be applied to `base` alone:

| blend | core | post, debiased | LB est | vs champion |
|---|---:|---:|---:|---:|
| B1S base6 + B1S cell6 (shipped) | 929.49 | 968.51 | 1107.54 | −0.00 |
| B1S base8 + B1S cell6 | 930.12 | 969.40 | 1108.43 | +0.89 |
| **H1 base6 + B1S cell6** | 931.32 | 971.39 | **1110.42** | **+2.87** |
| H1 base8 + B1S cell6 | 931.15 | 971.27 | 1110.30 | +2.76 |
| H1 base8 + H1 cell6 | 929.79 | 970.29 | 1109.32 | +1.78 |

Paired, mixed blend against the shipped one:

```
6 seeds   +1.99  SE 1.39  t +1.44  95% upper +5.56
8 seeds   +1.28  SE 1.14  t +1.13  95% upper +3.97
```

**PARK.** Three reasons, in order of weight:

1. **The paired statistic does not clear the bar** and gets *worse* with more
   seeds (t 1.44 → 1.13). Adoption needs Δ ≥ +3 and t ≥ 2.4.
2. **Picking the best of five blends is selection on the target.** The +2.87 row
   is the maximum of a table I computed after seeing the results, and the
   6-vs-8-seed choice inside it flips the answer by 0.11. `--feat-id-cohort` is
   closed for exactly this: "the A100 seed3 +8.13 with both axes together is
   self-selection on tree-path/early-stop variation".
3. The ensembled number (+2.87) is a single realisation; the paired mean (+1.28)
   is the disciplined estimate of the same quantity, and they disagree by more
   than either standard error.

Kept behind `--feat-h1`, default off.

## What would change the verdict

A base-only arm at 8+ seeds judged on the *base family alone* rather than on a
hand-picked blend. `base` at +3.88 with SE 2.82 is the one number here worth
more seeds: it is above the adoption threshold in magnitude and only fails on
precision. That is a different experiment from this one and should be
pre-registered as base-only before it runs.

Also untested: the method specifies a 1:1 replacement, so this arm bundles
"remove a known-useful column" with "add a new one". An additive variant
(keep `std_asof_pitcher_success_rate_delta`, add `h1_hand_delta`) separates
them and is one flag away.

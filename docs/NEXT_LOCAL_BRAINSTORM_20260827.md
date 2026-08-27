# Next local research — three CPU gates (2026-08-27)

## State before proposing anything

- Repository synced at `5a2ca78`; `LEDGER.tsv` has 852 rows.
- Champion remains `submissions/gskdep_0816.zip`, official LB
  `1111.3713632162`.
- Laptop environment matches the evaluation image on the critical stack:
  Python 3.11.15, pandas 2.0.3, numpy 1.26.4, sklearn 1.8.0. The RTX 5060
  Laptop GPU is idle, but all three proposals below start CPU-only.
- The last audit, `TM_TYPE_COMMAND_MIX`, is closed: the historical type-command
  structure is real, but both the raw and champion-unique direction flip sign
  between the two clean boundaries.

The score gap is not plausibly a calibration or blend-weight gap. At the current
score, +33 BSS needs an orthogonal residual correlation of about .018 and +65
needs about .026. The remaining candidates therefore have to expose information
that is not merely another deterministic transform of the existing 121/123
columns. Every gate below measures the champion-unique component and uses a
procedure-matched null before any GPU fit.

## Priority 1 — `CTX_ADJ_BATTER_PRESSURE`

### Hypothesis

The official batter success history averages together two effects: the batter's
own ability to induce difficult targets and the quality/context of pitchers he
faced. Estimate a context-and-pitcher expectation for each historical pitch,
subtract it from the outcome, and shrink the historical residual by batter.
The resulting frozen lookup asks whether a batter systematically makes command
harder or easier after pitcher quality and game context have been removed.

This is not `CTX_ADJ_PITCHER_SKILL`: that audit excluded pitcher identity from
the nuisance model and aggregated residuals by pitcher. Here pitcher history is
part of the nuisance adjustment and the residual is aggregated by batter. It is
also not `te_batter_ratio`, which shrinks the raw outcome without removing the
pitcher/context schedule.

### CPU gate

1. Use one fixed simple nuisance model; no context/model sweep.
2. Build 2022->2023 and 2023->2024 source-only/frozen boundaries.
3. Report coverage, variance, correlations with raw/std batter history and
   `te_batter_ratio`, champion reconstructibility, and champion-unique variance.
4. Run the full procedure-matched 400-permutation null on the unique component.
5. Require the same sign on both boundaries, latest boundary above null p99,
   and no R/F or early/late sign collapse before licensing a model feature.

### Expected cost / risk

About 30--45 minutes CPU after reusing the existing CTX audit frame. Main risk:
the batter block's next-season retention was only .37--.43, so the result may
show that the apparent batter effect is schedule composition rather than skill.

## Priority 2 — `BATTER_ARSENAL_FAMILIARITY`

### Hypothesis

Pitchers do not use an arsenal in a vacuum. A batter who has historically seen
mostly fastballs may induce different target difficulty against a breaking-ball
heavy pitcher than a batter with matching exposure. Build a source-only batter
pitch-family exposure vector from historical Trackman, compare it with the
current pitcher's **official row-local** ASOF pitch-mix vector, and use only a
fixed familiarity/mismatch score (for example Jensen-Shannon divergence plus
same-family overlap fixed before looking at residuals).

This is not `TM_TYPE_COMMAND_MIX`, which estimates pitcher command success by
pitch family, and it is not expected-pitch-mix context, which adjusts a pitcher's
mix by count/hand context. It is a batter-side historical exposure x current
pitcher-arsenal matchup. It is also not exact PB or graph topology: two players
need not have faced each other before, so it can cover unseen pairs.

### CPU gate

1. Reuse the exact player linkage already audited; no fuzzy rematching. Build
   batter exposure only from Trackman seasons before the target.
2. Use the official fastball/breaking/offspeed three-vector on the current row;
   never infer current pitch type and never consult another evaluation row.
3. Fix one shrinkage convention and the two algebraic scores before residuals;
   no taxonomy, distance, k, hand or context sweep.
4. Report batter/year coverage, exposure-vector persistence, unseen-pair coverage,
   correlations with batter ASOF and exact-PB, champion reconstructibility and
   champion-unique variance.
5. Residualise source OOF and target frozen; compare the complete procedure with
   a parameter-count-matched 400-permutation null on both clean boundaries.
   Require stable sign, latest p99 and segment stability.

### Expected cost / risk

About 30--50 minutes CPU, with the linked Trackman cache reused. Main risk: what
a batter has seen may be stable but orthogonal to command error, exactly as the
pitcher arsenal embedding was. The key distinction is matchup mismatch rather
than arsenal quality, and the matched-null gate decides whether that distinction
contains anything real.

## Priority 3 — `LOWRANK_PITCHER_COUNT_RESPONSE`

### Hypothesis

`te_pitcher_balls_before_strikes_before_*` estimates each pitcher-count cell
separately, while `skill_pc_hat` emits one scalar for the current count. Neither
explicitly shares strength through the empirical covariance of all 12 count
states. A source-only low-rank empirical-Bayes profile can learn stable pitcher
response modes such as ahead/behind/two-strike command and borrow information
for sparse cells without a high-order categorical sweep.

This is distinct from adding another count category (`FCC1`, closed), from
`te_ps`/`te_pchh` (closed), and from a generic skill-axis sweep. The audit's
object is the frozen 12-vector covariance and its champion-unique component,
not a new count key.

### CPU gate

1. Fix rank 2 before reading any score; use the existing k80 convention and no
   rank/k sweep.
2. Build pitcher x 12-count success profiles from seasons before the target,
   freeze the source basis, and expose only the current row's score plus one
   profile-amplitude coordinate.
3. Report year-to-year subspace/pitcher persistence, sparse-cell coverage,
   correlations with `te_pc` and `skill_pc_hat`, and champion reconstructibility.
4. Run source-OOF/target-frozen unique-signal transfer against a fully matched
   400-permutation null on 2022->2023 and 2023->2024.
5. Require stable sign, latest p99, and non-negative R/F and early/late before
   any CatBoost arm.

### Expected cost / risk

About 35--60 minutes CPU. Main risk: the existing count TE and `skill_pc_hat`
already consume the useful part. Rank is fixed because choosing it after seeing
the two boundaries would turn the audit into a representation sweep.

## Execution order and stop rule

Run only Priority 1 first. If it fails, run Priority 2; then Priority 3. Do not
combine weak positives and do not start a GPU experiment from raw rho alone.
The first candidate to pass all novelty, matched-null, transfer and segment gates
gets a separately written one-change GPU preregistration. If none passes, stop:
the result is that the remaining legal official-data transformations do not
contain a detectable champion-unique signal at the present noise floor.

# Research Triage — 2026-08-27

## Champion / integrity

- champion: GSKDEP + E-LB1, frozen
- LB: `1111.3713632162`
- artifact: `submissions/gskdep_0816.zip`
- repository at audit start: `e08e878`, clean after `tools\agent_sync.cmd start codex`
- no live job: no local Python process or licensed GPU job was started by this audit
- row-independence contract: unchanged; every proposed lookup must be frozen from
  official training rows and must return the same value for an isolated test row
- scope: the three candidates in the supplied master prompt followed by the three
  candidates in `docs/NEXT_LOCAL_BRAINSTORM_20260827.md`; first valid STOP rule wins

## Candidate 1 — `PB_MULTIYEAR_EXACT_POOL`

### Provenance

- current PB source seasons: **2024 only** for the shipped 2025 lookup
- actual inference source: `src/script_blend_b1s.py`, rendered by
  `tools/prepare_gsk_package.py` to `out/script_blend_gsk.py`
- exact generator: `tools/make_matchup_constants.py`
- packaged lookup: `model/matchup_constants_2024.npz`; its local and packaged
  SHA-256 are both
  `e64dac3d4580eb84511e3bdd1c23b4645881e67e2cc0630101dbfb746b47993c`
- key: exact `(pitcher_id, batter_id)` only
- source prediction: 2024 OOF ensemble from `VB2_base` seeds
  `42,7,13,3,4,5,6,8` and `ZD5` seeds `42,7,13,3,4,5`, blended 0.45/0.55,
  followed by the frozen slope, legacy shift and recent-middle correction
- fitted residual: `control_success - (postprocessed blend + recent-middle)`;
  the residual is globally centred before aggregation
- aggregation and shrinkage: observation-level residual sum and count,
  `offset = sum_residual / (n + 500)`; the source-applied offset mean is removed
- missing pair fallback: `0`
- correction operator: additive
- application order: blend -> slope 1.0416 -> subtract 0.0052 -> clip ->
  recent-middle -> exact PB (`pb0_*`) -> E-LB1 -> final clip
- duplicate status: **not a duplicate**. The shipped correction is one-season,
  while the proposed all-completed-season sufficient-statistic pool is not in
  the current generator or packaged artifact.

The packaged script uses `pb0_pitcher`, `pb0_batter`, and `pb0_offset`. The later
`pb_*` table also produced by the generator is not the table consumed by the
champion. This distinction was checked in the packaged script rather than inferred
from an older report.

### Coverage — structural upper bound only

These counts use only the exact raw pair keys in official training data. They show
why the idea is interesting, but they are **not** a performance result because a
valid pooled residual table cannot yet be formed.

| target | current preceding-season coverage | all-history `< target` coverage | expansion | current source unique pairs | current support p25/p50/p75 | all-history unique pairs | all-history support p25/p50/p75 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 0.413254 | 0.541020 | +0.127767 | 25,074 | 4 / 7 / 12 | 69,307 | 5 / 8 / 17 |
| 2024 | 0.380972 | 0.511587 | +0.130616 | 25,585 | 4 / 7 / 12 | 82,297 | 5 / 9 / 18 |

For reference, the shipped 2024 source contains 26,355 exact pairs with support
p25/p50/p75 `4 / 7 / 12`.

### Source parity audit

The exact all-history candidate requires comparable untouched/OOF residuals for
every historical source season. The local artifact inventory does not satisfy that
contract:

- `BND22_base/cell` provides a six-seed 2022 source and 2023 target, but its cell
  arm includes `--feat-skill`.
- `B1J6_base/cell` provides a six-seed 2023 source and 2024 target, but its cell
  arm is the standard cell generation and does not carry that same feature contract.
- comparable six-seed OOF residual artifacts for 2019, 2020, and 2021 are absent.
- older `H21`/`HB22`/`HC22` ledger references do not supply a complete local,
  same-generation artifact chain and therefore cannot be substituted.

Consequently, pooling the available residuals would mix model/fpipe generations,
and target 2023 could not represent “all completed seasons `< 2023`” at all. Making
the missing residual series honestly would require a new rolling OOF training
project, contrary to the CPU-only scope. Raw pair counts cannot replace residual
sufficient statistics.

### Direct transfer

2022->2023: **not measured** — the multiyear candidate lacks comparable 2019-2021
source residuals.

2023->2024: **not measured** — the candidate would mix the `BND22` and `B1J6`
cell/pipeline generations and still lack earlier comparable source residuals.

The existing current-PB result (`+3.80` on 2023->2024, 38.1% coverage) remains
valid evidence for the shipped one-season lookup. It is not evidence for the
unconstructed multiyear arm and was not reinterpreted as such.

### Decomposition

- same-coverage: unavailable without a valid candidate residual table
- coverage-expansion: raw-key ceiling is about +12.8pp / +13.1pp on the two target
  seasons, but its Brier effect is unknown and must not be inferred from coverage

### Verdict

**HOLD**

Reason: the mechanism is distinct and legal in principle, but the required
source-prediction parity cannot be guaranteed. Section 5C and the explicit section
11 STOP condition require an immediate stop rather than a mixed-generation pool,
a fabricated residual series, or a large unregistered GPU rebuild.

## Candidate 2 — `SKILL_ESTIMATOR_DISAGREEMENT`

**NOT EXECUTED.** Deferred by Candidate 1's source-parity STOP. Formula remains
fixed as `skill_pc_hat - skill_hat`; no novelty or residual score was inspected.

## Candidate 3 — `ASOF_EVENT_STATE`

**NOT EXECUTED.** Deferred by Candidate 1's source-parity STOP. No event-state
feature was generated and no duplicate verdict was manufactured.

## Included local queue — not executed

The user's three previously registered candidates are included after the master
prompt's three priorities, without changing their preregistered order:

4. `CTX_ADJ_BATTER_PRESSURE`
5. `BATTER_ARSENAL_FAMILIARITY`
6. `LOWRANK_PITCHER_COUNT_RESPONSE`

None was executed because the master prompt says to stop immediately when PB
source parity cannot be guaranteed and says not to opportunistically start a
fourth idea. Candidate 5's Trackman use is also incompatible with the master
task's “Do not touch Trackman again” rule; it remains in the separate local queue
only, not in this audit.

## Final recommendation

**No GPU candidate is licensed.** Candidate 1 is a genuine but currently
unmeasurable HOLD, not a FAIL or a PASS. Candidates 2-6 remain unmeasured because
the supplied contract explicitly stops at this condition.

[FACT]

- The shipped PB lookup is exact-pair, one-source-season, additive, k=500, and
  consumes `pb0_*` from the packaged artifact.
- All-history raw exact-pair membership would expand target coverage by roughly
  13 percentage points on both clean targets.
- The repository lacks a complete comparable OOF residual chain for all source
  seasons, and the two available six-seed boundaries do not share the same cell
  feature contract.
- No GPU, model fit, submission, champion constant, or prediction was changed.

[INFERENCE]

- The extra exact-pair coverage is potentially useful but cannot be scored
  honestly from raw pair membership alone.
- Combining the available arrays would confound multiyear pooling with a model
  generation change, so any apparent gain or loss would be uninterpretable.

[NEW HYPOTHESIS]

None executed. Reopening Candidate 1 would require a separate preregistration for
a same-generation rolling OOF residual archive covering every source season. That
is a new GPU project, not a continuation of this CPU audit.

# LUPI + Batter Trackman preregistration — 2026-08-27

This records the user's add-on CPU-only prompt after the six-axis queue. No GPU,
submission, leaderboard use, calibration, PB or champion change is allowed.

## Candidate 1 — `PRIVILEGED_GAP_AUDIT`

### Duplicate audit

Not duplicate. The closest work is:

- `src/teacher.py` / `--soft-target`: predicts `y`, then distils that probability;
  it never defines or predicts `q_priv-q_legal`, and its row-random seasonal leak
  is recorded in SETTLED.
- masked-pitch auxiliary: predicts current pitch family as an auxiliary task; it
  does not build a privileged success increment.
- TM2COMMAND/TM_DIST/TM_TYPE_COMMAND_MIX: frozen historical pitcher summaries,
  not current-pitch privileged information distilled into a legal gap.

No code/artifact was found satisfying all four required properties: explicit
privileged increment, legal-X gap learner, frozen next-season apply, and champion
residual transfer measurement.

### Fixed implementation

- exact/high-confidence current-pitch linkage only; never force multi-row matches
- boundaries: train seasons below 2023 -> evaluate linked 2023, then below 2024 ->
  evaluate linked 2024
- legal X: the boundary pack's legal numeric champion frame; categorical context
  may be ordinal/one-hot encoded once, identically in both teacher arms
- privileged Z: `pitch_type_group` one-hot plus every stable numeric current-pitch
  Trackman schema column among speed, spin, movement, release, extension and
  historical command/location coordinates; select by documented schema/name and
  dtype before reading outcomes, no subset sweep
- q_legal and q_priv: same fixed low-capacity estimator family and settings
- gap: `q_priv-q_legal`
- legal gap learner: fixed Ridge(alpha=100), source cross-fit and frozen target
- matched null: 400 deterministic repetitions; repeat gap-label permutation,
  source cross-fit/frozen apply, champion reconstruction when required, and final
  residual direction
- report teacher gain, gap variance, source OOF gap R2, target gap correlation,
  champion reconstructibility, unique component when R2>=.50, R/F, early/late
- linkage-bias table: season, R/F, count, target rate, pitcher/batter frequency,
  `asof_pitcher_n`, `asof_batter_n`; summaries only, no weighting

PASS requires every gate in the user's add-on prompt. PASS writes a separate GPU
preregistration and stops; it does not launch GPU.

## Candidate 2 — `BATTER_TRACKMAN_AUDIT`

Run only if Candidate 1 does not PASS.

### Existing work distinction

`BATTER_ARSENAL_FAMILIARITY` is now FAIL and used only the three-family historical
exposure versus current pitcher's official ASOF mix. It does not close a full
batter historical Trackman profile containing speed, movement, release and
location exposure. Pitcher-centric TM_DIST does not close the batter entity.

### Fixed implementation

- reuse official mapping only (`src/link_batters.py` / map2); no names or threshold
  relaxation
- report mapping counts, ambiguity and row/season coverage before scores
- for each target S, use Trackman seasons `<S` only
- fixed raw batter profile: pitch-family shares and mean/std of all stable numeric
  speed, horizontal/vertical movement, release, spin, extension and location
  fields actually present in the schema
- fixed PCA-8, reusing TM_DIST dimensionality; source standardisation/PCA frozen
- persistence: 2021->22, 2022->23, 2023->24 CCA/component correlations
- champion grouped-CV reconstruction and source-OOF/frozen-target unique profile
- fixed Ridge(alpha=100) residual direction and 400 batter-structure matched nulls
- support report at n<50/n<100/n<300; deterministic shrink/fill, no threshold sweep

Write final results to `docs/LUPI_BATTER_TM_AUDIT_20260827.md` and then stop.

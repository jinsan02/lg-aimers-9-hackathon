# BND and resolution research report — 2026-08-16

## Codex first pass (provisional)

Champion remains `submissions/gskdep_0816.zip`, LB **1111.3713632162**. No
submission was made and no champion constant, member, or blend weight changed.

## BND execution and integrity

`AimersBND` completed 24/24 fits on `DESKTOP-053T952` in about **46 minutes**.
All artifacts were preserved and the 24 ledger rows were merged locally.

The run cannot be consumed as the preregistered two-boundary prerequisite:

1. The document title and motivation requested the missing 2021->2022 and
   2022->2023 boundaries. The runner actually built **BND22 (2022->2023)** and
   **BND23 (2023->2024)**. The 2021->2022 boundary remains missing.
2. BND23 omitted `--drop-f-pre 2022`. This directly violates SETTLED FLAG
   `drop-f-pre-omitted`, which says every `--val-season 2023` judging run needs
   that flag. The observed signature is the known failure: base best iterations
   **11--19**, base BSS **38--46**, and cell BSS **0**.

The strengthened integrity gate now treats that flag as part of the artifact
contract and fails BND23 loudly. BND22 passes. The clean latest comparison is
the existing B1J6 base/cell pair on the same 5070 host and six seeds, with
`--drop-f-pre 2022`.

Consequently the usable map is:

| Boundary | Evidence | Status |
|---|---|---|
| 2021->2022 | none | **MISSING** |
| 2022->2023 | BND22 base/cell, six seeds | valid |
| 2023->2024 | B1J6 base/cell, six seeds | valid |
| 2023->2024 | BND23 base/cell | **INVALID; never compare** |

The original BND files were retained for incident evidence. No invalid tag was
added to `docs/INVALIDATED.tsv`: that registry is the temporal future-row
contamination gate, whereas this is a known regime/flag contract violation.

## Seven downgraded baseball axes

These are not new adoption trials. They are fixed CPU replays on the two valid
boundaries. Because the promised first boundary is absent, none is represented
as a completed three-boundary verdict.

| Axis | 2022->2023 | 2023->2024 | Provisional reading |
|---|---:|---:|---|
| historical lineup, `role*same_hand`, k1000 | -24.948 | +0.204 | sign flip |
| recent middle state | +2.638 | -1.477 | sign flip |
| recent success state | -2.732 | -5.063 | consistently negative |
| recent combined state | -2.141 | -5.775 | consistently negative |
| workload pace | +27.840 | -2.072 | sign flip; first leg driven by F (+267), R -0.077 |
| intent-execution | +0.231 | -2.530 | sign flip |
| adaptive PB shrinkage | -1.761 | -0.293 | consistently negative |
| prior PA-depth proxy | -0.508 | +0.610 | sign/segment instability |

Team-call style was audited directly rather than through BND. Combined
year-over-year correlations are .192/.253/.414 for 2021->22, 2022->23 and
2023->24. Components still flip and there is no error-aligned increment, so it
does not justify a GPU candidate.

No axis meets the preregistered recurrence rule. The missing 2021->2022 run
cannot rescue any axis that already disagrees between the two available clean
boundaries. It could only add formal evidence to the already-negative recent
success/combined and adaptive-PB axes, so it is not recommended as a score
search.

## Existing queue audit

- `--feat-window + cells`: already CLOSED; A100 +0.51 versus 4070 -5.85.
- `--baseline-col skill_pc_hat`: already CLOSED at -11.91 on the same 4070 seed.
- `xgb-current-121-blend`: downgraded provenance, but the strong v11 analogue
  adds only +0.094 at 2% and is negative in R/late segments.
- `--std-k 120`: already CLOSED at -4.61 (t=-1.40); k200 is -19.98.
- dropping `li,wexp`: already CLOSED at -3.50 (t=-1.19).
- `--skill-neutral-mode const`: already CLOSED, core +0.58 (t=.56).
- `--grow`: already measured in E64; Depthwise/Lossguide were about -63/-26.
- `--ptype`: not a clean untried axis; the predicted-pitch-type family and its
  nearby Trackman-context variants are already closed.

This queue contains no justified GPU work.

## New resolution-oriented research

All candidates were screened with a frozen fit-on-source/apply-to-target
protocol and a 400-repetition matched null. No GPU was used.

### TM2COMMAND supervised embedding — FAIL

A fixed 100-dimensional pitcher-season Trackman summary predicts four next-year
command profiles (success/middle/ball/reverse) using one multivariate Ridge
alpha=100. It is only partly reconstructible from champion features, but does
not transfer into champion error:

- 2022->2023: frozen rho **+.008331**, null p99 **.015916**, percentile 80.25.
- 2023->2024: frozen rho **-.003263**, null p99 **.007815**, percentile 67.5.

The sign flips and the latest R/F and both halves are negative.

### Trackman minus official-ASOF disagreement — FAIL

The four deltas between Trackman command estimates and row-local official ASOF
command are substantially duplicable (CV R2 .18--.53) and land below noise:

- 2022->2023: rho **+.001935**, null p99 .038515, percentile 3.75.
- 2023->2024: rho **+.000452**, null p99 .007766, percentile 11.25.

### Conditional Trackman geometry — FAIL

This preregistered representation uses 35 condition deltas (batter hand L/R,
3-ball, 2-strike, neutral; seven physical means), shrink k50, fixed PCA-4.
It is persistent and genuinely new, yet not error-aligned strongly enough:

- 2022->2023: rho **+.011199**, null p99 .015407, percentile 93.25; F negative.
- 2023->2024: rho **+.004256**, null p99 .007186, percentile 84.5; late ~0.

Neither clears the preregistered null p99 gate. No dimension, condition, k, or
feature-subset sweep is permitted after seeing these results.

### ASOF trajectory

Not rerun. It is the already-closed `MULTIVARIATE_LATENT_STATE` axis: frozen
rho .00625/.00668 but only the 75.5th/64.2nd percentiles of its matched null.

## Error-aligned signal map

| Family | Best honest evidence | Remaining hole? |
|---|---|---|
| career shape | career-middle LB -18.27; latent state inside null | no |
| reliability/uncertainty | best reliability arms +0.58/+2.26 with weak t; posterior uncertainty closed | no |
| matchup deviation | exact PB is already champion; learned PB feature -17.56, hand residual -191.95, MF ceiling +0.501, adaptive k negative | no |
| failure composition | cell route remains useful, but posterior stack -52.38 and reweight/reparametrize/project/delete variants fail | no |
| cold transition | cold rows have higher, not lower, resolution than warm rows; routing gives no gain | no |
| Trackman-vs-ASOF disagreement | percentiles 3.75/11.25 against matched null | no |

## Recommendation and requested review

**GPU candidate: NONE.** Do not launch a run merely to complete the queue and do
not rebuild BND21 for candidate selection. The next single action is independent
Claude review of (a) the BND contract failure, (b) the valid BND22+B1J6 replay,
and (c) the three matched-null FAILs. Claude should decide whether to append
formal SETTLED closures or request one narrowly specified corrective audit.

Questions for Claude:

1. Confirm BND23 is unusable despite the preregistration text explicitly saying
   “No `--drop-f-pre`”; the higher-priority SETTLED BANNED flag and the observed
   11--19-tree signature contradict that sentence.
2. Confirm the missing 2021->2022 boundary is unnecessary for score search once
   every plausible candidate already fails recurrence across the two valid legs.
3. Confirm no GPU candidate is licensed by TM2COMMAND, disagreement, conditional
   geometry, or the existing queue.

# Agent B — FUTURE3_SUCCESS_STATE, success-supervision geometry

**Verdict: FAIL.**

## Provenance of this report

The measurement is Codex's. `out/subagent_b_future3_audit.json` was written
complete at 02:30 on 2026-08-28; the session then ended before the report
existed, and Agents A, D and E produced no artifact at all. This document is
Claude reading that finished JSON and rendering the verdict the prompt asks for.
**No number here is new** — every figure is quoted from that file, and the file
is the audit trail. Isolation is preserved: nothing from any other agent enters
below, because no other agent produced anything to enter.

## The candidate

For the dominant success cell `1000` only, look at the next three pitches by the
same pitcher inside official train and bin the count of successes: `LOW` 0-1,
`MID` 2, `HIGH` 3. The success bit is untouched, failure cells are untouched,
other success cells are untouched, and inference sums the success subtypes, so
no future information is needed at prediction time.

## Structural health — passes

| quantity | value |
|---|---|
| official rows | 1,475,092 |
| dominant `1000` rows | 540,700 |
| complete (future3 available) | 540,192 — **99.906%** |
| end-of-history missing | **508 rows, 0.094%** |
| subtype counts | LOW 237,513 / MID 209,238 / HIGH 93,441 |
| subtype prevalence | .4397 / .3873 / .1730 |
| subtype entropy | 1.0322 nats, **2.807** effective classes |

Coverage is excellent and no class is small. The residual 508 rows retain the
original `1000` name, so the taxonomy gains a sixth success name that is almost
empty — workable, but it is the same near-empty-class hazard `--fm-coarse`'s
implementation had to be built around (`src/failmode.py`, the `reserve` set),
and it would need the same care.

## Why it fails

### 1. The subtype is dominated by the one trivial feature — the prompt's own named disqualifier

This is the decisive measurement. Multiclass log-loss skill against a constant,
fitted on the source era and frozen onto the target season:

| boundary | full legal numeric + low-card categorical | `asof_pitcher_success_rate` **alone** | incremental |
|---|---|---|---|
| ≤2022 → 2023 | **−0.00185** (worse than a constant) | +0.00398 | **−0.00583** |
| ≤2023 → 2024 | +0.00381 | +0.00633 | **−0.00252** |

Macro OvR AUC tells the same story: .5357 / .5465 for the entire legal feature
set against .5350 / .5393 for the single column.

**Read this as a probe result, not an information result.** The estimator is a
linear `SGDClassifier(loss="log_loss", alpha=1e-4)` on standardised numerics
plus one-hot categoricals (`out/_subagent_b_future3_audit.py:77`), and the
single-column arm is fed unscaled, so the two arms do not carry identical
effective regularisation. A linear probe with a hundred-odd correlated columns
losing out-of-time to a one-column probe is a plausible regularisation outcome
on its own, and the champion is a gradient-boosted tree that could find
structure this probe cannot. What the measurement licenses is the narrower
statement: **a frozen linear probe finds nothing in the subtype beyond
`asof_pitcher_success_rate`, and that column is already shipped.** That is the
check the prompt named and the same class of pre-screen used everywhere else
here, but it is not proof that no information exists.

One thing it does get right that matters: `boundary()` applies the champion's
`--drop-f-pre 2022` contract, so the source era matches the shipped recipe.

The direct mutual-information check agrees and quantifies how little is left:
`I(skill decile; subtype) = 0.0086` nats, mean skill by subtype .5323 / .5440 /
.5537, LOW share falling monotonically .529 → .314 from skill decile 0 to 9. The
gradient is real, shallow, and already shipped.

### 2. The target definition itself is non-stationary

LOW share by season: **.386 (2019) → .434 → .428 → .430 → .475 → .496 (2024)**,
with HIGH falling .206 → .139. That is an 11-point drift in what the label
*means*, in the same direction as the drift the model must transfer across.

### 3. The league gap reverses

Pooled over all seasons, F is much less LOW than R (.321 vs .456) and much more
HIGH (.271 vs .159). Restricted to 2023-2024, **F becomes more LOW than R (.528
vs .480)** and its HIGH share collapses to .116. A subtype whose league ordering
flips between the pooled and the recent window is not a stable supervision
target, and the champion is judged on the recent window.

### 4. It is not actually a test of the clue that motivated it

The legacy-geometry clue is a specific quantity: legacy has 5 success cells and
**2.122** effective success classes against corrected 3 and **1.856**. FUTURE3
produces **3.840** effective success classes — it overshoots the target geometry
by more than the gap it was meant to close.

```
corrected   3 success cells   effective 1.856
legacy      5 success cells   effective 2.122     <- the +4.15 observation
FUTURE3     6 success names   effective 3.840
```

So even a positive seed-3 result would not be evidence about Clue B, and the
legacy `+4.15` gives no calibration for what to expect here.

## The strongest argument against this verdict, and why it does not carry

`FMCOARSE` (2026-08-16) is direct evidence that **weak predictability does not
imply useless supervision**: the within-failure distinctions are also only weakly
predictable and conflict with the primary gradient at up to a 100% minibatch
rate, and deleting them still cost **−18.581** core. "The subtype is hard to
predict" is therefore not by itself a reason to reject a taxonomy split.

That counter-argument is stronger than it first looks, because the probe above
is linear and the failure-mode supervision would very likely fail the same
probe. **Point 1 alone would therefore leave this at HOLD. The verdict rests on
points 2, 3 and 4, which are descriptive statistics with no estimator in them.**

The distinction that decides it is what the label is *about*. The failure modes
(middle / ball / reverse) decompose **the current pitch's own outcome** — the
same event the row describes. FUTURE3 labels the current row with a property of
**other rows that come after it**. For that to shape the shared trees usefully,
the current row's features have to carry the future-pitch structure, and the
frozen-transfer measurement says they carry only the pitcher's mean skill, which
is already a shipped column. Points 2 and 3 then say the little that is carried
is not stable across the boundary the model must cross.

## Packaging note if anyone revisits this

FUTURE3 would be a second train-only label builder in the same class as
`src/failmode.py`, which is train-only precisely because the same algebra on
test rows recovers 2025 targets at 96.79%. Any implementation must never enter
a submission zip, and `tools/audit_rowindep.py` plus the package-content check
must be run before any packaging, not after.

## Not licensed, and what is banned

No GPU. The prompt bans horizon sweeps (1/3/5/10), threshold sweeps, alternative
bins and alternative cells, so a "corrected" variant aiming at ~2.12 effective
classes is **not** licensed by this report and would need its own preregistration
with a mechanism argument that survives the trivial-feature check above.

Source data: `out/subagent_b_future3_audit.json`.

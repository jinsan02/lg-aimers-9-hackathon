# F1 — F-league partial-pooling logit adapter

Pre-registered 2026-08-16 before reading any F1 candidate score. This is the
first item in `JunHyun_levers.md`'s model queue. It is not a league constant,
league-specific blend weight, F-only CatBoost, or league-relative feature
normalisation; all of those are already closed.

## Frozen question

Can a low-capacity residual head recover **within-F resolution** while leaving
every R prediction bit-identical?

The historical judging surface is fixed:

- source: models fitted through 2022, validation predictions on 2023;
- untouched target: deployment refits through 2023, predictions on 2024;
- baseline: `0.45 * B1SMOKE_base + 0.55 * B1SMOKE_cell`, joined by `row_id`;
- F rows only receive the adapter; R rows are copied exactly.

No target-2024 label, mean, distribution, coefficient or choice enters fitting.

## One model, one feature set

For F rows, with `p0` the fixed core:

```
p1 = sigmoid(logit(clip(p0, 1e-5, 1-1e-5)) + X beta)
```

There is **no intercept**, because an F constant shift is already closed and
cannot add resolution. `X` has exactly these six row-local, legal columns:

1. `std_asof_pitcher_ball_rate`
2. `std_asof_pitcher_ball_rate_delta`
3. `std_asof_pitcher_middle_rate`
4. `std_asof_pitcher_middle_rate_delta`
5. `balls_before`
6. `strikes_before`

The first four are produced by the frozen B1SMOKE base `fpipe` artifact; the
last two are official row inputs. On source-F only, replace non-finite values
with the source median, then centre and scale by source mean/std. A zero std is
an error. Store those 18 constants in the adapter artifact. Target rows only
apply the frozen transform; they never update it.

Fit one deterministic offset-logistic ridge:

```
sum_i BCE(y_i, p1_i) + 0.5 * 100 * ||beta||^2
```

using scipy L-BFGS-B, analytic gradient, initial beta all zero, maxiter 500,
`ftol=1e-12`, `gtol=1e-8`. No Ridge/depth choice, alpha sweep, feature subset,
interaction, coefficient cap, blend weight, temperature or post-hoc centring.

## Integrity gates

- exact elementwise `row_id` and target agreement for base/cell on both splits;
- host `DESKTOP-053T952`, seed 3, surface `val2023->test2024`;
- 121 ordered features and compatible B1SMOKE artifacts;
- all predictions finite and inside `[0,1]`;
- max absolute R-row change exactly zero;
- saved adapter -> fresh process -> full prediction max difference zero;
- reversal, half-frame and single-row drift exactly zero.

Any failure is INFRA FAIL, not a performance result.

## Performance gate

Report raw full-core BSS delta, F-only MSE contribution in global-Brier units,
F honest normalised resolution (fixed RNG 20260808), row halves, calendar
halves, R/F, reliability/resolution, RMS/correlation, beta and correction SD.

| untouched 2024 condition | action |
|---|---|
| full raw BSS delta < +10 | FAIL, close F1 |
| F honest normalised-resolution gain < +100 | FAIL, close F1 |
| either 2024 row half or calendar half is negative | FAIL, close F1 |
| all pass | build the same-spec 2024->2025 deployment adapter, then audit/package |

Submission candidacy is stricter: expected full gain at least +25 and every
integrity/package check must pass. Even then, request user approval before any
submission.

## Forbidden follow-ups

Changing alpha, feature subset, residual link or objective after the result;
adding an intercept; fitting a target-season constant; changing only one league
blend weight; trying a tree after ridge; using another test row; using 2025
Trackman or external statistics; submitting without approval.

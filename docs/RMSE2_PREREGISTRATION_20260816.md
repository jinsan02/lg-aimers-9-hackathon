# RMSE2 — corrected direct-Brier base path

Pre-registered 2026-08-16 after F1 closed and before changing the RMSE path or
reading a corrected RMSE result. This resolves the HOLD/INCONCLUSIVE record:
the old n=3 result (`+2.44`, t=.80) was underpowered, and code audit later found
that the RMSE deployment refit used the selection frame instead of `train_dep`.

## Single code correction

In `run_cat`'s `args.loss == "RMSE"` branch only, deployment refit must use:

- `train_dep[features]` rather than `train[features]`;
- `train_dep[TARGET]` rather than `train[TARGET]`;
- `_refit_weights(args, train_dep)` rather than selection weights.

Residual-baseline and label-smoothing algebra must use the same deployment
frame. Selection fit and validation predictions do not change. The already
implemented `fpipe.predict` regressor branch (`clip(model.predict(X),0,1)`) is
the only sanctioned inference path.

## Arms

Same 5070 session, judging surface fit<=2022 / val2023 / untouched2024, seeds
`3,4,5,6,8,13`, 121 features and P1 two-stage artifacts.

- fresh control `RMSE2CTL_base`: Logloss / eval Logloss;
- candidate `RMSE2CAND_base`: RMSE / eval RMSE;
- fixed cell in both cores: `B1J6_cell` same seed;
- fixed core: `0.45*base + 0.55*cell`.

All other flags match `B1J6_base`: depth8, lr .01, l2 10, border254,
refit-mult 1.5, drop-f-pre 2022, and the exact current feature recipe. No
common budget, calibration, member weight or hyperparameter change.

## Integrity

- same host/surface/six seeds;
- row_id and target elementwise identical across control/candidate/cell;
- strong fit-row and ordered-feature hashes agree;
- every candidate pack saves, loads in a new process and reproduces its stored
  full prediction; subset/reversal/single-row drift zero;
- control path must remain compatible with the current classifier route;
- deployment pack must carry the deployment row contract through 2023.

Any failure is INFRA FAIL and no performance conclusion is drawn.

## Gate

Primary statistic is the six paired fixed-core untouched-2024 delta.

- KEEP only at mean >= +3, paired t >= 2.4, seed-ensemble delta > 0, both row
  halves >= 0 and no large R/F collapse;
- DROP if Student-t 95% upper bound < +3;
- otherwise HOLD, with no tuning or seed extension.

Report base and core per seed, mean/SE/t/Student-t CI, median/sign count,
ensemble, source and target, halves, R/F, reliability/resolution, RMS/correlation,
best iterations, fingerprints and artifact parity.

No submission without explicit user approval.

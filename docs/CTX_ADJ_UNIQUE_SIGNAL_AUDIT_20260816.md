# CTX_ADJ_UNIQUE_SIGNAL_AUDIT — 2026-08-16

The fixed 121-feature reconstruction is StandardScaler + Ridge(alpha=100), at
pitcher-season level. Source `z_unique` is out-of-fold; target reconstruction is
fit on source only and frozen. Each of 400 matched-null repetitions permutes the
paired source/target skill direction and repeats the same OOF/reconstruction/
freeze/residual-fit procedure.

## Variance retained

- Var(z): **0.001480416**
- Var(z_unique): **0.000583444**
- retained: **39.4108%**
- corr(z_unique,z): **+0.434407**
- corr raw/std/skill_pc/skill_hat:
  **-0.033390 / +0.076580 / +0.006450 / +0.025077**
- independent OOF reconstruction R2 of z_unique from champion X: **0.085207**

## 2022 -> 2023

- rho: **-0.004603**
- null percentile: **38.75**
- p95 / p99: **0.017440 / 0.021704**
- R/F/early/late:
  **-0.011146 / +0.023801 / +0.006080 / -0.016088**

## 2023 -> 2024

- rho: **+0.002742**
- null percentile: **68.50**
- p95 / p99: **0.004947 / 0.006791**
- R/F/early/late:
  **+0.003000 / +0.000376 / +0.003875 / +0.001565**

## VERDICT: FAIL

The unique component is non-degenerate and statistically distinct, but it flips
negative on 2022->2023 and the latest boundary is inside matched-null noise.
Therefore the original CTX_ADJ is closed as **real raw signal whose transferable
portion is already represented by the champion**. No GPU candidate.

No GPU, CatBoost retrain, k/context/reconstruction/model/segment sweep, other
axis, submission, or routing was run.

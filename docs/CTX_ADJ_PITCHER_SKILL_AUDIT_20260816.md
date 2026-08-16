# CTX_ADJ_PITCHER_SKILL_AUDIT — 2026-08-16

One fixed CPU-only nuisance model: SGD logistic on legal row context, with no
pitcher identity or pitcher-history input. For season S, the model and k80
pitcher residual lookup use only seasons `< S`. Residual alignment uses the
valid BND22 and B1J6 boundaries and the existing 400-permutation matched null.

## CTX_ADJ_PITCHER_SKILL_AUDIT

Novelty:
- champion reconstructibility R²: **0.613212** (121-feature numeric frame,
  grouped CV by pitcher; preregistered **HOLD** band 0.50–0.80)
- corr raw pitcher success: **+0.718884**
- corr std pitcher success: **+0.525214**
- corr existing skill: `skill_pc_hat` **+0.713779**; shipped `skill_hat`
  **+0.595669**
- variance: **0.00160084**, not degenerate

Transfer:

2022 -> 2023
- rho: **+0.020314**
- matched-null percentile / threshold: **99.75th** / p95 **0.013448**,
  p99 **0.015311**
- R: **-0.003666**
- F: **+0.013446**
- early: **+0.031519**
- late: **+0.007811**

2023 -> 2024
- rho: **+0.007827**
- matched-null percentile / threshold: **99.25th** / p95 **0.006238**,
  p99 **0.007269**
- R: **+0.008598**
- F: **+0.000918**
- early: **+0.001853**
- late: **+0.016239**

Leakage / independence:
- **PASS**
- The context model and pitcher aggregation for S use official train seasons
  `< S` only. Deployment is a frozen train-derived `pitcher_id -> value` lookup;
  an evaluation row reads only its own pitcher_id and never another test row.

Existing-axis duplication:
- **NONE (but moderate redundancy)**
- Unlike `skill.py`, its target is not rest-of-season success and its inputs are
  not pitcher ASOF/std rates. Unlike strong-residual/dynamic-hierarchy heads, it
  does not fit champion residuals. It subtracts a pitcher-free context nuisance
  estimate from historical outcomes before pitcher aggregation. Nevertheless,
  R² .613 and correlations .596–.714 show substantial overlap with existing
  pitcher-history skill.

VERDICT:
- **HOLD**

한 줄 결론: 이 축은 두 경계의 전체 rho가 matched-null p99를 넘는 실제 신호지만,
재구성 R²가 사전 정의된 HOLD 구간이고 R/F·early/late 안정성이 부족하기 때문에
다음 GPU 후보로 **보류한다**.

No GPU, seed experiment, feature implementation, submission, coefficient/model/
context subset, or shrinkage sweep was run.

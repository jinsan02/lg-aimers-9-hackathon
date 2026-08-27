# Agent C — strict temporal consensus target

**Verdict: FAIL.** The two clean boundaries disagree in sign, and the two matched
nulls differ by 23x, so "clears p99 on each" is not one statement.

## Provenance

The harness is Codex's (`tools/_tmp_temporal_consensus_audit.py`, written 02:28
on 2026-08-28). It prints its result to stdout and writes no file, so when the
session ended nothing survived. Claude re-ran it unmodified with output captured
to `out/temporal_consensus_audit.log`; the parsed result is
`out/temporal_consensus_audit.json` (320 s, CPU only).

## The candidate

For a row in season S, fit three teachers on seasons ≤ S−3, ≤ S−2 and ≤ S−1 and
average their predictions. Signal common to several historical cutoffs should be
more invariant than one latest teacher. This is not the closed distillation
axis: `src/teacher.py:104` uses a row-random K-fold across all seasons at once,
so it never implemented strict temporal legality.

Teacher: fixed `SGDClassifier(log_loss, alpha=1e-5, max_iter=30, average=True)`
on the strict-cutoff 121-feature base pipeline. Consensus is the equal mean.
Reconstruction is 5-fold OOF / source-frozen Ridge(alpha=100) on the champion's
112 numeric features; the null is 400 within-(R/F × early/late) permutations
through the same scalar Ridge transfer.

## Diagnostics — the consensus is legal and complete, and it decays

| season | cutoffs | rows | coverage | sd | teacher disagreement (sd) | pair corr | corr(y) | corr(champion) |
|---|---|---|---|---|---|---|---|---|
| 2022 | 2019/20/21 | 247,472 | 1.0000 | .1648 | .0718 | .798–.918 | **.1374** | **.8579** |
| 2023 | 2020/21/22 | 245,525 | 1.0000 | .1216 | .0941 | .736–.902 | .0357 | .5477 |
| 2024 | 2021/22/23 | 253,507 | 1.0000 | .0973 | .0969 | .689–.976 | **.0497** | .5228 |

Coverage is perfect and legality holds. But the consensus **degrades on exactly
the seasons that matter**: its correlation with the target falls from .137 to
.036/.050, its own dispersion shrinks by 41%, and teacher disagreement grows to
match the consensus's entire remaining spread (.0969 against sd .0973).

## Transfer — the decisive result

| | 2022 → 2023 | 2023 → 2024 |
|---|---|---|
| source reconstructibility R² | .8126 | .7040 |
| **target frozen reconstructibility R²** | .3830 | **−.2231** |
| unique variance share (source / target) | .187 / .561 | .296 / .419 |
| raw rho (source → target) | +.00404 → **−.08179** | +.01157 → −.00041 |
| **unique rho (source → target)** | +.01466 → **−.10753** | +.01714 → **+.00718** |
| matched-null percentile | 100.00 | 99.75 |
| null p95 / p99 | .0906 / **.0915** | .0039 / **.0048** |
| segments (R / F / early / late) | −.033 / +.003 / −.106 / −.107 | +.012 / −.006 / +.010 / +.002 |

Four things kill it, and no one of them is a close call.

**1. The sign flips between the boundaries.** Fitted on 2022 the direction has
unique rho **+.0147**; frozen onto 2023 it is **−.1075**. On the next boundary
the same fit-and-freeze keeps its sign (+.0171 → +.0072). A direction that
reverses on one boundary and holds on the next has not demonstrated invariance,
which is the entire hypothesis. This is the shape that closed six axes on
2026-08-27 and both add-on audits on 2026-08-28.

**2. The two nulls are not comparable.** Null p99 is **.0915** on the first
boundary and **.0048** on the second — a factor of 23. "Above p99" therefore
means something different in each column, and the first boundary's enormous null
says the permuted directions themselves routinely reach |rho| .09 against the
champion residual, so that surface cannot discriminate at the scale the
hypothesis lives at.

**3. The second boundary's frozen reconstruction R² is negative (−.2231).** The
source-fitted Ridge is worse than the target's own mean, so on the boundary
where the sign *does* hold, the "unique" component is not a clean orthogonal
residual — it is the consensus plus the reconstruction's failure. The +.00718
cannot be read as an increment over the champion.

**4. It is largely the champion already.** corr(consensus, champion) is .858 /
.548 / .523 and source reconstructibility R² is .70–.81, so most of what the
consensus knows is already shipped.

## No rescue

The preregistration fixed `y_soft = 0.75·y + 0.25·consensus` in advance and
banned teacher-architecture, alpha, teacher-count and recency sweeps. None is
run. A stronger teacher would not repair a sign reversal, and the null-scale
problem is a property of the boundary, not of the estimator.

Data: `out/temporal_consensus_audit.json`, log
`out/temporal_consensus_audit.log`, harness
`tools/_tmp_temporal_consensus_audit.py`.

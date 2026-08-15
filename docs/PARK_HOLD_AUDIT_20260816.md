# PARK/HOLD audit — 2026-08-16 overnight

Reviewed after E-LB1 became champion and while the pre-registered GSK3 transfer
confirmation was running. This document prevents old labels from being mistaken
for an executable queue.

## Actually actionable

1. **GSK2 / GENERAL_SKILL cell-only** — active confirmation. GSK2 produced the
   strongest model-path signal: core mean +3.141, t=4.168, ensemble +3.108, but
   F=-3.126. GSK3 repeats the exact 121->123 change on fit<=2021 / val2022 /
   untouched2023 with fresh base/control. Only a full PASS licenses deployment.
2. **E-LB2 w_cell** — two no-GPU probes are built and fully audited from the
   1110.980630 champion. This is the next highest-information submission axis,
   but only the user may submit and no score is available overnight.
3. **Expected pitch-mix context ratio** — statistically PARK (+1.64, t=1.87),
   but not currently executable as a submission experiment. Its 124-feature
   models were trained via `--extra-feats data/processed/expected_mix.csv`.
   `train_gbdt2.py` joins those columns after `fpipe.fit`; neither the table nor
   a row-local reconstruction is stored in the pack, and `fpipe.transform` has
   no `extra_feats` path. A local score therefore cannot be shipped by the
   common inference contract. Before any GPU repeat, implement frozen Trackman
   context tables in the artifact, row-local transform, new-process parity and
   subset tests. Expected value remains below the +3 adoption bar, so this is
   lower priority than GSK/E-LB2.

## Superseded or already resolved — do not run

- `--loss RMSE` HOLD is superseded by corrected RMSE2 n=6 DROP.
- GENERAL_SKILL full-core PARK is superseded for the cell-only question by GSK2
  and its GSK3 confirmation.
- H1 PARK/base-only question is superseded by the pre-registered fixed-core DROP.
- P3-C HOLD is superseded by P3-C2's measured FAIL; do not change cell subsets.
- FM_MULTILABEL, RANK16 and row-filter/two-strike are closed by later measurements.
- `--te-k b:500` OPEN/CONFOUNDED is superseded by the clean n=6 DROP.

## Low-value unresolved labels — no overnight GPU

- `--refit-mult 2.0`: corrected status HOLD, old mean +0.727 with very wide CI;
  the later common-budget experiment reduced variance but gained only +0.278.
- `--te-halflife 2`: +0.164 with wide CI; every broader recency lever failed.
- `--anchor-last-pitch`: +0.88 and source validation negative.
- F-league resolution hole: real and still OPEN, but all tested routing,
  shrinkage and league-specific corrections reverse across seasons. GSK's F
  result is diagnostic evidence, not permission for an F-only coefficient.

These axes are not added together and are not rescued by more seeds or nearby
parameter sweeps. If GSK3 fails/HOLDs, the efficient overnight action is to stop
GPU work after preserving the result; it is not to spend the remaining hours on
sub-1-point PARK arms.

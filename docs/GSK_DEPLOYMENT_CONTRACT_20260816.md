# GSK deployment contract — conditional on GSK3 PASS

Written before GSK3 finishes. This does not relax its gate. If GSK3 is DROP or
HOLD, nothing in this document runs.

If and only if GSK3 returns PASS under its frozen rule, train six deployment
cell members on `DESKTOP-053T952` with the submission surface:

- tag `GSKDEP_cell`, seeds `3,4,5,6,8,13`;
- selection val2024, deployment refit <=2024;
- no `--drop-f-pre` (submission surface contract);
- corrected 12-cell depth-5 recipe plus exactly `--feat-skill`;
- 123 features, adding only `skill_hat,skill_hat_vs_std` to B1S cell;
- base family stays the immutable eight B1S base members;
- blend remains 0.45 base / 0.55 cell;
- SLOPE, legacy SHIFT, recent-middle, exact-PB and final E-LB1 shift all remain
  fixed. No coefficient is re-derived from GSK3.

Before packaging require six packs, one fit hash, one feature hash, classes
0..11, success cells [9,10,11], new-process replay, and reversal/half/single-row
drift zero. The resulting ZIP is an unsubmitted candidate only and must pass
the standard 245,789-row smoke plus strong subset audit. The user decides any
submission.


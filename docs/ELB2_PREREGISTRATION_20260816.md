# E-LB2 — leaderboard blend-weight probes

Pre-registered after E-LB1 final scored `1110.9806302398` and before building or
submitting either E-LB2 arm. Overnight work builds and audits packages only;
the user controls every submission.

## Frozen baseline and change

Baseline is `submissions/elb1_final_0816.zip`, SHA256
`e99a5772d0bd3f649a6e5cee8972cb0f3db205ce7be957a662446f358e0dbfd3`.
It keeps eight B1S base members, six B1S cell members, SLOPE 1.0416, legacy
SHIFT 0.0052, recent-middle, exact-PB and the newly verified final output shift
`+0.002515795361`.

The only changed value is `_W_CELL`:

- `elb2_w045_0816.zip`: `0.55 -> 0.45`
- `elb2_w065_0816.zip`: `0.55 -> 0.65`

No member, seed, per-family averaging, calibration, final shift, feature,
lookup or dependency changes. The weights remain row-local and predictions
must be subset-independent.

## Interpretation and gate

With score `S0` at w=.55 and official scores `S45,S65`, fit the local quadratic
at h=.10. Because logit SLOPE and clipping follow the blend, this is a local
quadratic approximation rather than E-LB1's algebraically exact shift curve.
Require concave curvature, fitted `w* in [0.40,0.70]`, and fitted gain >=+1.
Otherwise stop. If it passes, build one final weight package and re-run all
audits; do not adjust SHIFT in the same submission. A later SHIFT re-check would
be a separate one-change experiment.

## Packaging gate

Each ZIP must differ from E-LB1 final only in `script.py`, with a unique
`_W_CELL = 0.55` replacement. Require root script, POSIX members, identical 21
non-script members, finite [0,1] probabilities, 245,789-row runtime <600 sec,
and reversal/half/player/scattered/single-row drift exactly zero.


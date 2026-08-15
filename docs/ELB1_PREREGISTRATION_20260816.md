# E-LB1 — leaderboard-selected global SHIFT probe

Pre-registered 2026-08-16 before building or submitting either probe. Actual
submission is reserved for the user; the overnight task only builds and audits
the two ZIPs.

## Official-rule basis

Dacon's 2026-08-12 answer in the official Q&A explicitly allows leaderboard
scores to select or adjust a model, hyperparameter or ensemble weight, and says
that interpolation between submitted candidates is also allowed. The same
answer preserves the independent-row rule: leaderboard score may be used only
as an evaluation result for choosing a fixed setting, never to use other test
rows or the test frame's distribution to form a row's prediction.

- Q&A: https://dacon.io/competitions/official/236743/talkboard/417082
- independent-row notice:
  https://dacon.io/competitions/official/236743/talkboard/417123
- rules: https://dacon.io/competitions/official/236743/overview/rules

For E-LB1, a row receives the same prediction whether it is alone or in the
full frame. The only new value is one fixed calibration hyperparameter.

## Arms

Baseline is the immutable champion `submissions/b1s8_20260813.zip`, SHA256
`c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332`, LB
`1108.4333490288`.

At the final output, after the frozen slope/shift, recent-middle and exact-PB:

```text
p(delta) = clip(p_champion + delta, 0, 1)
delta_plus  = +0.010
delta_minus = -0.010
```

Every other byte of the package must be identical to the champion, except the
single line in `script.py` that adds the fixed delta. No model, blend weight,
middle/PB table, feature, row order or dependency changes.

## Exact quadratic

Without clipping, Brier and therefore leaderboard score are exact quadratics in
delta. With `h=.01`, baseline score `S0`, and scores `S+`, `S-`:

```text
delta_star = -h * (S+ - S-) / (2 * (S+ + S- - 2*S0))
gain_star  = (S+ - S-)^2 / (-8 * (S+ + S- - 2*S0))
```

The champion prediction range is far from 0/1 at this scale; the packaged
script still clips safely. If curvature is non-negative, `abs(delta_star)>.02`,
or a later confirmation point materially misses the parabola, stop and do not
adopt.

## Overnight integrity contract

- filenames under 30 characters; `script.py` at ZIP root; POSIX archive paths;
- all 21 non-script members byte-identical to the champion;
- public-sample script smoke and 245,789-row synthetic inference under 600 sec;
- finite probabilities in [0,1];
- strong subset/reversal/half/single-row drift exactly zero;
- manifest records ZIP SHA256, script SHA256 and delta.

No submission, score fitting, E-LB2 packaging or optimal-delta package is made
until the user supplies the two official leaderboard scores.

# P3-B — one common tree budget for every base seed

Pre-registered 2026-08-15, **before any P3-B run**. Fixed; not adjustable after
a number is read.

## Hypothesis

The base arm's validation curve is flat near its optimum, so early stopping
picks a different iteration per seed for reasons that are lottery rather than
signal. Measured on this host: between two identical runs the pick moved by up
to **+459 iterations, worth +2.62 BSS**, and across the `B1J6_base` seeds the
stopping points span **600 to 874**. If that spread is noise, replacing it with
one common budget should reduce seed variance without costing mean performance,
and may improve it.

## The budget, and why the earlier one was wrong

**Rejected**: `CTRL_base`'s median 1303.5. Those stopping points were chosen on
**val2024**, so a budget derived from them and then judged on 2024 is circular —
the budget would already have seen the season it is scored on. That is not an
honest unseen-season test and it is not used.

**Used**: `B1J6_base`, the valid judging-surface control on the same host.

| | |
|---|---|
| host | `DESKTOP-053T952` |
| surface | fit ≤2022 → val **2023** → untouched **2024** |
| seeds | 3, 4, 5, 6, 8, 13 |
| `best_iter` | 874, 782, 854, 799, 600, 807 |
| sorted | 600, 782, 799, 807, 854, 874 |
| median | (799 + 807) / 2 = 803.0 |
| integer rule | **floor** |

**Common selection budget = 803**, for every seed. It is derived only from
2023-selected stopping points and is judged once on untouched 2024.

## Single-change contract

- every candidate seed's selection budget is **803**; no per-seed budget;
- the deployment refit uses the existing `_refit_trees(803, 1.5)` unchanged;
- **forbidden**: a different median definition, any quantile, any multiplier
  sweep, any per-seed adjustment;
- cell, blend weight, features, and every post-processing constant are frozen;
- the fixed cell is the same one used for the control's core.

## Control — a same-session fresh control is REQUIRED

Item 8 of the instruction: if the code path has taken performance-relevant
changes since the control, parity must be established, and **if parity is
unclear a same-session fresh control is required**.

Parity was checked, not assumed, with `tools/training_path_diff.py` and
`tools/feature_path_parity.py` against `0169d8d`, the commit `B1J6_base` ran
under:

- `features.py`, `target_enc.py`, `roster_transition.py`, `graph_features.py`
  — **byte-identical**;
- `season_std.build_anchors` — the function whose shrinkage denominators changed
  from `n0`/`sn` to `n0c`/`snc` — **numerically identical, max |diff| 0.000e+00**
  on 61,463 real rows;
- `skill.build` — the new code is behind `neutral_mode == "const"`, which
  requires a flag this recipe does not pass;
- `season_std.add_std` end-to-end — **UNPROVEN**. It cannot be driven from
  outside `fpipe` (`season_prior` is a table built upstream, not a flag), so it
  was not verified.

One unproven link is enough. **`B1J6_base` is NOT reused as the control.** P3-B
fits **12** base models — 6 candidate at budget 803 and 6 fresh control under
`B1J6_base`'s exact command — in one session on one host. The reuse condition in
the instruction is not met, and quietly reusing it is the v16 (−6.15) / v17
(−53.6) failure mode.

## Reporting

Per seed: previous `best_iter`, its distance from 803, base BSS delta. Then base
ensemble delta; fixed-core delta; paired mean, SE, t, 95% CI from
`tools/judge.py`; early/late and R/F; RMS and correlation against the control;
reliability and resolution. And explicitly: **did the reduction in `best_iter`
variance actually produce a better test Brier**, or only a tighter spread?

## Gate

| condition | outcome |
|---|---|
| core delta ≤ 0 | **FAIL** |
| 95% upper < +3 | **DROP** |
| delta < +3 or t < 2.4 | **PARK** — no further structural search |
| delta ≥ +3 **and** t ≥ 2.4 **and** n = 6 **and** no interval worse | **report as a submission candidate and stop** |

No submission without explicit approval. If P3-B produces no candidate, P3-C
(success-block-balanced cell CE) is designed next, and its single class weight,
its analytic deweighting formula and its seed-3 gate are pre-registered before
any implementation, with weight sweeps forbidden.

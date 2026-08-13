# HANDOFF.md

Read first: [AGENTS.md](AGENTS.md) -> [EXPERIMENT.md](EXPERIMENT.md) -> this file

## Current Agent

Claude

## Next Agent

Claude

## Status

`IDLE` — no GPU job is running. `desktop-5070` is free; `desktop-4070` and
`hsu-server` are offline (the 4070 is the A100's ProxyJump, so both go together).
Last updated 2026-08-13 23:30.

Champion is **B1S8, LB 1108.4333490288, rank #34**
(`submissions/b1s8_20260813.zip`, sha256
`c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332`).
Two submissions landed today, +5.744 then +0.887, **+6.631 cumulative over v11**.
Nothing is queued.

---

## The coordinate that makes local numbers readable

**Debiased local score (slope/shift + recent-middle applied) + 139.03 = LB**,
confirmed on two points (B1S6 +139.04, B1S8 +139.03).

This holds **inside the B1S family only**. The previous family's +128.4 was
carried over once and got the *sign* of a prediction wrong. Re-measure the offset
before trusting it on any new structure or surface.

## The adoption bar (do not soften it)

```
Δ >= +3  AND  t >= 2.4  AND  n >= 6 paired seeds  AND  same machine  AND  same surface
reject when the 95% upper bound < +3
otherwise PARK -- never rescued by more seeds, sweeps, or blend search
```

More seeds have never changed a conclusion (6 -> 12 -> 18). The machine effect is
11 points, larger than seed noise, so seeds cannot buy their way past it.

---

## Closed today (2026-08-13)

| Step | Axis | Verdict |
|---|---|---|
| 1 | `LEGLBL` cell arm | completed, 6/6 seeds verified in LEDGER |
| 2 | P0S decomposition | **closed with the hypothesis reversed** — see below |
| 3 | `--te-k b:500` isolated | DROP — core +0.17, SE 0.79, t +0.22, 95% upper +2.21 |
| 4 | 2022->2023 R-only stress | skipped — it is a post-KEEP gate and no KEEP candidate exists |
| 5 | two-strike true routing | CLOSED — source +33.87 / target −35.82 on the segment |

### P0S: where the +5.744 came from is still mostly unexplained

Of B1-S's +5.744 LB gain, the measured parts are:

```
P1 (three fixes together)          -0.72   null
corrected failure-mode labels      -3.97   the BUGGY labels were better
member count 6 -> 8 base            -0.89   (of the separate +0.887 submission)
------------------------------------------
unattributed                       ~+11.3
```

The most plausible home for the remainder is the champion's own training command,
which predates LEDGER and was never recorded. **Do not decompose P1 into single
arms again** — the fixes are non-additive and three separate causal stories I
built on subtraction were all wrong.

Mechanism for the label result: the old global `shift(-1)` pulls the next
pitcher's first diff onto 6.72% of rows. That is noise in an **auxiliary**
multiclass head — the success bit is read directly and is correct either way —
and auxiliary-head noise regularises. Two of six seeds carry most of it
(+8.32, +8.75 vs +0.8…+2.4).

### One decision is open and I did not take it

`FLAG failmode-corrected-labels` scores **core +3.97, SE 1.46, t +2.71**, which
*passes* the adoption bar. LB conversion puts it at ~1112.24, about +3.8 over the
current champion. Three ways forward:

- **A** — ship the legacy labels as a challenger. Shipping a known bug on purpose.
- **B** — rebuild it as *deliberate auxiliary-label noise* (recommended). Same
  mechanism, honest implementation, survives Phase-2 code review. Costs a new
  experiment.
- **C** — hold. Two seeds carry the mean and the CI lower bound is +0.21.

---

## Next: STEP 7 — a new F-league lever

**Do not re-measure the hole.** It is measured (`tools/segment_resolution.py
--split test`, 2024): F base hon/norm 507.8 vs R 839.3 (ratio .605); F cell 429.8
vs R 861.8 (ratio .499). The models decline to discriminate on F and fall back
toward the mean, and the cell member — stronger overall — is *relatively worse*
there.

Mechanisms already failed, **including their parameter variations**:

- F-only model (data-starved)
- league-conditional blend weight (+1.484 forward, −2.119 reverse)
- Bernoulli bootstrap
- simple shrinkage
- segment drift correction

Allowed to design: league x season relative representation · league-specific
rank/percentile · shared model with a league-relative feature representation ·
cell support / sample-geometry analysis.

**Before implementing**, check `docs/SETTLED.md`, `src/fpipe.py`, `src/features.py`
and `LEDGER.tsv` that the idea is not effectively `feat-v5`, `league_runner`, or an
existing domain feature. If it is, do not run it.

Any lever must clear **both** the 2023->2024 and 2024->2023 directions.

## Research backlog (after STEP 7)

- H1 additive: keep `std_asof_pitcher_success_rate_delta` and *add* `h1_hand_delta`
  rather than replacing it 1:1 (the replacement parked at +1.28)
- H1 base-only, pre-registered
- `--refit-mult 2.0` — HOLD
- RMSE loss — HOLD
- `--te-halflife 2` — HOLD

PARKed axes, which must not be stacked with each other: H1 (+1.28), expected pitch
mix (+1.64), season anchor (+0.88).

---

## Context

| | |
|---|---|
| Champion | **B1S8 — LB 1108.4333490288**, rank #34 |
| Top of board | 1,288.18 last observed |
| Machines | `desktop-5070` online · `desktop-4070` + `hsu-server` offline |
| Ledger | 635 rows across 4 machines |
| Settled | 123 FLAG lines, read by `precheck.py` |

Closed axes with their mechanism live in `docs/SETTLED.md`; the long-form history
is in `docs/EXPERIMENTS_LOG.md`; today's shareable summary is
`docs/HANDOFF_20260813.md`. **This file is the baton, not an archive** — it grew to
860 lines of finished instructions once, which is how a stale task gets picked up
twice.

## Before starting anything

```bash
tools\agent_sync.cmd start claude          # Windows app; bare bash resolves to WSL
python tools/precheck.py <every flag>      # exit 2 means forbidden
bash tools/ledger_sync.sh                  # pull the remotes' LEDGER rows before judging
```

`ledger_sync` is not optional at session end: the trainer writes its row on the
machine that ran it, so a run finished on the 5070 is invisible here until it is
pulled. Twelve `TS2` rows and three `LEGLBL_cell` rows were missing from the local
ledger tonight for exactly that reason.

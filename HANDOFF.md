# HANDOFF.md

Read first: [AGENTS.md](AGENTS.md) -> [EXPERIMENT.md](EXPERIMENT.md) -> this file

## Current Agent

Claude

## Next Agent

Claude

## Status

`IDLE` — no GPU job running. `desktop-5070` free (4% util, desktop processes only);
`desktop-4070` and `hsu-server` offline. Last updated 2026-08-15 01:21.
`.deployed_commit` on the 5070 is `b4024c41`; `src/` is identical to it at HEAD.

Champion is **B1S8, LB 1108.4333490288, rank #34** (`submissions/b1s8_20260813.zip`,
sha256 `c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332`).
Unchanged — nothing has cleared the bar since. Ledger 691 rows, SETTLED 136 FLAG lines.

---

## What today closed

| Axis | Fresh-control verdict |
|---|---|
| `--te-k b:500` | DROP — core +0.17, 95% upper +2.21 |
| two-strike routing | CLOSED — source +33.87, next season −35.82 |
| `NOISE12` (3.873% row-random corruption) | DROP — core −1.725, 95% upper +0.96 |
| `--max-ctr-complexity 3` | DROP — core −0.277, 95% CI [−1.36, +0.81] |
| `GENERAL_SKILL_ADD` | **PARK** — core +1.104, 95% CI [−1.69, +3.90] |
| F league-relative representation | DUPLICATE, not run — see below |
| `H1_ADDITIVE` (`--feat-h1 --h1-additive`) | DROP — core −0.304, 95% CI [−3.25, +2.64] |

## The three things that will change how you read numbers

**1. Fresh controls flip signs.** A candidate measured against the champion's own
stored artifacts is not the same measurement as one against a control trained in
the same session. CTR3 base was **+0.540 historical / −0.187 fresh**; the CTR3 core
was **+0.289 historical / −0.277 fresh**. Both happened to land on the same verdict;
next time they may not. Use the fresh comparison as primary.

Reuse an existing control only when host, surface, seeds, fingerprint, features,
effective params and the relevant code path are all unchanged. Today's controls
are `CTRL_base` and `NULLC_cell`, both on DESKTOP-053T952, fingerprint
`0.5352282202778269`.

**2. Re-run variation is the early-stopping pick, not GPU noise.** `NULLC_cell`
re-ran `B1S_cell`'s exact command and reproduced it: per-seed core
`[+0.50, 0.00, +0.01, 0.00, −0.00, −0.00]`, with corr(|Δbest_iter|, |Δval|) = **0.9906**
and every seed at Δiter 0 landing on 0.000 exactly. The base arm is the opposite
case — it stops at 950–1800 where the eval curve is flat, and between two identical
runs its pick moved by up to **+459 iterations**, worth +2.62. So `base +0.57, SE 0.44`
is *not* a per-seed noise floor to discount against. Print `best_iter` in every
paired report; where it did not move, the delta is mechanism.

**3. A flag reaching one arm does not mean it reached the other.** The cell arm
builds its own `CatBoostClassifier`. `--max-ctr-complexity 3` never reached it: the
packaged model said 4 and one seed reproduced its own control to the cent, exit code
0 throughout. Fixed (`_cell_params`), and `_snap()` now records `get_all_params()`
for all four constructors into the pack. Scope audited: 158 prior cell runs, none
affected, **no old verdict reopened**.

## H1 is finished, both ways

The parked replacement arm bundled two changes: it added `h1_hand_delta` and
dropped `std_asof_pitcher_success_rate_delta`, and dropping delta families is
separately worth −4.99. `--h1-additive` keeps the column, so only the composed
prior moves — 121 → 122 features, nothing removed.

```
primary fresh-vs-fresh core  -0.304  SE 1.145  t -0.27  CI [-3.25, +2.64]
base  +2.161  SE 2.346  median +3.761  5/6 positive
cell  -1.235  SE 0.900  2/6 positive
```

Removing the confound did not rescue it. Base is still the only positive family
and still imprecise — its SE here is **the largest measured**, because the base
arm's stopping point moved by 255.8 iterations on average and 712 on seed 5,
which alone cost −8.42. **Base-only is not reopened by this**: choosing a family
after seeing the split is what pre-registration exists to prevent, and this base
estimate is looser than the one that already parked.

## Open signals, none scheduled

- **Legacy failure-mode labels**: base-fixed core **+4.15, SE 1.11, t 3.76,
  6/6 positive**. Statistically KEEP-grade, mechanism unisolated — `clean14` cannot
  be defined (the two legacy-only cells are 100% corruption-populated) and NOISE12
  showed row-random corruption does not reproduce it. **RESEARCH_ONLY, not scheduled.**
  No boundary-targeted rescue, no rate sweep, no legacy-mimic variants.
- **`GENERAL_SKILL_ADD` cell family**: +2.568, t 2.18, 6/6 positive while base is
  −0.077. A family sign split, the same shape as H1 with the signs reversed. A
  cell-only arm is a **new pre-registration**, not this experiment's verdict.
- **`--fm-multilabel`**: PROVENANCE INCOMPLETE / mechanism technically open / LOW.
  No same-era baseline exists, and the path calls `_pitch_labels` without a
  `fit_mask`. Recorded as an INFRA BUG that is deliberately not scheduled.

## PARK axes — never add them together

```
H1 replacement  +1.28    expected pitch mix  +1.64
anchor-last-pitch +0.88  GENERAL_SKILL_ADD   +1.10
```
(H1 *additive* is DROP, not PARK — it does not join this list.)
"1+1+1 is +3" reasoning is banned. No seed extension to rescue a PARK.

## Why F is not the next GPU job

`x − mean(game_type, season)` is arithmetically `--feat-v5`'s column plus a
per-(league, season) step, and that step averages **0.0781 sd on R rows** — 89.09%
of the frame — against 0.6368 sd on F. `--feat-v5` is CLOSED at −26. Rank,
percentile, robust-z and transport share the same eight columns and the same intent,
so they inherit the objection, and choosing among them is the sweep the plan bans.

Design input if the axis is ever reopened with a genuinely different mechanism: the
league gap is column-specific and does not survive the regime break uniformly. Over
2023–2024, `asof_pitcher_ball_rate` sits **+0.978 sd** and `asof_pitcher_middle_rate`
**−0.851 sd**, both stable across all six seasons, while the pitcher success-rate
family has **converged to ~0** (+1.282 in 2022 → +0.451 → −0.002) and batter success
is collapsing. The leagues differ in *how pitchers miss*, not in *how good they are*.

## skill-hand: open in provenance, overlapping in mechanism

`skill.py` implements `axis="hand"` and it has **never run** — `--skill-axes` 0
runs, `skill_hand` 0, `axis=hand` 0. But its three extra inputs
(`te_pitcher_batter_hand_ratio` / `_dev` / `_n`) are all already champion
features, `_dev` is the exact input H1 composes, and the axis's stated
motivation — "+31 in the cross-fit residual" — is the same +31 that
`FLAG pitcher-batter-hand-residual | CLOSED` rejected: year-over-year
correlation **+0.0108**, transfer **−191.952**. With H1 now DROP on the same
input family, a richer supervised fit on those inputs has no independent
rationale. Do not promote it without one.

## Next candidates (neither started)

1. **season-transfer attribution map** — CPU. Which of the 121 features survive
   into the next season, by family. Not yet designed; this is the queue's head.
2. **FM_MULTILABEL_V2** — needs the implementation repaired first (partition-safe
   label recovery, no `nan_to_num(...,0)` on unknown auxiliary labels), which is
   a rewrite, not a re-run.

HOLD, low expected value: `--refit-mult 2.0`, `--loss RMSE`, `--te-halflife 2`.

## Adoption rule (unchanged)

```
KEEP  delta >= +3 AND t >= 2.4 AND n >= 6 paired AND same machine AND same surface
DROP  95% upper < +3
else  PARK          KEEP only ever triggers temporal stress
```

Local→LB for the B1S family: **debiased score + 139.03** (confirmed on two points).
Do not carry that offset to another structure — the previous family's +128.4 got a
prediction's sign wrong.

## Before starting anything

```bash
bash tools/ledger_sync.sh                  # the trainer writes its row on the runner
python tools/precheck.py <every flag>      # exit 2 means forbidden
```

Judging tools: `tools/arm_compare.py` (paired, prints best_iter),
`tools/cell_arm_delta.py` (holds a base family fixed),
`tools/orphan_audit.py` (lead generator only — its raw deltas pair baselines
naively and put distillation at +241).

# HANDOFF.md

Read first: [AGENTS.md](AGENTS.md) -> [EXPERIMENT.md](EXPERIMENT.md) -> this file

## Current Agent

Claude

## Next Agent

Claude

## Status

`IDLE` — no GPU job running, and none was run today. `desktop-5070` free;
`desktop-4070` and `hsu-server` offline. Last updated 2026-08-15 03:40.
`.deployed_commit` on the 5070 is `b4024c41`; `src/` is identical to it at HEAD.

Champion is **B1S8, LB 1108.4333490288, rank #34** (`submissions/b1s8_20260813.zip`,
sha256 `c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332`).
Unchanged — nothing has cleared the bar since. Ledger 691 rows (the season-transfer
map trains nothing, so it adds none), SETTLED 141 FLAG lines.

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

## skill-hand: measured, and it goes the wrong way

`MATCHUP` holding 6/6 across boundaries would, read carelessly, reopen the hand
axis. It splits into the platoon (`pitcher_hand`, `batter_hand`,
`matchup_same_hand`, `dom_same_*`) and the pitcher's own record by handedness
(`te_pitcher_batter_hand_*`) — **both closed**, so both are information.

```
HAND_TE / HAND_RAW   in-sample 1.039 (.779-1.236)   unseen 0.646 (.576-.761)
retention  2021->22  RAW .320  TE .279
           2022->23  RAW .400  TE .366
           2023->24  RAW .431-.493   TE .215-.236
```

The pitcher-specific hand record is worth about as much as the platoon
in-sample and about two thirds as much out of sample, on every boundary, worst
at the most recent. **MATCHUP's stability is the platoon.** Combined with the
paragraph below, the axis is now **DO NOT BUILD**, not merely deprioritised.

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

## The season-transfer map is done — read it before proposing a feature

Full report: [docs/SEASON_TRANSFER_MAP_20260815.md](docs/SEASON_TRANSFER_MAP_20260815.md).
It cost no GPU and no training: the champion refits on fit+val and so has **no
unseen season**, but `MV21_*`, `MVB22_native`/`MVCELL22_s42` and
`MVN3_s3,4,5`/`MVCELL_s42` were run with `--test-season`, which splits the test
season out before the refit. Same host, feature lists identical to B1S8's,
every reconstruction pinned at `replay_max_abs_diff = 0`.

**Three things it establishes.**

1. **Permuting a family measures routing, not information.** The numeric frame
   is **rank 90 of 112**: `std = asof + delta`, `dev = level / te_pitcher_ratio`,
   `shr = (rate·n+p·k)/(n+k)` at k=200 — all exact. `SEASON_STD` leaks on
   **13/13** columns, so its first place in every family table says nothing about
   information. Use the six closed blocks for information questions; they test
   CLOSED. And no deletion follows from a zero: `STD_DELTA` reads ~0 next season
   while deleting it actually costs **−16.22**.
2. **What survives a boundary is the pitcher and the platoon.**
   `PITCHER_HISTORY` STABLE 6/6 with its share *rising* out of sample (50–65% →
   63–74%), `MATCHUP` STABLE 6/6. Against that `BATTER_HISTORY` 0.21–0.43,
   `GAME_STATE` 0.03–0.50, `RECENT` median 0.31, `CALENDAR` median 0.00.
   **Do not build new derived features on the batter, game-state or calendar
   axes** — that is the reason `--feat-v4` (−18.72), `--feat-count-cat` (−9.57)
   and `--feat-count` (−1.07) all failed.
3. **Base and cell use the same information.** Agreement is spearman **+0.678**
   by representation family but **+0.943** by information block. A family sign
   split is a candidate competing with a shape the other arm already uses. This
   is the standing explanation for base-only arms looking good and failing at
   the blend — it is not a licence to pick an arm after seeing the split.

**The scope limit is now closed.** The surrogate's cell half was suspect because
the MV taxonomy has 14 cells against B1S8's 12. `B1SMOKE_base`/`B1SMOKE_cell`
(2026-08-13, the 5070) turned out to be the B1S8 recipe on exactly the judging
surface at seed 3 — **12 classes, 3 success cells, feature list identical,
replay 0** — so the calibration cost no GPU. On the champion's own taxonomy:

```
block            base same->next (ret)   cell same->next (ret)   gap
PITCHER_HISTORY  48.92 -> 66.16  1.35    55.14 -> 64.15  1.16   -2.01
MATCHUP          15.36 -> 17.54  1.14    14.21 -> 15.88  1.12   -1.66
CALENDAR_ID       7.61 ->  5.30  0.70     7.68 ->  5.36  0.70   +0.06
COUNT             7.37 ->  4.93  0.67     8.23 ->  8.52  1.04   +3.59
BATTER_HISTORY   12.30 ->  4.59  0.37     8.98 ->  3.88  0.43   -0.71
GAME_STATE        8.42 ->  1.48  0.18     5.76 ->  2.21  0.38   +0.73
base-cell spearman +0.943   max gap 3.59pp   (pre-registered: >=.85 and <=10pp)
```

Same information, different routing — **confirmed on the champion itself**, not
just on a surrogate. The proxy's cell advantage on COUNT and GAME_STATE
reproduces but smaller (+0.37 and +0.21 against +0.47 and +0.36), and every
other block reproduces within ±0.02. So the MV surrogate was a good stand-in for
*transfer* all along; its poor PredictionValuesChange agreement measured
routing, which is exactly what the two taxonomies disagree about.

## The supervision-geometry night (2026-08-15) — both tracks closed

The attribution map said base and cell route the same information differently,
so the open question was the objective, not another feature family. The Murphy
decomposition agreed: on the judging surface a perfect recalibration of the core
is worth **+13.3 BSS** (reliability is 0.013% of uncertainty) while +0.001 of
absolute resolution is worth **+400**. Two supervision geometries were tried.

**Ranking (PairLogitPairwise, group16) — BLOCKED, not judged.** The fitted model
cannot be scored. Loading the saved `.cbm` and predicting **100 rows on the
laptop's CPU** segfaults. Not OOM (22 GB free), not this machine (two hosts),
not `task_type=GPU` alone (40k x 60 trees is fine), not the categorical set
(121 features and 9 categoricals at 60k x 30 trees is fine) — it is scale: at
300 trees `predict` hangs, at 1224 it segfaults. The old BANNED note blamed the
refit; the refit was only the first thing that ever touched the model. CPU
training is the sole remaining route at a measured **9.7 h/seed**, so ~58 h for
the 6-seed judgement. No number was produced and no verdict is claimed on the
hypothesis. The four-process infrastructure (`--rank-stage 1/15/2/25`), the
handoff frame-equivalence guard and `tests/test_rank_contract.py` survive.

**FM_MULTILABEL_V2 — FAIL at the single-seed gate.** Both label defects repaired
first: partition-safe recovery (113 rows) and complete-case training (99.849%
kept, 1,691 dropped) instead of `nan_to_num(..., 0)` asserting "did not happen"
for "unknown". Artifact sound, fingerprint identical to the controls. **ML_CORE
876.73 against CONTROL_CORE 888.81, delta −12.07** — the pre-registered FAIL
condition. Mechanism: the multilabel member's resolution is **0.002173, the base
arm's 0.002174 and below the cell arm's 0.002207**, so swapping it into the cell
slot costs exactly the resolution the cell geometry was contributing.

**Two packaging traps found, both fixed.** A CatBoost object drops custom
attributes through joblib, so `_rank_calib` came back MISSING from every saved
rank pkl and inference would have shipped a clipped raw PairLogit score as
P(success) — mean 0.2279 against a calibrated 0.5417. `_fm_multilabel` had the
same exposure: without it a reloaded multilabel model looks binary and column 1,
P(middle), is read as P(success). Both now travel in the pack dict, and the rank
path **raises** rather than degrading when the calibration is absent.

**`schtasks` on the 5070 is not broken** — it was registered `Interactive only`
and the console session belongs to a different account, so it waited forever
(`267011` = never started). `/ru SYSTEM` fires. See `tools/run5070.sh`.

## Next candidates

1. **FM_MULTILABEL_V2 — closed by measurement, not HOLD any more.** The calibration fired the branch that
   deprioritises it: family-specific feature admission is not supported, and it
   may only be revisited as *the same information under a different supervision
   geometry*, never as a family-specialisation play. It still needs the
   implementation repaired first (partition-safe label recovery, no
   `nan_to_num(...,0)` on unknown auxiliary labels), which is a rewrite.
2. **next-season latent skill — dead, do not build.** The obvious follow-up to
   `PITCHER_HISTORY` rising out of sample was that `skill.py` aims at the rest of
   the *current* season. Retargeting it at the next season is genuinely unbuilt
   and leakage-clean, but it is a duplicate (pearson +0.926 with the current
   estimator, R² 0.990 on the frame) and it is **worse on its own ground**: on
   the same 205,033 rows of 2024, rest-of-season target explains 63.7%, plain
   k=80 shrinkage 61.4%, next-season target **59.5%**. The right horizon costs a
   season of history and 20% of pitchers, and that costs more than it buys.

No GPU candidate is licensed. The GPU has been idle since 2026-08-14.

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

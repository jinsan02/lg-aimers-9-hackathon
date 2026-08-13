# EXP-000 — repository / experiment-history audit

Step 1 of the 2026-08-13 improvement plan: before writing any code, establish
which of the five proposed axes already exist. The plan's own rule is that a
duplicate is reported, not implemented.

Searched: `tm-feats`, `pitch_type_group`, `fastball_rate`, `breaking_rate`,
`offspeed_rate`, `rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`,
`H1`, `hand_dev`, `pitcher_hand`, `batter_hand`, `recent_middle`,
`prev5_game_middle`, `SLOPE`, `SHIFT`, `failmode`, `ZD5` — across `src/`,
`tools/`, `docs/SETTLED.md`, `LEDGER.tsv`, and the shipped pkl feature lists.

---

## H1 — the document exists; the feature does not

**Correction to the first version of this audit.** `H1_HAND_MATCHUP_METHOD.md`
was supplied on 2026-08-13 (it lives outside the repo, in the user's KakaoTalk
downloads). The first version of this file said the plan's hand-matchup
exclusion was "sound on the evidence" because `--te-dev` already computes the
ratio H1 describes. That was wrong. It computes H1's *ingredient*, not H1.

What H1 specifies:

```
hand_dev            = pitcher_hand_ratio / pitcher_ratio       <- exists
prior               = league mean success of season S-1        <- exists
personalized_prior  = clip(prior * hand_dev, 0, 1)             <- does not
H1_delta            = (personalized_prior - prior)
                      * 80 / (std_pitcher_n + 80)              <- does not
```

and it **replaces** `std_asof_pitcher_success_rate_delta` 1:1 rather than being
added to the 121.

Searched `src/` and `tools/` for `hand_dev`, `ph_hier`, `H1_delta`,
`personalized_prior`, `hier_delta`: **zero hits.** `docs/SETTLED.md` has no H1
line. The ledger's `H1B22` is an unrelated plain baseline on the 2022 surface
(`--te p,pc,ph,b,pi --te-dev ... --drop-f-pre 2022`), and `TH1_hl2` is
te-halflife. **H1 has never been built and never been run.**

Every ingredient is in the champion's 121, verified in the shipped pkl:

| column | present |
|---|---|
| `te_pitcher_batter_hand_ratio` | yes |
| `te_pitcher_batter_hand_ratio_dev` (= `hand_dev`, same k=50 form) | yes |
| `te_pitcher_ratio` | yes |
| `std_pitcher_n` | yes |
| `std_asof_pitcher_success_rate_delta` (the column H1 replaces) | yes |
| `std_season_prior` in the artifact | yes |

So CatBoost holds all three inputs and would have to reconstruct
`(clip(prior·hand_dev,0,1) − prior)·80/(n+80)` from splits. That is the exact
argument `src/skill.py` was built on and measured: a smooth shrinkage weighting
is a weighted average, trees approximate it in steps, and a learned linear
combination explained 59.0% of the pitcher-skill target against a GBDT's 46.5%
on the same inputs. H1 is the same shape of quantity.

**Consequence for the plan.** §2.3 excludes the hand-matchup family on the
grounds that H1 is already implemented. It is not, so the premise is false. The
generic members of that family stay excluded on their own merits — `same_hand`,
`R_R/R_L/...`, and plain `pitcher × batter_hand` interactions really are
covered by the `ph` TE axis. But **H1 itself is an untested candidate**, and by
the plan's own duplication test it is the one item in the hand family that is
not a duplicate. It is not in the plan's five experiments because the plan
believed it was done.

---

## EXP-A — Legacy calibration re-derivation

**Duplicate:** no. **SETTLED:** not closed. **Proceed.**

`SLOPE = 1.0416`, `SHIFT = 0.0052` live inline in `src/script_blend_v11.py` and
are already flagged as weakly-evidenced in `CHAMPION_v11_RECIPE.md` part 6.
Nothing has re-derived them on new OOF.

⚠ **The plan's acceptance procedure is in-sample.** "OOF Brier 최소화 기준으로
최적화" then "OOF Brier before/after" fits and scores `a, b` on the same rows,
which cannot lose. Measured 2026-08-13 on B1-J 6-seed OOF, the difference this
makes is not small:

```
core, unseen 2024                             894.93
+ isotonic fitted on 2024 itself              929.48   (+34.56)
+ isotonic fitted on val2023, frozen          795.78   (-99.14)
    same, after removing the mean bias        814.89   (-81.93)
```

A flexible monotone map has +34.56 of apparent headroom and loses 82 points
when it has to transfer one season -- and that is *after* debiasing, so the
shape itself is season-specific, not just the base rate. A two-parameter map is
far more constrained and may well transfer where isotonic does not, but the
experiment has to be fit-on-source / freeze / apply-to-target to show it. This
is also `docs/SETTLED.md` rule 9 and the `measure-what-you-ship` rule: the
constants must come from the OOF of the model that actually ships, which is
`B1S`, not the judging surface.

## EXP-B — Corrected-label ZD5

**Duplicate: yes, already done. Report instead of re-running.**

`src/failmode.py` was fixed on 2026-08-13 (pitcher-grouped shift, per-partition
label recovery, fit-frozen taxonomy) with `tests/test_failmode.py` covering it.
**Every cell arm run since is already a corrected-label ZD5**:

| run | surface | seeds | cell |
|---|---|---|---|
| `B0JL_cell` | val2023 → unseen 2024 | 3 | 882.57 |
| `B1J6_cell` | same, with `--p1` | 6 | 886.52 (ens. 889.78) |
| `B1S_cell` | submission shape | 6 | 916.42 (ens.) |

What is *not* done is a clean legacy-vs-corrected A/B: the corrected runs also
carry the P0 season cutoff and the P1 contract fixes, so their delta against
the shipped `cat_ZD5` is not attributable to the labels. If that attribution is
wanted it needs a legacy-label arm on the current code, which is one flag away
but has to be built. The plan's own instruction ("기존에 동일 실험이 있다면 새
코드 대신 결과를 보고한다") is why this is reported rather than re-run.

Also noted: the plan's "iteration 5000은 이미 닫힌 축" is right
(`failmode-cell-iters-5000`, −8.67), and every corrected cell run still pins at
the 3000 cap (best_iter 2951–2999), exactly as the shipped ZD5 did.

## EXP-C — Recent-middle re-verification

**Duplicate:** partial. **Proceed with C-1/C-2.**

The 8 offsets and the NaN offset are documented in `CHAMPION_v11_RECIPE.md`
part 4 and generated by `tools/make_matchup_constants.py::middle_adjustment`
(same 8 quantile bins, `MID_K = 500` shrinkage — so the plan's C-2 shrinkage
already exists in code and only needs re-fitting on new OOF).
`tools/analyze_v12_failure.py` studied the axis when v12 changed it to
career-middle and lost LB −18.271. What has never been done is C-1: recompute
bin residuals from OOF and check the legacy hard-coded numbers reproduce.

Two things to carry in: the legacy lookup is **non-monotone** (bin 1 +0.00913
→ bin 2 +0.00146, sign flip at 5→6), and the generator's row join was fixed on
2026-08-13 — it previously matched predictions to train rows **by position**.
Any comparison against the legacy numbers is a comparison against constants
produced by that join.

## EXP-D — Trackman expected pitch mix

**Duplicate: substantially. Read before writing.**

Both halves of the proposed construction already exist:

| plan step | existing code |
|---|---|
| §8.1 league context prior, no player linkage | `src/tm_context.py` — keys `balls/strikes/outs/inning/top_bottom/pitcher_hand/batter_hand`, `MIN_N = 200` fallback to a coarser key |
| §9 personal pitch mix | `asof_pitcher_{fastball,breaking,offspeed}_rate`, **already in the champion's 121** |
| P(group \| pitcher, count) from Trackman | `src/build_tm_pitchmix.py` (E114), per-season expanding, `tmx_*` |

And the champion does not carry three pitch-mix features but **twelve**:

```
asof_pitcher_{fastball,breaking,offspeed}_rate
  + _shr        (shrunk)
std_asof_pitcher_{fastball,breaking,offspeed}_rate
  + _delta      (within-season change)
```

Dropping the pitch-mix delta family costs −4.99 (`local-targeted-pruning`), so
this family is live and load-bearing.

The genuinely new element is therefore narrow: **the context-ratio multiplier**
`P(g|context)/P(g)` applied to a personal rate the model already has in four
forms. That is a smaller marginal claim than the plan assumes, and it sits
between two closed neighbours — `--tm-feats` CLOSED at −6.04 (t −2.14) and
`predicted-pitch-probability-features` CLOSED. The plan is right that the
closed `--tm-feats` path was pitcher×season summaries and is structurally
different from this; it is not right that the personal pitch mix is unexploited.

Verdict: **not a duplicate, but re-scope.** The §18 correlation check is the
load-bearing test here, against all twelve existing pitch-mix columns, not
against three.

## EXP-E — Expected pitch physics

Conditional on D by the plan's own rule; D is re-scoped, so E is deferred.
Note `src/build_tm_command.py`, `src/build_mechanics_history.py` and
`src/build_tm_consistency.py` already read `rel_speed`, `spin_rate`,
`extension` and per-pitch release spread — read those before building
`expected_rel_speed`.

---

## Order this changes

The plan's sequence stands with two edits:

1. **A before everything, and on `B1S` OOF, fit-on-source/freeze/apply.** Not
   in-sample.
2. **B is a report, not a run.** The GPU time it would have taken goes to the
   legacy-label attribution arm only if that attribution is actually wanted.

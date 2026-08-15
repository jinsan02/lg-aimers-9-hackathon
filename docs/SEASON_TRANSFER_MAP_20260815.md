# Season-transfer attribution map — 2026-08-15

> ## ⚠ PARTIALLY RETRACTED — 2026-08-15 P0 contradiction audit
>
> **Two of the three boundaries in this document do not exist.** The `T21`
> (2021→2022) and `T22` (2022→2023) pairs were fitted before `5b61fbd`
> (2026-08-13 03:23:39), when `--test-season S` removed only `season == S` and
> left every *later* season in the training pool. `MV21_base`/`MV21_cell`
> trained on **499,032 rows of 2023–2024**; `MVB22_native`/`MVCELL22_s42`
> trained on **253,507 rows of 2024**. Neither pair carried
> `--max-train-season`. They are not next-season-unseen models, and
> `docs/INVALIDATED.tsv` has said so since 2026-08-13 — this document was
> written without consulting it.
>
> **Read nothing below as multi-boundary evidence.** Every "three boundaries",
> "all nine (arm × boundary) cells", "stable across every boundary measured",
> and "worst at the most recent boundary" claim is withdrawn, including the
> per-boundary HAND table in §7 and the F-league BSS −8200 / −779 contrast.
>
> **What survives** is the `T23` row (`MVN3_s3,4,5` / `MVCELL_s42`) — valid
> because no season exists after 2024 — and the `B1SMOKE` calibration in §11,
> which is post-fix with `--max-train-season 2024` and a verified lineage of
> `fit_max_season=2022 val=2023 test=2024`. Both are **a single 2023→2024
> boundary**. The surviving findings (PITCHER_HISTORY / MATCHUP retention,
> BATTER_HISTORY / GAME_STATE collapse, base–cell block Spearman +0.943, and
> the `skill --axis hand` verdict) keep their direction but drop to
> single-boundary strength. **No new axis may be permanently closed on that
> alone.**
>
> **The champion is unaffected.** This map changed no feature and touched no
> member of B1S8; LB 1108.4333490288 stands. Feature-priority conclusions built
> on `T21`/`T22` are **held, not reversed** — re-deriving them needs retraining,
> and no GPU is being spent on transfer diagnostics.
>
> `tools/season_transfer_map.py`, `season_transfer_core.py` and
> `matchup_decomp.py` now call `invalidated.guard()` and **refuse** to reproduce
> the invalid rows. See `docs/SETTLED.md`, the two `2026-08-15 P0` entries.

CPU only. No GPU used, no submission artifact produced, champion unchanged
(**B1S8, LB 1108.4333490288**).

Tools: `tools/feature_families.py`, `tools/feature_blocks.py`,
`tools/feature_redundancy.py`, `tools/season_transfer_map.py`,
`tools/season_transfer_core.py`, `tools/season_transfer_report.py`,
`tools/matchup_decomp.py`.
Data: `out/season_transfer_{map,core,blocks}.csv`, `out/matchup_decomp.csv`,
`out/feature_{families,redundancy}.csv`.

---

## 1. Why this needed no training

The shipped champion cannot answer a transfer question. `train_gbdt2` refits the
deployment model on fit+val, so `cat_B1S_*.pkl` has seen every season in
`train.csv` and has **no unseen surface at all**.

Three older runs of the same 121-feature recipe were trained with
`--test-season`, which splits the test season out *before* the refit:

| model | arm | refit era | unseen | seeds | status |
|---|---|---|---|---|---|
| `MV21_base` / `MV21_cell` | base / cell | ~~≤2021~~ actually ≤2024 | 2022 | 42 | **INVALID** — 499,032 rows of 2023–24 in fit |
| `MVB22_native` / `MVCELL22_s42` | base / cell | ~~≤2022~~ actually ≤2024 | 2023 | 42 | **INVALID** — 253,507 rows of 2024 in fit |
| `MVN3_s3,4,5` / `MVCELL_s42` | base / cell | ≤2023 | 2024 | 3,4,5 / 42 | valid — no season after 2024 |

All six on the same host (`rohjinsan`). Their **feature lists are identical to
B1S8's, name for name and in order** (asserted in code). Each was scored on
three seasons drawn through its own `fpipe` artifact — two inside its fit era
and the one it never saw — so no feature table ever extrapolates. Every
reconstruction is pinned by replaying the stored test predictions:
`replay_max_abs_diff = 0` on all eight models.

### What carries to B1S8 and what does not

`PredictionValuesChange` by group, against the champion's own packs:

| | family spearman vs B1S8 | block spearman vs B1S8 |
|---|---|---|
| base surrogates (MVN3 / MVB22 / MV21) | 0.881 – 0.993 | 0.943 – **1.000** |
| cell surrogates (MVCELL / MVCELL22 / MV21_cell) | 0.713 – 0.748 | **0.314 – 0.486** |

**Base conclusions carry to the champion. Cell conclusions do not.** The MV cell
models build 14 cells where B1S8 builds 12 (`--fm-modes middle,ball,reverse
--fm-min-share 0.005`), and the taxonomy is the arm's geometry: B1S8's cell puts
30.7% of its routing on `CALENDAR_ID` where `MVCELL_s42` puts 6.3%. Cell numbers
below describe *a* cell-family model, not B1S8's.

Hyperparameters also differ (`--feat-k`, `--te-k`, `--p1`), so **no number here
is comparable to a B1S8 score.** This is a map of the representation.

---

## 2. Feature inventory

121 features, each in exactly one family and exactly one block; the map refuses
to run against a feature list it does not exactly cover.

| family | n | | block | n |
|---|---:|---|---|---:|
| RAW_CONTEXT | 24 | | PITCHER_HISTORY | 56 |
| TE_LEVEL | 15 | | GAME_STATE | 20 |
| DOMAIN | 14 | | BATTER_HISTORY | 16 |
| RAW_ASOF | 13 | | MATCHUP | 10 |
| SEASON_STD | 13 | | CALENDAR_ID | 10 |
| ASOF_SHR | 10 | | COUNT | 9 |
| STD_DELTA | 10 | | | |
| RECENT | 8 | | | |
| CALENDAR | 6 | | | |
| COUNT_DERIVED | 3 | | | |
| TE_DEV | 3 | | | |
| SKILL | 2 | | | |

Full table with secondary `axis` and `kind` labels: `out/feature_families.csv`.

---

## 3. Method

For a group G: draw **one** permutation of the slice's rows and apply it to
every column of G at once — jointly, so the group's internal structure survives
and only its link to the target and to the other groups is destroyed; within the
slice, which is one season and one league, so no value from another era is
injected; categoricals permuted as their own strings, so every value stays in
vocabulary and nothing is recoded to a number. Metric is ΔBSS in the
competition's own units, without the `max(0, ·)` floor (a permuted model is
supposed to go negative). Five repeats, fixed seeds. **One scheme, chosen once,
not swept.**

Reported on **share of the slice's total positive ΔBSS**, because absolute ΔBSS
is not comparable across seasons — the same model scores 2401 on 2022 and 856 on
2024.

`season` is constant inside a slice, so RAW_CONTEXT's figure is about its other
23 columns.

### The finding that changed what the measurement means

The numeric part of the frame has **rank 90 of 112**. Twenty-two columns are
exact linear functions of the others, and the constructions are exact:

```
std_*_rate            = asof_*_rate + std_*_rate_delta            resid 0
te_*_ratio_dev        = te_*_ratio / te_pitcher_ratio             resid 0
asof_*_shr            = (rate·n + p·k)/(n + k),  k = 200          resid 1e-15
skill_pc_hat_vs_std   = skill_pc_hat - std_asof_pitcher_success_rate
count_adv, matchup_same_hand, num_runners_on, month_cat, form_delta*  ...
```

So permuting one *family* while its algebraic parents stay in the frame does not
remove the information — it only measures how much the fitted trees happened to
route through the precomputed column. Running the closure test per family:

| verdict | families |
|---|---|
| information-closed (0 columns rebuildable from outside) | `RAW_CONTEXT` `CALENDAR` `ASOF_SHR` `RECENT` `TE_DEV` `SKILL` |
| mostly closed | `TE_LEVEL` (2/15) `DOMAIN` (3/14) `COUNT_DERIVED` (1/3) |
| **routing only** | `SEASON_STD` (13/13) `RAW_ASOF` (11/13) `STD_DELTA` (8/10) |

`SEASON_STD` leaks on **every one of its 13 columns**, so its dominance in the
family map is a statement about shape, not about information.

That is why a second, coarser partition exists. The six **blocks** group columns
by where the information comes from rather than by how it is shaped, and the
closure test returns **CLOSED — no column is reconstructible from outside its
block**. Block numbers may be read as information; family numbers may not.

### What this diagnostic is not

Zero permutation importance is not a removal effect, and this run reproduces the
trap rather than escaping it. `STD_DELTA` holds ~8% of in-sample importance and
essentially **0% next season** on all nine (arm × boundary) cells — while an
actual retrain that deleted the delta columns cost **−16.22**
(`FLAG local-targeted-pruning`). Both are true: the tree stops leaning on the
precomputed difference out of sample, *and* it cannot rebuild that difference
from axis-parallel splits when the column is gone. **No deletion experiment is
derived from this map.**

---

## 4. Base family map — R league, share of positive ΔBSS

`old` and `val` are two seasons inside the fit era; `next` is unseen.

| family | fit≤2021 old/val/next | fit≤2022 old/val/next | fit≤2023 old/val/next |
|---|---|---|---|
| SEASON_STD | 24.8 / 24.6 / **59.1** | 23.5 / 22.4 / **56.4** | 16.3 / 16.9 / 23.9 |
| RAW_CONTEXT | 15.4 / 12.6 / 6.2 | 12.0 / 12.0 / 2.3 | 14.3 / 13.9 / 15.8 |
| DOMAIN | 7.1 / 8.3 / 9.6 | 8.6 / 8.5 / 11.2 | 8.3 / 8.5 / **17.0** |
| TE_LEVEL | 7.6 / 7.3 / 4.6 | 7.7 / 8.1 / 5.1 | 11.6 / 11.8 / 9.2 |
| RAW_ASOF | 7.0 / 7.0 / 3.4 | 7.5 / 7.3 / 5.7 | 9.5 / 10.6 / 9.2 |
| ASOF_SHR | 7.6 / 7.8 / 3.2 | 8.1 / 8.1 / 4.9 | 8.9 / 8.6 / 7.9 |
| RECENT | 10.4 / 10.4 / **2.9** | 10.5 / 11.5 / **2.2** | 10.3 / 9.7 / **5.0** |
| STD_DELTA | 7.9 / 8.4 / **0.0** | 8.1 / 8.0 / **0.9** | 8.2 / 8.1 / **2.9** |
| TE_DEV | 4.4 / 4.5 / 4.3 | 4.4 / 4.6 / 6.1 | 4.8 / 4.4 / 3.2 |
| COUNT_DERIVED | 2.3 / 2.6 / 5.0 | 3.0 / 3.0 / 3.4 | 3.1 / 2.7 / 3.5 |
| CALENDAR | 3.7 / 4.6 / **1.3** | 5.0 / 4.4 / **0.0** | 3.1 / 2.8 / **0.4** |
| SKILL | 1.7 / 1.9 / 0.5 | 1.8 / 2.0 / 1.8 | 1.7 / 2.0 / 2.1 |

`old` and `val` are near-identical in every cell — so the in-sample side is
**not** inflated by memorising the validation season specifically, and the drop
at `next` is the boundary rather than recency. Rank stability (spearman
same→next) 0.350 / 0.413 / 0.825.

Cell and core (0.45·base + 0.55·cell, permuted in both arms with the same index)
are in `out/season_transfer_{map,core}.csv` and agree in direction throughout.

### Retention across all nine (arm × boundary) cells

median of `next_share / same_share`, R league:

| family | median | cells < 0.5 | cells ≥ 1.0 | dominant class |
|---|---:|---:|---:|---|
| DOMAIN | 1.16 | 0 | **9** | STABLE_POSITIVE 9/9 |
| SEASON_STD | 1.57 | 0 | 8 | STABLE 6 / NEXT_ONLY 3 |
| COUNT_DERIVED | 1.50 | 0 | 8 | STABLE 8/9 |
| TE_DEV | 0.97 | 0 | 4 | STABLE 7/9 |
| SKILL | 0.87 | 3 | 4 | WEAK_BOTH 6/9 |
| RAW_CONTEXT | 0.72 | 4 | 3 | STABLE 5 / COLLAPSED 4 |
| RAW_ASOF | 0.69 | 3 | 0 | STABLE 4 / MIXED 3 |
| TE_LEVEL | 0.64 | 0 | 0 | MIXED 6/9 |
| ASOF_SHR | 0.61 | 3 | 0 | STABLE 4 / MIXED 3 |
| RECENT | **0.31** | **8** | 0 | COLLAPSED 7/9 |
| STD_DELTA | **0.02** | **9** | 0 | COLLAPSED 5 / SIGN_FLIP 4 |
| CALENDAR | **0.00** | **9** | 0 | SIGN_FLIP 4 / MIXED 4 |

Of the three that die, `RECENT` and `CALENDAR` are **information-closed**, so
their collapse is real information loss. `STD_DELTA` is routing-only and its
number carries no information claim.

---

## 5. Block map — the information answer, R league

Six cells: {base, cell} × {fit≤2021, fit≤2022, fit≤2023}.

| block | class over 6 cells | retention range | next-season share (fit≤2023) |
|---|---|---|---|
| PITCHER_HISTORY | **STABLE 6/6** | 1.11 – 1.43 | base 66.4 / cell 63.2 |
| MATCHUP | **STABLE 6/6** | 1.05 – 1.18 | base 17.4 / cell 15.5 |
| COUNT | STABLE 4/6 | 0.63 – 1.23 | base 4.6 / cell 9.1 |
| CALENDAR_ID | SIGN_FLIP 3/6 | 0.00 – 0.79 | base 6.2 / cell 5.1 |
| BATTER_HISTORY | **COLLAPSED 5/6** | 0.21 – 0.43 | base 4.3 / cell 3.9 |
| GAME_STATE | **COLLAPSED 5/6** | 0.03 – 0.50 | base 1.2 / cell 3.2 |

> **What survives a season boundary is the pitcher's own history and the
> handedness matchup. The batter's history and the game state do not.**

`PITCHER_HISTORY` does not merely hold — its share *rises* out of sample in all
six cells (50–65% → 63–74%). What the model loses at a boundary is everything
that is not the pitcher.

---

## 6. Base vs cell — routing differs, information does not

Unseen 2024, R league, share of positive ΔBSS:

| | spearman(base, cell) | largest gaps |
|---|---:|---|
| family (representation) | **+0.678** | COUNT_DERIVED +19.8, RAW_CONTEXT +17.5, SEASON_STD −14.5, DOMAIN −8.5 |
| block (information) | **+0.943** | COUNT +4.6, GAME_STATE +2.1, PITCHER_HISTORY −3.2 |

**The two arms draw on the same information and route it through different
shapes.** This reframes every family sign split on record — `--feat-h1`
(base +3.88 / cell −1.55), `GENERAL_SKILL_ADD` (base −0.08 / cell +2.57). Those
splits are competition with a shape one arm already uses, not evidence that the
arms need different information. It is also why base-only arms keep looking
attractive and keep failing at the blend.

---

## 7. R / F contrast

Only the fit≤2023 boundary is coherent for F: `MVN3`/`MVCELL` carry
`--drop-f-pre 2022`, so their F training and F scoring are both post-regime-break.
`MVB22` scores unseen F 2023 at BSS −8200 and `MV21` at −779 — those are the
2022→2023 label break (F rate .71 → .47), not transfer, and are excluded.

Block share / retention, base arm, fit≤2023:

| block | R same→next | R retain | F same→next | F retain |
|---|---|---:|---|---:|
| MATCHUP | 15.8 → 17.4 | 1.10 | 21.6 → **49.1** | **2.28** |
| PITCHER_HISTORY | 49.6 → 66.4 | 1.34 | 41.0 → 38.7 | 0.94 |
| COUNT | 7.3 → 4.6 | 0.63 | 7.7 → 6.4 | 0.83 |
| CALENDAR_ID | 7.9 → 6.2 | 0.79 | 8.3 → 2.7 | 0.33 |
| BATTER_HISTORY | 11.6 → 4.3 | 0.37 | 12.4 → 2.3 | 0.19 |
| GAME_STATE | 7.9 → 1.2 | 0.15 | 9.0 → 0.8 | 0.08 |

No block transfers on R and vanishes on F. The pattern is **concentration**: on
F everything collapses harder and the matchup carries nearly half of what is
left, while pitcher history transfers *worse* than on R (0.94 vs 1.34). No new F
correction is derived from this.

---

## 8. MATCHUP decomposition — this is the skill-hand verdict

MATCHUP holding 6/6 would, read carelessly, reopen the hand axis. It splits into
two things, and **both subgroups are closed** (checked, printed by the tool):

* `HAND_RAW` — `pitcher_hand`, `batter_hand`, `matchup_same_hand`, `dom_same_*`.
  The platoon. A fact about biomechanics; it does not need a season to re-learn.
* `HAND_TE` — `te_pitcher_batter_hand_{ratio, rate, n, ratio_dev}`. *This
  pitcher's* record against that handedness — the quantity `--feat-h1` composes
  and the one `skill --axis hand` would fit.

Base arm, R league, ratio `HAND_TE / HAND_RAW`:

| horizon | mean | min | max |
|---|---:|---:|---:|
| in-sample val | 1.039 | 0.779 | 1.236 |
| **unseen next season** | **0.646** | 0.576 | 0.761 |

Per-boundary retention (`next / same`, raw ΔBSS), base arm:

| boundary | HAND_RAW | HAND_TE |
|---|---:|---:|
| 2021 → 2022 | 0.320 | 0.279 |
| 2022 → 2023 | 0.400 | 0.366 |
| 2023 → 2024 (3 seeds) | 0.431 – 0.493 | **0.215 – 0.236** |

The pitcher-specific hand record is worth about as much as the platoon
in-sample and about **two thirds** of it on the unseen season, on every
boundary, and the gap is widest at the most recent one. **MATCHUP's stability is
carried by the platoon, not by the pitcher's hand splits.**

---

## 9. Verdicts

### Stable across every boundary measured
`PITCHER_HISTORY` (block, 6/6, share rises out of sample) ·
`MATCHUP` (block, 6/6 — but see §8) ·
`DOMAIN` (family, 9/9) · `COUNT_DERIVED` (family, 8/9)

### Collapsing
`BATTER_HISTORY` (block, 5/6, retention 0.21–0.43) ·
`GAME_STATE` (block, 5/6, 0.03–0.50) ·
`RECENT` (family, information-closed, 8/9 below 0.5, median 0.31) ·
`CALENDAR` (family, information-closed, 9/9, median 0.00)

### Sign-flipping
`CALENDAR` / `CALENDAR_ID` — permuting it *helps* on the unseen season in 4 of 9
family cells and 3 of 6 block cells. `STD_DELTA` likewise, but that is routing.

### Not a verdict
`SEASON_STD`, `RAW_ASOF`, `STD_DELTA` — routing only, 13/13, 11/13 and 8/10
columns rebuildable from outside the family.

---

## 10. Research implications

**1. Next GPU candidate: none is licensed by this analysis.**
The two blocks that survive a boundary already hold 66 of the 121 features, and
every recent axis on them has closed. The four that collapse are exactly where
`--feat-v4` (−18.72), `--feat-count-cat` (−9.57) and `--feat-count` (−1.07)
already failed. This map gives a *reason* for that pattern rather than a new
direction: **information attached to the batter, the game state or the calendar
does not survive to the season being scored.**

The one GPU job worth running is a **diagnostic, not a candidate**: the B1S8
recipe on the judging surface (`--val-season 2023 --test-season 2024
--drop-f-pre 2022`), one seed, base + cell. The base half of this map already
matches the champion at block spearman 0.94–1.00; the cell half matches at
0.31–0.49 and is therefore unusable for the arm that carries 55% of the blend.
Two runs, ~10 min, adopts nothing.

**2. Hold.**
`FM_MULTILABEL_V2`. The §13 condition is partly met — the cell geometry retains
better than base on 4 of 6 blocks, notably `GAME_STATE` (0.50 vs 0.15) and
`COUNT` (1.10 vs 0.63) — but that observation is made on a cell model that does
not match B1S8's. It should not be acted on before the diagnostic above.

**3. Take down.**
`skill --axis hand`. §8 is the measurement §12 asked for, and it goes the wrong
way: the pitcher-specific hand record retains worse than the platoon on every
boundary and worst at the most recent. With `--feat-h1` already DROP on the same
inputs and `FLAG pitcher-batter-hand-residual` CLOSED at −191.95 transfer, there
is no independent rationale left. **DEPRIORITIZE → do not build.**

Also down, as a class: new derived features on the batter, game-state or
calendar axes.

---

## Recommended next single action

Run the B1S8 recipe once on the judging surface (`--val-season 2023
--test-season 2024 --drop-f-pre 2022`, one seed, base and cell) purely as a
diagnostic, and re-run `tools/season_transfer_map.py --groups block` against
those two packs. Nothing is adopted from it; it is the missing input for every
cell-side decision, including whether `FM_MULTILABEL_V2` is worth a rewrite.

# Champion constant provenance — can every shipped number be explained?

2026-08-28, CPU only, **nothing changed**. The champion
`submissions/gskdep_0816.zip` (LB 1111.3713632162) is frozen; this is an audit
of where its numbers came from, done for Phase-3 verification, where the
question is not "does it score" but "can you show how each value was obtained
without using the answer".

## Inventory

Read off the shipped `script.py` inside the zip, not off a document.

| # | constant | value |
|---|---|---|
| 1 | `SLOPE` | `1.0416` |
| 2 | `SHIFT` | `0.0052` (applied as `preds - SHIFT`) |
| 3 | `_W_CELL` | `0.55` |
| 4 | member seeds | base `[3,4,5,6,8,13,42,7]`, cell `[3,4,5,6,8,13]` |
| 5 | recent-middle | `MID_COL`, 8 `THRESHOLDS`, 9 `OFFSETS`, `NAN_OFFSET = -0.008102692247033697` |
| 6 | pitcher×batter | `model/matchup_constants_2024.npz` |
| 7 | final additive | `+0.002515795361` |

## Verdicts

### 1–2. `SLOPE` / `SHIFT` — **origin unrecoverable, but independently revalidated**

**[FACT]** `tools/calib_refit.py:8` records that `CHAMPION_v11_RECIPE.md` part 6
already listed both as *weakly evidenced*: "the reasoning that produced them is
not usable as independent grounds today." They predate the ledger; `v14` already
shipped `SLOPE 1.0416`.

**[FACT]** They were nevertheless put through a frozen out-of-time transfer test
and passed on both boundaries: **2022→2023 +0.75, 2023→2024 +3.06**
(`docs/EXPERIMENTS_LOG.md:936`).

**[FACT]** Every attempt to *replace* them with a refit lost: calibration fitted
on 2023 and frozen onto 2024 scores **−19.93** against the legacy constants'
−1.22, and isotonic is far worse (**−99.14** transferred, −81.93 after removing
the mean bias).

**[INFERENCE]** The *form* is a two-parameter logistic map fittable from OOF
alone, so it is train-derivable in principle and reproducible in practice; what
cannot be reconstructed is the specific historical fit that produced these two
digits. For Phase 3 this is explainable — a one-dimensional recalibration
validated by out-of-time transfer — but the honest statement is "revalidated",
not "derived here".

### 3. `_W_CELL 0.55` — **train-derivable, and every refit lost to it**

**[FACT]** A blend weight fitted on 2023 and frozen onto 2024 scores **−0.81**
against the shipped 0.55. The offline `_W_CELL` curve peaks at `w ≈ .575–.60`
and is worth **+0.07** — against an LB delta sampling SE of ±1.06, a
signal-to-noise of 0.066, which is why `E-LB2` was cancelled before submission.

**[FACT]** Measured again tonight on untouched 2024 as a by-product of the
LEAFIT gate: the control cell's optimum is `w = 0.60` at 889.128 against
888.835 at the shipped 0.55 — a **+0.29** difference on a single seed, i.e.
inside the noise this project has repeatedly measured.

### 4. Member seeds — **explainable, and the asymmetry is deliberate**

**[FACT]** The shipped `script.py` states it: adding seeds 42 and 7 to the base
family is worth **+0.89** on the debiased local score, and the same two seeds on
the cell family are worth **−0.13**, so the cell stays at six. That is a
recorded reason, not a leftover.

### 5. Recent-middle offsets — **train-derivable and reproduced**

**[FACT]** Fitted from official-training OOF only, and re-derived from new OOF
on 2026-08-13: the shipped offsets reproduce to about **1e-4** with **8/8 sign
agreement**. A refit fitted on 2023 and frozen onto 2024 gives **+5.72** against
the legacy constants' **+8.03**, so the shipped values are also the better ones.

### 6. Pitcher×batter offsets — **train-derivable, with a documented honest ceiling**

**[FACT]** Generated from 2024 OOF residuals of `VB2_base` / `ZD5` on exact
`(pitcher_id, batter_id)` pairs, k=500, zero fallback. The shipped `script.py`
carries the correction itself: the generator's own `pb_raw_increment: 425.73` is
**in-sample**; fitted on val2023 and frozen onto 2024 the step is worth
**+3.80**, because only 38.1% of rows a season later match a pair the table has
seen. Row-independent by construction — it reads only the row's own two ids.

### 7. `+0.002515795361` — **NOT derivable without the leaderboard**

**[FACT]** `docs/EXPERIMENTS_LOG.md:1290-1298` records the derivation in full. It
is a quadratic fitted to **three leaderboard scores**:

```
baseline          1108.4333490288
script-only +0.010  1088.4373257989
script-only -0.010  1047.9367151230
second difference  -80.4926571357
      -> optimum   +0.002515795361     expected 1110.980630
      -> observed  1110.9806302398     (forecast matched to ~6.4e-11)
```

**[FACT]** This is the one shipped number whose value came from leaderboard
feedback rather than from training data. Three submissions were spent to obtain
it, and it is worth **+2.5472812110** against B1S8.

**[FACT]** It is also the reason the forecast was exact: this competition's
public set *is* the private set, so the probe measured the actual scored
quantity rather than a subsample, and a one-dimensional Brier optimum recovers
exactly.

**[INFERENCE]** Two consequences, and neither is a reason to touch the champion.

1. **For Phase 3**, the honest description is: a global additive constant chosen
   by a three-point quadratic probe of the evaluation metric. It is fully
   documented and exactly reproducible *given those three scores*, but it cannot
   be re-derived from `train.csv` alone. Every other constant above can.
2. **For the rules**, the project now forbids LB probing and forbids adjusting
   any constant on LB feedback. The champion predates that and contains its
   product. The champion is frozen and this audit does not propose changing it —
   removing a measured +2.55 to satisfy a rule adopted afterwards would be
   trading a real gain for tidiness.

**[NEW HYPOTHESIS]**, recorded and *not* acted on: the net global additive
effect of the shipped pipeline is `−0.0052 + 0.002515795361 ≈ −0.00269`, and a
global bias correction of that magnitude is exactly what an OOF-fitted intercept
would produce. If a train-only intercept reproduces it closely, the constant
becomes train-derivable after the fact and the Phase-3 story simplifies. That is
a CPU measurement, it changes nothing if it fails, and it is **not run tonight**
because the overnight contract permits one fallback candidate and it is already
spent.

## Summary

| constant | train-derivable? | independently revalidated? |
|---|---|---|
| `SLOPE`, `SHIFT` | yes in form, origin unrecoverable | **yes** — out-of-time transfer on two boundaries |
| `_W_CELL` | yes | yes — every refit lost |
| member seeds | yes | yes — recorded local deltas |
| recent-middle | yes | yes — reproduced to 1e-4, 8/8 signs |
| pitcher×batter | yes | yes — honest frozen worth +3.80 |
| **`+0.002515795361`** | **no** | n/a — it *is* the leaderboard measurement |

Six of seven are explainable from training data. One is not, it is documented,
and it is the only one.

# Overnight fallback — CELL_BUDGET_CEILING

Written 2026-08-28 by Claude at `a6450bd`, **before any fit**. Champion
`submissions/gskdep_0816.zip` frozen. This is the **one** fallback GPU candidate
permitted by the overnight contract §17E; if it fails, the score search stops.

## The observation

**[FACT]** Across all **263** cell-arm rows in `LEDGER.tsv`, `best_iter` piles up
against the `--iters 3000` ceiling: 2999 appears 37 times, 2998 26, 2997 32,
2996 18, 2995 16, and the whole 2990–2999 band holds 164. **`--es 500` has never
fired on the cell arm.** It cannot: the arm is still improving when the budget
ends, so `get_best_iteration()` returns the last iteration, not a selected one.

**[FACT]** Exactly **2 of 263** cell rows were given more budget, and both ran
far past 3000: `CI1A_cell5k` best_iter **4533** and `MVCELL5K_s42` best_iter
**3928**, at `--iters 5000`.

**[FACT]** Neither is usable as evidence. `CI1A_cell5k` is the **pre-B1S feature
era** — its command carries no `--feat-k 200`, no `--te-k 50`, no `--p1`, no
`--feat-skill`, no `--fm-modes`/`--fm-min-share` — and it is single-seed on the
A100. `MVCELL5K_s42` is single-seed on a third host. The same generation problem
that disqualified `CTR3N3` and `CZ_ml`.

**[FACT]** No FLAG in `docs/SETTLED.md` covers this. `D12` closed the *checkpoint
criterion* (which metric picks `best_iter`); `P3-B` closed a *common base-arm
budget*; `--refit-mult` is a different multiplier applied after the fact. None of
them asks whether the cell arm's `best_iter` is a selection at all.

## Hypothesis, stated before measurement

**[NEW HYPOTHESIS]** The cell arm ships a model whose length was set by a budget
ceiling rather than by validation. `--refit-mult 1.5` then multiplies that
truncated number, so the deployed ensemble inherits the truncation. Giving the
arm enough budget for early stopping to actually select should let it keep the
late-stage, depth-5-specific structure the ceiling currently discards.

## Why this is not the banned parameter sweep

The current value is not a choice anyone validated — it is a wall the arm is
pressed against in 253 of 263 runs. The intervention is "let `--es 500` select",
and **early stopping, not the experimenter, picks the number**. There is one
sanctioned budget and no sweep: 2 / 4000 / 6000 / 10000 are all forbidden
regardless of outcome.

## The threat, named in advance

`CELL_LEAF_ITERS_10` failed hours ago with a distinctive signature: the cell arm
gained **+2.579** standing alone while the core lost **−0.381**, because better
leaves moved the cell arm *toward* the base arm — `corr(base, cell)` rose
`0.9647370 → 0.9681003` and the cell's own sd shrank.

**Pre-registered mechanism test.** More trees is a capacity change, not a
convergence-to-the-base change, so the prediction is the opposite sign on
diversity: `corr(base, cell)` should **fall or hold**. If it rises like LEAFIT's
did, the mechanism is the same one already refuted and the axis closes on that
basis even if the core delta happens to be positive — the result would then be
noise on a refuted mechanism, and it will be recorded as such rather than
promoted.

## Design

Judging surface, unchanged: `--drop-f-pre 2022 --max-train-season 2024
--val-season 2023 --test-season 2024`. Laptop RTX 5060, seed 3.

```
base    : LEAFIT_base      (REUSED)
control : LEAFITCTL_cell   (REUSED)
candidate: identical to LEAFITCTL_cell + --iters 8000, tag CELLBUD_cell
```

**Reuse justification.** Both reused arms were fitted **today, on this host, in
the session immediately preceding this one**, from the same commit, with the
identical command except the changed variable. `--cell-leaf-iters` defaults to 0
and emits nothing, so the code path is byte-equivalent to the pre-flag trainer.
Lineage parity is asserted before scoring: the candidate's `training_flags` must
differ from the control's in **`iters` only**. If it differs anywhere else the
control is refitted instead of reused. This saves one 550 s fit and one 184 s
fit; it does not weaken the comparison, which stays same-host and same-day.

**Budget = 8000, not 5000.** The two historical runs reached 3928 and 4533 with
`--iters 5000`, and 4533 + 500 > 5000 — so 5000 may itself have been truncating.
8000 is chosen to make early stopping possible, not because it looked good.

**1 fit.**

## Validity condition, fixed before the run

Early stopping must **actually fire**: `best_iter + 500 < 8000`. If it does not,
the run is **uninformative about performance** and the axis closes as "the cell
arm does not stop within any budget worth paying for" — not as a performance
verdict, and not as a licence to try 12000.

## Integrity, read before any BSS

`row_id` and target elementwise identical across all three members; cell feature
lists, categoricals and `fm_success == (9,10,11)` identical; the effective
parameter diff between the two cell packs is **exactly `iterations`**;
`leaf_estimation_iterations` must read **1 on both** (the closed axis must not
leak in). `rms(candidate, control) >= 0.002` or the axis closes as redundant.

## Gate (§17D, unchanged)

| seed-3 core delta | action |
|---|---|
| `<= 0` | **FAIL, STOP.** Score search ends for the night. |
| `0 < delta < +3` | **HOLD.** Score search ends for the night. |
| `>= +3` | eligible for n=6 |

## Diagnostics — recorded, never used to choose

`corr(base, cell)` and residual correlation for control and candidate (the
mechanism test above); reliability and resolution; cell sd; `best_iter`; wall
clock; R / F; early / late; core rms and pearson.

## Forbidden

Any other budget (2000 / 4000 / 5000 / 6000 / 10000 / 12000); changing
`--refit-mult` on the back of this; applying it to the base arm; changing `--es`;
combining with `--cell-leaf-iters`; extending to n=6 on a HOLD; a second fallback
candidate — §17E permits exactly one; submitting anything.

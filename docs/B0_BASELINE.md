# B0-JL — the new zero point

Measured 2026-08-13 on `desktop-5070` (RTX 5070 Ti), seeds 3, 4, 5.
This is the baseline every post-audit judgement is measured against.

**Do not compare it to the old v11 LB 1101.802, and do not try to make it
match.** Different surface, different machine, different code. The audit found
ten rolling runs invalid because future seasons leaked into the fit pool
(`--test-season 2023` kept 253,507 rows of 2024); every number produced before
that fix is on a different footing. Forcing agreement would just hide it.

---

## Score

Judging surface `--val-season 2023 --test-season 2024`, unseen 2024, 253,507 rows.

| arm | seed 3 | seed 4 | seed 5 | mean | sd |
|---|---:|---:|---:|---:|---:|
| `B0JL_base` (binary, depth 8) | 860.40 | 872.10 | 871.43 | **867.98** | 6.57 |
| `B0JL_cell` (14-cell MultiClass, depth 5) | 880.42 | 885.11 | 882.18 | **882.57** | 2.37 |

`.45 / .55` core: **890.50** per-seed mean, **892.54** ensembled.
Validation 2023 (245,525 rows): base 588.07, cell 596.56, core 611.34.

Debiased-to-LB conversion (`+128.4`) is **not** applied here. That offset was
derived on the submission surface, which does not use `--drop-f-pre 2022`;
this one does.

## Sanity

- `collapse check: ok` — every arm's raw BSS is positive and prediction means
  sit +0.0019 to +0.0029 from the truth. Selection now reads `raw_bss()`, so a
  collapsed run reports a negative number instead of a clipped 0.0.
- Partition asserts passed on all six runs: no season > 2024 in the frame, no
  2023 or 2024 row in the fit partition, no validation/test overlap.
- Early stopping behaved on `base` (best_iter 671 / 744 / 839, cap 3000).
- **`cell` is pinned at the cap** (2998 / 2998 / 2990) — exactly as the shipped
  `cat_ZD5` was. Iterations ran out; this is not convergence. Raising the cap to
  5000 is already closed (`failmode-cell-iters-5000`, −8.67), so the cap is
  acting as transfer regularisation. Treat it as a fixed part of the recipe, not
  as a tuned value.

## Provenance

```
feature fingerprint  ac03d14a49872b1e   (121 features, both arms)
rows                 fit 870,752 | val 245,525 | unseen 2024 253,507
cell taxonomy        12 cells, 99.85% recovery, 3 success cells
refit trees   base   s3 1008  s4 1117  s5 1260
              cell   s3 4473  s4 4498  s5 4486
host                 DESKTOP-053T952
```

Resolved args are in `out/lineage_B0JL_base.json` / `_cell.json` in full. The
two commands differ only in `--depth 8` vs `--depth 5 --failmode-cells
--fm-modes middle,ball,reverse --fm-min-share 0.005`:

```
--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev
--feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain
--feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254
--refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024
--val-season 2023 --test-season 2024 --seeds 3,4,5
```

⚠ **Both lineage records carry `"commit": ""`.** The 5070's `C:\aimers` is
populated by scp and is not a git checkout, so `git rev-parse` found nothing.
The source was local `571d791`. `tools/deploy_5070.sh` now stamps
`.deployed_commit` and `tools/lineage.py` falls back to it, marking the record
`"source": "file"` so a stamped revision is never read as a verified checkout.
B0-JL itself keeps the hole — it is recorded, not repaired.

### One stray ledger row

`LEDGER.tsv` also holds `2026-08-13 03:29 | B0JL_base | seed 3 | best_iter 673 |
579.24 | 859.97`. **That is not part of this baseline.** It is the remnant of
the first launch, where cmd split the unquoted `3,4,5` and only seed 3 ran; two
GPU jobs then overlapped and both were killed. Its artifacts were purged.

It is left in the ledger rather than deleted, and deliberately *not* added to
`docs/INVALIDATED.tsv`: the guard matches tags exactly, and the string
`B0JL_base` is also the family name every report passes, so listing it would
refuse the valid baseline. It cannot contaminate anything on its own — it has no
`_sN` suffix, and every glob in the tools is `*_{tag}_s*`.

## What this baseline is and is not

- It **is** the reference for finalist adoption on the judging surface, at
  n ≥ 6 paired seeds, same host, t ≥ 2.4.
- It is **not** a submission shape. It uses `--drop-f-pre 2022`; the submission
  surface does not, and mixing the two flag sets is how `P2'-B` produced
  `best_iter` 12–21 (`SETTLED.md` → `drop-f-pre-omitted | BANNED`).
- It is **not** a P1 run. `--p1` was off, so this reproduces legacy-like
  behaviour on purpose: the fit-only TE prior and the neutral first-season
  skill fallback both change the artifact, and B1-J exists to measure that
  change against this line rather than against a moving target.
- Three seeds is a screen, not an adoption bar. Finalists need six.

---

# B1-J — the adoption surface

Same host, same core, seeds 3, 4, 5, 6, 8, 13, with `--p1` (all three P1
validation-contract fixes). This is the audit's B1-J: the surface a finalist is
adopted on. B0-JL above stays what it is, a three-seed screen.

## Score

| arm | mean (6 seeds) | sd | ensembled |
|---|---:|---:|---:|
| `B1J6_base` | 866.23 | 2.10 | 874.00 |
| `B1J6_cell` | 886.52 | 2.82 | 889.78 |
| `.45/.55` core | 892.31 | — | **894.93** |

Validation 2023 ensembled: base 602.54, cell 601.45, core 616.80.

## Paired against B0-JL (common seeds 3, 4, 5)

| arm | unseen 2024 | val 2023 |
|---|---|---|
| base | −0.65 (SE 3.09, t −0.21) | +7.74 (SE 3.71, t +2.09) |
| cell | +2.45 (SE 1.28, t +1.92) | +0.81 (SE 1.30, t +0.62) |
| **core** | **+1.12** (SE 0.68, t +1.65) | **+4.16** (SE 1.64, t +2.53) |

The point of this table is not the size. It is the sign.

A half-applied P1 -- the two cheap fixes with the artifact still shared between
stages -- measured **−2.41 (t −2.51) on unseen 2024 while gaining +3.56 on
validation**. Two surfaces disagreeing at t > 2.4 in opposite directions is not
seed noise; it was the stale deployment prior. Completing contract 1/2 removed
the conflict: both surfaces are now positive. Nothing about the two cheap fixes
changed, and the selection-stage score is untouched by construction (seed 3
base val was 588.68 before and after).

Neither surface reaches the +3 adoption bar, and it does not matter here. P1 is
a correctness contract applied regardless of score direction (audit 1.5), not a
candidate. What this measures is the cost of the contract, and the cost is not
negative.

## Provenance

```
feature fingerprint  ac03d14a49872b1e   (121 features, both arms -- same as B0)
rows   selection fit 870,752 (<= 2022) | deployment fit 1,116,277 (<= 2023)
       val 245,525 | unseen 2024 253,507
refit trees   base   901 - 1312
              cell   4428 - 4500   (still pinned at the 3000-iteration cap)
host                 DESKTOP-053T952
source               a913cec (stamped via .deployed_commit)
```

`collapse check: ok`. Prediction means sit +0.0017 to +0.0029 from the truth on
both arms -- tighter than B0, where base ran to +0.0029 with sd 6.57 across
three seeds. B1-J base sd is 2.10 across six.

## What this is not

- Not a submission shape. Still `--drop-f-pre 2022`.
- Not a temporal generalisation claim. Same-seed t is stochastic stability on
  one machine, nothing more (audit 6.3). The 2022→2023 stress case is separate.
- Not attribution. `--p1` bundles three fixes; the individual arms are what say
  which one moved the number.

---

# Attribution — the four single-change arms

Each arm adds exactly one flag to the fixed B0-JL core, seeds 3, 4, 5, same
host. Screening scale (audit 6.3): these terminate obviously-negative
candidates and declare nothing.

`.45/.55` core, against B0-JL:

| arm | unseen 2024 | val 2023 |
|---|---|---|
| `--te-fit-prior` | −0.01 (SE 1.36, t −0.01) | +2.61 (t +1.15) |
| `--skill-neutral-first` | **−5.59** (SE 0.96, **t −5.83**) | **+5.97** (t +6.04) |
| `--two-stage-artifact` | +1.01 (SE 1.36, t +0.74) | +0.15 (t +0.96) |
| `--anchor-last-pitch` | +1.80 (SE 1.16, t +1.56) | +0.58 (t +0.32) |
| `--p1` (the first three together) | +1.12 (SE 0.68, t +1.65) | +4.16 (t +2.53) |

## The three P1 fixes are strongly non-additive

```
sum of the three single arms   −4.59
measured together (--p1)       +1.12      interaction +5.71
```

This kills the arithmetic I used before these arms existed. When only the two
cheap fixes had run I attributed their −2.41 to the stale TE prior and the
season drift behind it (.5495 in 2019 → .4861 in 2024). **The TE prior is not
the cause: alone it is −0.01 on unseen 2024.** The drift numbers were right and
the causal claim was wrong — it was reasoning from a plausible mechanism
instead of from a measurement, and the measurement disagrees.

What actually costs is `--skill-neutral-first`, and only in combination does it
come back. Do not attribute any part of a bundled result by subtraction here.

## Every contract fix hurts base and helps cell

Unseen 2024, per family:

| arm | base | cell |
|---|---|---|
| `--te-fit-prior` | −6.63 (t −2.34) | +5.02 (t +3.51) |
| `--skill-neutral-first` | −16.34 (t −3.92) | +1.64 (t +1.32) |
| `--two-stage-artifact` | −2.48 (t −1.16) | +3.58 (t +2.25) |
| `--anchor-last-pitch` | +1.02 (t +0.50) | +2.83 (t +2.43) |

The sign is the same in all four rows, and the `.45/.55` blend is cancelling
most of it. `--skill-neutral-first` touches one thing -- the 2019 rows
(211,627, 15.4% of the frame) lose their skill estimate to NaN -- and the
binary family loses 16 points.

That is worth a follow-up arm, not a revert. The audit's wording is "사전 고정
neutral/결측"; NaN was my choice because CatBoost handles it natively. A base
model reacting this hard suggests NaN is being learned as *a 2019 marker*
rather than as absent information, which a fixed neutral value would not be.

## Status

- The three P1 fixes are a correctness contract, applied regardless of sign
  (audit 1.5). These numbers are their cost, not a vote.
- `--anchor-last-pitch` is a live candidate: +1.80 with a 95% upper bound of
  +6.8, so rule 5 neither adopts nor rejects it at n=3. It goes to six seeds on
  B1-J (`ANCH6`), which is the adoption surface -- these screens ran on B0-JL.

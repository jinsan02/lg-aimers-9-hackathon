# SK40 and ICOH — two rule-violation recoveries, pre-registered together

Written 2026-08-16 by Claude at `81a1287`, **before any fit**, and queued to run
unattended after FMCOARSE stage 1. Champion `submissions/gskdep_0816.zip` is
frozen. Neither experiment depends on FMCOARSE's outcome.

Both axes were recorded CLOSED in violation of this project's own rules. Neither
is a new hypothesis: the original numbers stand, only the statistics behind the
verdicts were wrong.

---

# SK40 — `--std-k 40` on the submission surface, 5070, both arms

## What is wrong with the existing verdict

`docs/SETTLED.md` FLAG `--std-k 40` reads *CLOSED, submission surface −1.954,
SE 2.377, t=−.822*. Two defects:

1. **Rule 5.** DROP requires the 95% upper bound below +3. With n=8 the
   critical value is `t(.975, df=7) = 2.365`, so the bound is
   `−1.954 + 2.365 × 2.377 = +3.67`. That is a **PARK**, not a CLOSE.
2. **Cross-machine, which is BANNED.** The number is `SK2_k40 − VB2_base`, a
   **4070** run, **base arm only**. The champion lives on the 5070, and FLAG
   `cross-machine-compare` is BANNED precisely because *std-k 40 with identical
   config and identical 6 seeds diverged by 11 points across two machines*.

So the axis has never had a same-machine, both-arm, core-level judgement on the
surface that decides.

## Why k = 40 and not k = 20

k20 looks more attractive — the judging surface gave k10 +2.86, **k20 +3.21
(t=+2.01)**, k40 +4.27 — and an earlier audit recommended it on that basis.
**That curve is not usable as a selection signal**, because it is contaminated
by the same machine effect: the identical k40 arm scored **+4.27 on the 4070 and
+10.31 on the A100**. Choosing the best-t point off a cross-machine curve is
exactly the self-selection that `--feat-id-cohort` was (wrongly, but for the
right kind of reason) rejected for.

`k = 40` is therefore chosen because it is **the value carrying the recorded
rule-5 violation**, not because it looked best. No selection is involved.

**Pre-committed consequence:** if k40 DROPs cleanly here, the **entire downward
`--std-k` direction closes with it** — no k20, no k10, no k30, no sweep, and the
judging-surface positives are then recorded as surface-transfer failure, the
same shape as `--te-halflife 2`.

## Design

Submission surface (`--val-season 2024`), 6 shared seeds `3,4,5,6,8,13`, one
host (`DESKTOP-053T952`), one session, fresh controls on **both** arms because
`--std-k` feeds `--feat-std`, which both arms use. 24 fits.

```
SK40CTL_base / SK40CTL_cell : champion recipe, --std-k 80   (control)
SK40_base    / SK40_cell    : identical, --std-k 40         (candidate)
```

One change. Nothing else differs.

---

# ICOH — `--feat-id-cohort` at n = 6

## What is wrong with the existing verdict

FLAG `--feat-id-cohort` is CLOSED on *transfers −3.18 / +13.93 / +8.13, 4070
−2.42*. All **8** ledger rows are **single-seed** (`IC1A_idcohort` s3,
`IC1V_idcohort` s42, `IC1A22`/`IC1A21` s3, plus the four `IC2*` role variants),
verified directly in `LEDGER.tsv`. The measured base seed spread on this project
is **−7.46 to +6.74**, so every one of those four numbers sits inside the noise
floor and the axis was rejected by an *argument*, never by a statistic.

Two of the three boundaries are strongly positive (+13.93, +8.13) and the
mechanism is documented and mundane: the id prefix encodes debut cohort
(242/243 → 2022, 244 → 2023, 245 → 2024), which is real information the model
otherwise has to infer.

## Design

Submission surface, same 6 seeds, same host and session, fresh controls on both
arms. `--feat-id-cohort` is a `store_true` flag; `--id-cohort-roles` stays at its
default `pb`, and **no role variant is run** — choosing among p / b / pb after
seeing a result would recreate the original defect. 24 fits.

```
ICOHCTL_base / ICOHCTL_cell : champion recipe                (control)
ICOH_base    / ICOH_cell    : identical + --feat-id-cohort   (candidate)
```

---

# Shared contract for both

**Deciding surface** is the submission surface, fixed here in advance. No
judging-surface run is performed for either axis, and no judging-surface number
may be quoted as an expected leaderboard gain — the correction the GSK review
produced.

**Gate**, `tools/judge.py` unmodified, on the paired per-seed delta of the fixed
0.45/0.55 core:

| condition | verdict |
|---|---|
| Δ ≥ +3 **and** t ≥ 2.4 **and** n = 6 | **KEEP** |
| 95% Student-t upper (2.571 × SE) < +3 | **DROP** |
| otherwise | **PARK**, no seed extension |

**Integrity before any number is read**: one host, one surface, six shared
seeds; `row_id` and target identical **elementwise** across all 24 members of
each experiment; feature hash and effective-parameter snapshot compared; every
candidate pack reloaded in a fresh process reproducing its stored predictions
with max diff 0. For ICOH the feature count changes by design, so
`member_fingerprint.py` is expected to flag it and that flag is not an error.

**Diagnostics** (never used to choose): overall; R / F; row halves; calendar
halves; known / cold pitcher; reliability and resolution; RMS and Pearson
against the control; per-seed deltas.

**Forbidden**: any other `--std-k` value in either direction; `--id-cohort-roles`
variants; seed extension on a PARK; combining either axis with the other or with
FMCOARSE; changing the deciding surface after seeing a number; adding these to
any PARKed axis; submitting without explicit user approval.

## Execution note

Both run from `scripts/chain_sk40_icoh_5070.bat`, which **waits for FMCOARSE
stage 1 to finish before starting** (one GPU job per machine) and **aborts after
60 minutes** if the completion marker never appears. Failing closed is
deliberate: a crashed FMCOARSE must not silently become a concurrent run.

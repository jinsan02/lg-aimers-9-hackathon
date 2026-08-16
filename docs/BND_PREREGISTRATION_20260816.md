# BND21/BND22 — clean bounded boundary arrays, and the contract for spending them

Written 2026-08-16 by Claude at `fbe7792`, **before any fit**. Champion
`submissions/gskdep_0816.zip` is frozen. Nothing here is itself a candidate for
adoption: this run produces **artifacts**, not a verdict.

## Why

FLAG `human-baseball three-transition gate` downgraded seven verdicts because
two of their three rolling boundaries rest on arrays now listed in
`docs/INVALIDATED.tsv` (`MV21_base`/`MV21_cell` for 2021→2022,
`MVB22_native`/`MVCELL22` for 2022→2023), and the clean alternatives those
audits were assumed to use (`H21_*`, `HB22_*`, `HC22_*`) **do not exist in
`out/` at all**. Seven axes are therefore stuck on single-boundary evidence,
and the standing rule forbids killing an axis on one boundary alone.

The unlock is one shared prerequisite, priced once instead of seven times:
`--max-train-season`-bounded base/cell pairs on both missing boundaries.

## What is built

Champion recipe, submission-surface form, with the boundary moved back and
**both** arms bounded. 4 arms × 6 seeds `3,4,5,6,8,13` = **24 fits**, one host
(`DESKTOP-053T952`), one session.

```
BND22_base / BND22_cell : --val-season 2022 --max-train-season 2022
BND23_base / BND23_cell : --val-season 2023 --max-train-season 2023
```

`--max-train-season` is on **both** arms, not just the cell arm as in the
2024-boundary scripts. On the 2024 boundary it is a no-op because no season
follows; on 2022 it removes 2023 and 2024 from the deployment refit. Omitting it
is precisely the defect that invalidated ten rolling runs on 2026-08-13
(`train_gbdt2.py:2018-2030`), and `_assert_partitions` now refuses the run
rather than printing "partitions ok", so a mistake here fails loudly.

No `--drop-f-pre`: the submission surface does not carry it, and FLAG
`--drop-f-pre 2022 on the submission surface` measured it at **−61.0**.

## The contract for spending them — fixed here, in advance

These arrays will be used to re-judge seven downgraded axes:
`historical-lineup-role`, `recent-state-empirical-bayes`, `workload-pace`,
`team-call-style`, `intent-execution-bilinear`, `pb-familiarity-adaptive-k`,
`prior-pa-depth-proxy`.

1. **Seven axes × two boundaries is fourteen looks.** At the usual 5% level,
   roughly one spurious "significant" boundary is expected by construction.
   The re-judgement therefore requires **both new boundaries to agree in sign
   with the surviving 2023→2024 leg**, not a single boundary clearing a bar.
2. **No axis is reopened for adoption by this run.** The most a re-judgement
   can produce is *re-closed on three clean boundaries* or *genuinely open,
   warrants its own preregistration*. Adoption still needs a same-machine,
   6-seed, submission-surface core comparison of its own.
3. **The arrays are not a candidate pool.** Ranking the seven by their new
   numbers and pursuing the best is post-hoc family selection and is BANNED.
4. **Sign flips are evidence against**, not a reason to average.

## Integrity

`row_id` and target elementwise identical across the 6 members of each arm;
`partitions ok` printed by `_assert_partitions` for every fit and read before
any BSS; fit-row counts recorded per boundary; the ledger synced with
`tools/ledger_sync.sh` before anything is judged. Cross-boundary comparison of
raw BSS is meaningless (different validation seasons, different base rates) and
will not be quoted.

## Forbidden

Adding a 2020→2021 boundary to "complete the series" (2019–2020 are the seasons
`--min-season 2021` removed, and that axis is CLOSED at −95.09); using these
arrays to re-tune any champion constant; judging any axis on one boundary;
selecting among the seven after seeing results; submitting without explicit
user approval.

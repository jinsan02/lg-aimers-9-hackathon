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

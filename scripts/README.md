# scripts/ — one-shot runners

## Where things go

| Location | Contents |
|---|---|
| `tools/*.py`, `tools/*.sh` | **Reusable.** Tools, verifiers, launchers |
| `scripts/` | **Runs once, soon.** Experiment runners and submission helpers |
| `scripts/archive/` | Finished runners. Results live in `LEDGER.tsv` and `docs/SETTLED.md` |

**Never put a `.sh` or `.bat` in the project root.** The rule was here already and got
broken twice — 30 files on 08-08, 51 more on 08-12 — so `tools/agent_sync.sh end`
now exits 2 when it finds one.

## What is currently live (2026-08-12)

| File | Why it survives |
|---|---|
| `make_final_constants.bat` | **Builds `model/final_constants_2024.npz`, which the v11 champion reads.** Without it the submission cannot be reproduced |
| `make_matchup_constants.bat` | Same family — the pitcher × batter residual table |
| `verify_v11pb.bat` · `audit_v11pb.bat` | Verification and audit of the current champion |
| `run_tdec1_5070.bat` | Queued — column-token decoder on `desktop-5070` |

The other 126 runners moved to `scripts/archive/`. Reproduce from the verbatim
command in `LEDGER.tsv`; the files themselves remain in git history.

## Why the archive is kept at all

`LEDGER.tsv` records the **verbatim command** for every seed, so the ledger alone is
enough to reproduce. But the ledger only starts at 08-08 01:30 — for anything older,
the script is the only record, so those are not deleted.

## Archive index (2026-08-07 ~ 08)

| Script | Experiment | Verdict |
|---|---|---|
| `o1.bat` `o1.sh` | Re-validate Optuna's best config on holdout seeds | rejected −7.44 (t=−2.65) |
| `zd5.bat` | Rebuild the depth5 cell member (with refit) | **adopted** — shipped in v15 |
| `tm.bat` | 13 Trackman pitcher×season summaries | rejected −6.04 (t=−2.14) |
| `r23base.bat` | 2023R baseline (for blend weight selection) | reference data |
| `rm.bat` `rm2.bat` `rm3.sh` | refit multiplier 1.5 / 1.7 / 2.0 | rejected (+0.60 over 18 seeds) |
| `bias.sh` | One-season-ahead bias series (BI2021–2024) | basis for the SHIFT decision |
| `div.sh` `div2.sh` | Diversity members re-validated (LightGBM, subspace, cells) | LightGBM and subspace rejected |
| `fleague.bat` | Keep vs drop old-regime F | structural artefact — no usable conclusion |
| `v17.bat` | Rebuild with old-regime F dropped | rejected −53.6 |
| `surf_a.bat` `surf_b.sh` | Unseen-surface axis re-judging, arms A and B | aborted (the `season` drop was BANNED) |
| `chain.sh` `chain2.sh` `chain3.sh` | A100 queue chaining (rm3 → surface → cell zoo) | only chain3 was valid |
| `surf4070.bat` | Four surface axes (lr, depth, te-k, std-k) | all rejected |
| `stdk.bat` | std-k 10/20/30/60 fine sweep | held (t < 2.4) |
| `k40rep.bat` | **Reproduce std-k 40 on the same machine** | ★ the run that caught the cross-machine comparison error |
| `tek.bat` | Per-axis te-k (reliability-based) | rejected +0.58 / +2.26 |
| `a100base.sh` | A100-only baseline (AB_base 883.41) | reference data — the basis for every A100 verdict |
| `cellctx.sh` | Cell × ball-count context (32 / 100 cells) | rejected −5.13 / −74.71 |
| `night2.bat` | F-only model plus extra refit seeds | F-only crashes (not enough data) |
| `night3.bat` | Cell seeds 6→8, te-k 100 reproduction | zero gain / rejected |
| `twostrike.bat` | Two-strike segment model and routing | rejected −13.61 |
| `distill.sh` | Distillation, first attempt | **leak** — the teacher saw 2024 |
| `distill2.sh` | Distillation re-run (after fixing the refit loss) | leak unchanged — discarded |
| `v19.sh` | Build a submission with std-k 40 | dequeued (the evidence was a machine artefact) |
| `pre-0807/` | 19 runners from before 08-07 | see `docs/EXPERIMENTS_LOG.md` |

Everything from 08-08 onward is in `LEDGER.tsv` with its full command, so this table
is not extended past that date.

# AGENTS.md — single entry point for every agent

> **Read this file before doing anything.** Claude and Codex share this working tree.
> `CLAUDE.md` is the same document (see the last section for why it is a stub).
>
> Current state → [EXPERIMENT.md](EXPERIMENT.md) · Baton → [HANDOFF.md](HANDOFF.md)
> Closed questions → [docs/SETTLED.md](docs/SETTLED.md) · Run log → `LEDGER.tsv`
> Long-form history → [docs/EXPERIMENTS_LOG.md](docs/EXPERIMENTS_LOG.md)

---

# 1. Competition

**LG Aimers 9th — KBO pitch control-success probability** (Dacon 236743).
Predict a 0–1 probability per pitch **using only information available before the
pitch is thrown**.

- Page: https://dacon.io/competitions/official/236743
- Deadline **2026-09-01** (Phase 2 submissions) → 09-07 code/PPT → 09-14 finalists
- Target: `control_success` (1 = success, 0 = failure)
- Failure is defined as ① middle of the strike zone ② outside the zone
  ③ opposite side from the catcher's call

## Metric — Brier Skill Score

```
brier          = mean((pred - y)^2)
baseline_brier = r * (1 - r)          # r = true success rate (hidden constant)
score          = max(0, 100000 * (1 - brier / baseline_brier))     # higher is better
```

- Calibration drives the score. **Never round to 0/1.**
- **Each evaluation row must be predicted independently.** No post-processing that
  uses the test distribution.
- **Public = Private = 100% of the test set.** There is no hidden split, so plain
  generalisation matters more than public-LB overfitting.
- Certification threshold: Public ≥ 549.51 (organiser's baseline run).

## Standing

| | |
|---|---|
| Current LB | **1,101.802** — `submissions/v11_pb_posix_0809.zip` |
| Rank-1 | 1,198.02 |
| Goal | reach the 1,120s by legal means |

## Data

| File | Shape | Notes |
|---|---|---|
| `data/train.csv` | 1,475,092 × 49 | 2019–2024, includes target |
| `data/test.csv` | 5 × 48 | **sample only.** The server injects the real 245,789 rows (2025) |
| `data/trackman_history.csv` | 1,793,078 × 30 | 2019–2024 pitch log. Cannot be joined 1:1 |
| `data/sample_submission.csv` | | row_id order/columns the submission must match |

**`test.csv` columns are the contract**: 47 features besides `row_id`. Using a
train-only column as a feature breaks inference. Full spec:
[data/data_description.md](data/data_description.md).

Column groups: game context (`season`, `game_month`, `game_dayofweek`, `inning`,
`top_bottom`, `game_type`) · count/score (`balls_before`, `strikes_before`,
`outs_before`, `run_*`, `score_diff_*`) · runners/leverage (`runner_on_*`,
`num_runners_on`, `base_state`, `*_win_expectancy`, `li`) · players
(`pitcher_id`, `batter_id`, `*_hand`, `*_team_id`) · **19 `asof_*` history
features** (cumulative rates up to the pitch — officially provided, legal to use;
zero-sample rows are missing → cold start).

## Disqualifying information

- Anything determined after the current pitch (actual location, call, pitch type,
  Trackman measurements)
- 2025 Trackman data
- **Any feature built from other rows of `test.csv`** — cumulative, frequency,
  distribution, rolling, target encoding, post-hoc rescaling

## Rules

| Item | Limit |
|---|---|
| Language | Python only |
| External data | **Forbidden** |
| Pretrained weights | Public weights, MIT/Apache-2.0-class licence, bundled in the zip |
| Remote APIs | Forbidden |
| Submissions | 5 per day |
| Reproducibility | Must reproduce locally; `random_state` fixed |

---

# 2. Evaluation server

| Item | Spec |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | NVIDIA L4 (22.4 GiB), CUDA 12.8 |
| CPU / RAM | 6 vCPU / 28 GB |
| Python | **3.11.15** |
| Internet | **Blocked** — no runtime downloads |
| Inference budget | ≤ 10 min for 245,789 rows |
| Install budget | ≤ 10 min |
| Zip size | ≤ 10 GB (≤ 32 GB unpacked) |

Preinstalled and **version-pinned**:

```
torch==2.7.1+cu128  pandas==2.0.3  numpy==1.26.4  scipy==1.15.3
scikit-learn==1.8.0  joblib==1.5.3  threadpoolctl==3.6.0  narwhals==2.21.2
transformers==4.46.3  accelerate==1.9.0  sentencepiece  regex  tqdm  loguru  pyyaml  rich
```

**Strategy: use only preinstalled packages and keep `requirements.txt` empty** —
zero install time, zero install risk. Never re-declare a preinstalled version in
`requirements.txt`; that is how version conflicts happen. sklearn version parity
matters most, because the model pkl files must load.

## Submission zip

```
submit.zip
├── model/            trained artefacts
├── script.py         inference entry point — the server runs this
└── requirements.txt  (kept empty)
```

- `script.py` must sit at the **zip root**; a wrapper folder is an install error.
- The server unpacks, runs `script.py`, and grades `./output/submission.csv`.
  `data/` is read-only and holds the real evaluation rows.
- **Error classes**: install errors (bad zip layout, failed install) do *not* consume
  a submission. Runtime errors in `script.py` **do**. Always smoke-test locally.
- Any feature built outside the pipeline must be reproduced inside `script.py`.

## Model feasibility (measured, `src/bench_inference.py`)

Inference is **not** the bottleneck: RF(100 trees) predicts 245,789 rows in 0.4 s on
6 cores — 0.1% of budget. The real constraints are install time, no internet, and
package-version conflicts.

| Family | Verdict |
|---|---|
| sklearn (RF, HistGB) | ✅ preinstalled |
| LightGBM / XGBoost / CatBoost | ✅ needs a `requirements.txt` line (1–2 min wheel) |
| Torch NN (MLP, TabM, FT-Transformer) | ✅ torch preinstalled; bundle weights in the zip |
| Large seed × fold ensembles | ✅ budget is ample; watch the 10 GB zip limit |
| AutoGluon | ⚠️ dependency storm → version conflict / install timeout |
| TabPFN | ⚠️ 1.47M rows exceeds context; subsample ensembles only |
| Remote APIs | ❌ forbidden |

---

# 3. Machines

Training runs on remote boxes. **The laptop is for analysis, judgement, and
submission packaging only** (thermals).

| Alias | Hardware | Access | Project | Python |
|---|---|---|---|---|
| `desktop-5070` | **RTX 5070 Ti 16 GB** (Blackwell) | `ssh desktop-5070` (Tailscale 100.121.174.83, user `jinsan`) | `C:\aimers` | `C:\aimers\.conda\python.exe` (3.11.15) |
| `hsu-server` | **A100 40 GB** · 80 vCPU · 503 GB RAM | `ssh hsu-server` (ProxyJump via `desktop-4070`, port 8822) | `~/aimers` | `~/venv451/bin/python` (3.10.12) |
| `desktop-4070` | RTX 4070 Ti SUPER 16 GB | `ssh desktop-4070` | `C:\aimers` | `.venv\Scripts\python.exe` (3.11.15) |
| laptop (here) | RTX 5060 8 GB | — | `C:\aimers` | `uv run python` (3.11.15) |

Aliases already live in `~/.ssh/config`. Tailscale must be up.
**`hsu-server` hops through `desktop-4070`** — if the 4070 is off, the A100 is
unreachable too.

## `desktop-5070` — joined 2026-08-12. Its environment matches the eval server exactly

```
python 3.11.15 · torch 2.7.1+cu128 (CUDA OK) · catboost 1.2.10
pandas 2.0.3 · numpy 1.26.4 · sklearn 1.8.0     ← identical to the eval server
```

The A100 runs pandas 2.3.3 / numpy 2.2.6 / sklearn 1.7.2, which does **not** match.
**Bake submission pkl files on the 5070.** Python is not on PATH there — call the
absolute path. Shell is `cmd.exe`.

## Shells differ per machine (this has cost us repeatedly)

- `hsu-server` → **bash**. Normal.
- `desktop-4070`, `desktop-5070` → **`cmd.exe`**, not PowerShell and not bash.
  `;` is not a separator; use `&` instead of `&&`. No `head`/`tail`/`grep`
  (use `findstr`). Korean output is CP949 and will mojibake when read as UTF-8 —
  **judge from `.npz` artefacts, never from console text.**
- laptop → Git Bash (POSIX), plus PowerShell.

## Code sync

```bash
scp src/*.py   hsu-server:~/aimers/src/
scp src/*.py   desktop-4070:C:/aimers/src/      # note the slash direction
scp src/*.py   desktop-5070:C:/aimers/src/
scp tools/*.py hsu-server:~/aimers/tools/
```

The laptop is always the source of truth. Never edit on a remote.

## Long-running jobs (must survive ssh disconnect)

```bash
# A100 — reparents to PID 1
ssh hsu-server "cd ~/aimers && setsid nohup bash X.sh > out/X.log 2>&1 < /dev/null & disown"

# 4070 — this launcher only. See rule 15.
bash tools/run4070.sh <name> 'C:\aimers\X.bat' 'C:\aimers\out\X.log'
```

On the Windows boxes `setsid nohup`, `start /b`, and `Start-Process` all die with
the ssh session. WSL `tmux` breaks Windows exe interop
(`UtilAcceptVsock accept4 failed 110`). `schtasks` is the only survivor on the
4070, and `tools/run4070.sh` wraps it safely (host is the 4th argument).

> ⚠️ **`schtasks` does not work on `desktop-5070`** (2026-08-12). The task creates,
> `/run` reports success, and `Last Result` is `267011` — but nothing executes: no
> GPU load, no redirect target created, not even the outer log file. The same batch
> runs fine when invoked directly. Until that is diagnosed, launch long 5070 jobs as
> a **backgrounded direct ssh call** and keep the laptop awake:
>
> ```bash
> ssh -o ServerAliveInterval=30 desktop-5070 'cmd /c C:\aimers\scripts\X.bat'
> ```
>
> Each arm inside the batch must redirect its own stdout to `out\<tag>.log` and write
> `out\<tag>.exit`, so a dropped ssh session is detectable rather than silent.

## Environment drift

| | pandas | numpy | sklearn |
|---|---|---|---|
| eval server · laptop · 4070 · 5070 | 2.0.3 | 1.26.4 | 1.8.0 |
| **A100** | **2.3.3** | **2.2.6** | **1.7.2** |

Predictions were verified **identical to 12 decimals** across machines
(`tools/env_check.py`) — the pipeline avoids version-sensitive ops. Re-verify if a
new library enters.

**But never compare an A100 result to a 4070 result** — the same config and the
same seeds diverged by 11 points (rule 5).

## Merging the ledger

`LEDGER.tsv` accumulates on whichever machine trained. Collect before judging:

```bash
bash tools/ledger_sync.sh
```

---

# 4. Non-negotiables — each one has a price tag we already paid

## Before running

1. **Pass `python tools/precheck.py <all flags>`.** Exit code 2 means forbidden.
   Whole scripts work too: `--file some.sh`.
   → On 08-07 we re-queued the `season` drop that E08 had already closed at −580.
2. **Read `docs/SETTLED.md`.** Closed questions carry the number *and the mechanism*.
3. Check GPU occupancy. **One job per machine.**

## Measurement

4. **The judging surface is `--val-season S-1 --test-season S`** (deployment shape:
   refit on ≤S-1, evaluate untouched S). Self-validated holdout is reference only —
   the cell member's contribution flipped +2.8 vs −18.0 between the two surfaces.
5. **Build the comparison arm on the same machine.** Identical config and identical
   6 seeds diverged **11 points** between A100 and 4070. More seeds cannot suppress
   it. `tools/surf_report.py` cross-checks hostnames in `LEDGER.tsv` and invalidates
   mixed comparisons automatically.
6. **Adopt at t ≥ 2.4; reject when the 95% upper bound < +3.** n ≥ 6 seeds, paired.
   → refit-multiplier 2.0 looked like +2.44 (t=1.91) at 6 seeds; at 18 it was +0.60.
7. **Never read a segment's BSS as loss.** A base rate far from 0.5 depresses BSS.
   The F league looked like −301 by BSS but its **MSE was lower** than R
   (.24690 vs .24770).
8. **Verify the training set, not just the flags.** Run
   `python tools/member_fingerprint.py <tags>` — `fpipe['priors']` is computed from
   the training rows, so it fingerprints the data. Exit code 2 means the members
   were trained on different data and their blend weight is meaningless.
   → 08-12: a `--drop-f-pre 2022` copied between surfaces produced a "v14f
   reproduction" that was 52 points weaker.

## Post-processing and submission

9. **A post-hoc constant is valid only if the run that measured it used exactly the
   submission's training data and flags.** → v16, −6.15.
10. **Never transplant a conclusion across structures.** → v17, −53.6.
11. **One change per submission.** → v16 moved SHIFT and SLOPE together and needed
    back-solving.
12. **`python tools/audit_rowindep.py <script>` must pass before submitting.**
    Predictions must be bit-identical under any batching. Dacon requires row
    independence explicitly.
13. **Never tune a constant against LB feedback.** Public = Private, so that is
    fitting the answer key, and Phase 2 code review will see an unjustified constant.

## Code

14. **Never filter a log through grep.** `2>&1 | grep -E "^\[cat|..."` swallowed a
    traceback whole and a dead run read as "complete" (first distillation attempt).
    Write the full log to a file and check `Error|Traceback` yourself.
15. Keep the existing structure and style. Never change several things at once.
16. **Long 4070 jobs go through `bash tools/run4070.sh` only.** Using a nearby dummy
    time in `schtasks` makes every task re-fire at that time — nine fired at once and
    split the GPU nine ways.
17. **`src/failmode.py` is train-only.** The same algebra on test rows recovers 2025
    targets (verified 96.79%). It produces supervision labels, never features;
    `fpipe.transform` does not import it; it is **never** in the submission zip.
18. **No runners in the project root.** They go in `scripts/`. The rule existed and
    was broken twice (30 files on 08-08, 51 on 08-12), so
    `tools/agent_sync.sh end` now exits 2 when it finds one.

---

# 5. Experiment record

**No manual step.** `train_gbdt2.py` and `train_tabdecoder.py` append one row per
seed to `LEDGER.tsv`:

```
date · host · tag · model · val season · test season · seed · best_iter · val BSS · test BSS · full command
```

What an agent still writes by hand:

- When an axis **closes** → one line in `docs/SETTLED.md` (verdict · number ·
  **mechanism**)
- After a submission → LB score and prediction error in `docs/EXPERIMENTS_LOG.md`
- When work ends → `HANDOFF.md`

---

# 6. Agent roles

## Claude — decides what gets measured

- Loss decomposition, residual analysis, hypotheses, experiment priority
- **Independently reviews** Codex's first-pass analysis and issues the verdict
  (t value, surface, machine parity, training-set parity)
- Plans the next experiment after an adopt/reject
- Decides whether to submit; derives post-processing constants
- Screens for rule violations (row independence, external data, LB probing)

## Codex — implements the specified experiment exactly

- Experiment scripts, flag wiring, error fixes
- Tooling (`tools/*.py`), refactors, tests
- Per-machine execution, monitoring, artefact recovery
- Submission packaging and smoke tests
- **First-pass analysis** after verifying integrity (paired statistics, surface /
  machine / training-set parity, mechanism, risks, follow-up candidates)

**Boundary**: Codex labels its analysis `Codex first pass (provisional)`, separates
numbers from inference, and recommends — but does not add a `docs/SETTLED.md`
verdict or decide a submission on that basis alone. Claude reviews the code, the
ledger, and the rules before finalising **the verdict and the next plan**, and
writes *what to measure and why* rather than editing implementation details.

---

# 7. Source of truth

1. The actual code
2. `LEDGER.tsv` — the commands that really ran. Trust it over memory.
3. `docs/SETTLED.md`
4. `docs/EXPERIMENTS_LOG.md`
5. `EXPERIMENT.md` / `HANDOFF.md`

When a document disagrees with the code, **check the code and fix the document.**

---

# 8. Handoff — two agents, one working tree

Repository: `github.com/jinsan02/lg-aimers-9-hackathon` (**private**).
`data/` is git-ignored; redistribution is forbidden.

## Starting

```bash
bash tools/agent_sync.sh start codex     # or claude
```

> ⚠️ **In Windows apps (Codex desktop and friends) use the wrapper, not bare `bash`:**
>
> ```cmd
> tools\agent_sync.cmd start codex
> tools\agent_sync.cmd end   codex "what I did"
> ```
>
> Bitten twice on 08-08: ① bare `bash` resolved to WSL and died with
> `E_ACCESSDENIED`; ② even calling Git Bash directly, a non-login shell inherits the
> Windows PATH and loses `dirname`, `grep`, `head`, `tr`. The wrapper opens
> `bash.exe --login` and blocks both.
>
> **Ignoring a failure here** leaves you with no fetch and no ledger merge, and you
> will overwrite someone else's commits. If the wrapper fails, stop and diagnose.

It does: fast-forward-only pull · **halt if anything is uncommitted** · merge
`LEDGER.tsv` across machines · print the HANDOFF status · list remote GPU jobs.

A halt means **the previous session never called `end`.** Investigate; do not
overwrite.

Then read: `AGENTS.md` → `EXPERIMENT.md` → `HANDOFF.md` → check `docs/SETTLED.md`
for your axis → pass `tools/precheck.py`.

## Finishing

```bash
bash tools/agent_sync.sh end codex "DT_self/DT_seq re-validation results"
```

It does: **reject root-level runners (exit 2)** · merge the ledger · `git add -A` ·
commit · push · print the handoff status.

By hand:

1. **Update `Current Agent` / `Next Agent` / `Status` in `HANDOFF.md`** — that is the baton
2. Codex fills in `Codex first pass (provisional)` and `Review requested from Claude`
   (facts/numbers · interpretation/mechanism · risks · recommendation · the questions
   Claude must decide)
3. When Claude closes an axis → one line in `docs/SETTLED.md` (verdict · number · mechanism)
4. After a submission → LB and prediction error in `docs/EXPERIMENTS_LOG.md`

## On conflict

If `git pull --ff-only` fails, **do not auto-merge.** Two agents edited the same file
differently and a human should look. Dump `git log --oneline origin/main..HEAD` and
`git diff origin/main` into HANDOFF and stop.

## Territory

- Prefix commits with `[claude]` / `[codex]` (`agent_sync.sh` does it)
- **Never refactor the other agent's code without a reason recorded in HANDOFF**
- `docs/SETTLED.md` is **append-only.** Removing a line requires a measurement that
  overturns it

---

# 9. Repository layout

```text
AGENTS.md                this file — the single entry point
CLAUDE.md                stub pointing here (see below)
EXPERIMENT.md            current state board
HANDOFF.md               agent-to-agent baton
LEDGER.tsv               ★ automatic. one row per seed
LEVERS.md, LEVERS_NEXT.md   lever backlog

docs/SETTLED.md          ★ closed questions — precheck.py reads this
docs/EXPERIMENTS_LOG.md  long-form history
docs/experiment_guide.md per-machine how-to

src/
├── train_gbdt2.py       main trainer (CatBoost/LGB/XGB, every flag)
├── train_tabdecoder.py  column-token causal decoder
├── fpipe.py             ★ feature pipeline — fit and transform live in one file
├── failmode.py          failure-mode label recovery (⚠ train-only, never shipped)
├── teacher.py           distillation teacher (axis closed)
└── script_blend_v*.py   submission inference
tools/                   analysis and verification (precheck / surf_report / audit_*)
scripts/                 live runners only; finished ones in scripts/archive/
data/  model/  out/  submissions/     (all git-ignored)
```

## Running things

```bash
uv sync
```

```bash
uv run python src/train_gbdt2.py --model cat --tag x --l2 3 --feat-v2
```

Native venv is `C:\aimers\.venv` (uv-managed). Set `$env:PYTHONPATH="src"` first for
scripts that need it. Local pins match the eval server exactly.

## Why `CLAUDE.md` is a stub, not a symlink

Claude Code auto-loads `CLAUDE.md`, and the intent is for it to *be* this file.
A real symlink needs `core.symlinks=true` plus Windows Developer Mode; on this
laptop `mklink` is denied and git checks a symlink out as a plain text file
containing the target path — which would silently feed Claude the string
`AGENTS.md` instead of these rules.

Once Developer Mode is on (Settings → Privacy & security → For developers), convert
it with:

```bash
git config core.symlinks true && rm CLAUDE.md && ln -s AGENTS.md CLAUDE.md && git add -A CLAUDE.md
```

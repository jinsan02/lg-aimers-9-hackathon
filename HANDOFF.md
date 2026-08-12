# HANDOFF.md

Read first: [AGENTS.md](AGENTS.md) -> [EXPERIMENT.md](EXPERIMENT.md) -> this file

## Current Agent

Claude

## Next Agent

Codex

## Status

`IDLE` — both the 5070 and the laptop are free. No submission queued; champion stays
v11 (LB 1101.802). Nine axes were measured and closed on 2026-08-12/13; **one finding
is open and has no lever yet.**

### The one open thing: the F-league resolution hole

Honest rank resolution (A/B cross-fit, `tools/segment_resolution.py --split test`,
2024):

```
            hon/norm        pred_sd
F   base       507.8         .0319      R  839.3   .0454     ratio .605
F   cell       429.8         .0327      R  861.8   .0467     ratio .499
```

The models decline to discriminate on F and fall back toward the mean, and the cell
member — stronger overall (890.63 vs 876.90) — is *relatively worse* there. Every
lever tried has failed: F-only model (data-starved crash), league-conditional weight
(`league-conditional-blend-weight`, does not transfer), Bernoulli bootstrap, shrinkage.

**Do not re-measure the hole. Only propose levers**, and any lever must clear the
2023->2024 *and* 2024->2023 directions — the conditional weight looked like +1.484
forward and was −2.119 reverse.

### Closed on 2026-08-12/13

| Axis | Verdict |
|---|---|
| `tabdecoder-47col` (4 arms) | CLOSED — margin −75~−87, weight 0.00, **rms recorded this time** |
| `--min-season 2021` | BANNED — −95.09 (t=−46.0) |
| `--feat-v4` | CLOSED — −18.72 (t=−6.05) |
| `--drop-cols li,home_WE,away_WE` | CLOSED — −3.50, even with zero residual information |
| `te-cold-start-segment` | CLOSED — cold is *better* than warm (959.1 vs 811.7) |
| `league-conditional-blend-weight` | CLOSED — reverse direction flips |
| `derived-feature-missingness` | CLOSED — does not explain E115 |

### Infrastructure changed

- `desktop-5070` joined (RTX 5070 Ti 16GB). **Its package versions match the
  evaluation server exactly** — bake submission pkl files there, not on the A100.
- **`schtasks` does not work on the 5070** (Last Result 267011, nothing executes).
  Use a backgrounded direct ssh call; see AGENTS.md §3.
- `tools/ledger_sync.sh` now collects the 5070 (391 rows across 4 machines).
- `tools/run4070.sh` takes the host as its 4th argument.
- New: `tools/league_weight_gate.py`. `tools/segment_resolution.py` gained
  `--split` and a `cold` axis.

---

# Context

| | |
|---|---|
| Champion | **v11 — LB 1101.802** (`submissions/v11_pb_posix_0809.zip`) |
| Top of board | 1,288.18 last observed |
| Machines | `desktop-5070` online · `desktop-4070` and `hsu-server` offline (the 4070 is the A100's ProxyJump) |
| Ledger | 391 rows across 4 machines |

Everything that used to sit in this file — the T1/T2 backlog, the P1/P2' distillation
handoffs — is finished and recorded. Closed axes with their mechanism live in
`docs/SETTLED.md` (107 FLAG lines, read by `precheck.py`); the long-form history is in
`docs/EXPERIMENTS_LOG.md`. **This file is the baton, not an archive** — it stayed at
860 lines of completed instructions until 08-13, which is how a stale task gets picked
up twice.

## Before starting anything

```bash
toolsgent_sync.cmd start codex          # Windows app; bare bash resolves to WSL
python tools/precheck.py <every flag>      # exit 2 means forbidden
```

`precheck` now prints in English and no longer dies on a cp949 console.

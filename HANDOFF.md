# HANDOFF.md

Read first: [AGENTS.md](AGENTS.md) -> [EXPERIMENT.md](EXPERIMENT.md) -> this file

## Current Agent

Codex, handed over by Claude 2026-08-16 at `f9672e9`.

## Next Agent

**One GPU job is live**: `AimersBND` on `DESKTOP-053T952` (5070), registered
under SYSTEM so it survives ssh disconnect and the laptop being off. 24 fits,
roughly 2.5-3 h from 2026-08-16. It produces **artifacts, not a candidate** —
see `docs/BND_PREREGISTRATION_20260816.md` for the spending contract, which is
fixed in advance and binding.

```
watch:  ssh desktop-5070 "type C:\aimers\out\bnd.log"     # ==== BND COMPLETE ====
        ssh desktop-5070 "schtasks /query /tn AimersBND /v /fo list" | grep -i "last result"
        267009 = running, 0 = finished
clean:  ssh desktop-5070 "schtasks /delete /tn AimersBND /f"
then:   bash tools/ledger_sync.sh    # the trainer writes its row on the runner
```

Nothing is queued for submission. Champion `submissions/gskdep_0816.zip`, LB
**1111.3713632162**, is frozen and unchanged by everything below.

## Status

**2026-08-16 GPU queue: four axes measured, four DROPs, 49 fits, champion
untouched.** Every one was pre-registered before any fit, on a deciding surface
named in advance, with fresh controls on both arms and a large negative
pre-accepted.

| Axis | Surface | Core delta | Verdict |
|---|---|---|---|
| `--drop-f-pre 2022` (FPRE) | submission, n=6 | **−61.008** SE 1.361, t −44.84, 0/6 | DROP |
| `--fm-coarse failure` (FMCOARSE) | judging, n=3 | **−18.581**, 0/3 | DROP |
| `--std-k 40` (SK40) | submission, n=6 | **−7.780** SE 1.600, t −4.86, 0/6 | DROP |
| `--feat-id-cohort` (ICOH) | submission, n=6 | **−2.780** SE 1.856, 95% upper +1.99 | DROP |

Full verdicts and mechanisms: `docs/SETTLED.md`, FLAGs `--drop-f-pre 2022 on the
submission surface`, `FMCOARSE collapse the failure block`, `--std-k 40 (SK40)`,
`--feat-id-cohort (ICOH)`.

**Four results that change what should be proposed next.**

**1. The shared tables are load-bearing across leagues.** FPRE removed 105,308
pre-2022 F rows whose target mean shifts .7087 → .4729 while R stays flat. F lost
−134.49, which is unsurprising. **R lost −51.23**, and R has no label-regime
problem at all. Row filters run before `fpipe.fit`, so they truncate the TE /
season-standardisation / asof-anchor tables every league draws on. This is the
same mechanism recorded for `--min-season 2021` (−95.09), now demonstrated rather
than inferred. **Treat any row-dropping proposal as a table-truncation proposal.**

**2. The cell-supervision family is closed in every direction anyone has
pushed.** Reweight (P3-A/B/C/C2) FAIL, reparametrise (FM_MULTILABEL/V2) FAIL,
project (SUCCESS_AUX_GRADIENT) nothing survives, **delete (FMCOARSE) −18.6**. The
motivation for FMCOARSE remains factually true — 7.4× of the mutual information
sits in distinctions the aggregated Brier cannot see, and those distinctions
conflict with the primary gradient at up to a 100% minibatch rate — and the
supervision is load-bearing anyway. The candidate also early-stops at ~2350
against ~2980 and fits in 97 s against 236 s, which is what less structure to
learn looks like. **Do not reopen without a mechanism that is none of those four
operations.**

**3. SK40 closes the downward `--std-k` direction, and only the downward one.**
Pre-committed in the preregistration: no k20, no k10, no k30, no sweep. The
judging-surface positives (k20 +3.21 t=2.01) are recorded as surface-transfer
failure, the same shape as `--te-halflife 2`. **`--std-k 120` is upward and is
still an open rule-5 violation** (recorded upper bound +3.86) — it is not covered
by this closure.

**4. The single-seed noise floor is now measured on the deciding surface.** The
base arm's paired per-seed standard deviation is `3.126 × √6 = 7.66` BSS points.
That places ICOH's two surviving headline "transfers", **+13.93 and +8.13, at 1.8
and 1.1 sd** — inside the noise. Any axis still resting on single-seed ledger rows
should be read against 7.7, not against zero.

### Method notes worth carrying

- **A rho pre-screen must quote a matched null.** A random direction in the
  champion's own feature space already correlates with its out-of-time residual
  at |rho| median 0.0035-0.0058. The +3 bar is rho = 0.0055, **at or below that
  median**. Rho below the null median is not weak evidence, it is no evidence.
  Details in `docs/SETTLED.md` FLAG `rho pre-screen needs a random-direction null`.
- **The leaderboard cannot resolve below about +2 to +3** (LB delta sampling SE
  ±1.06). Do not read a small LB move as confirmation or refutation.
- **One control pair may serve two experiments** when it is the same command in
  the same session on the same host. The SK40/ICOH chain did this and saved 8
  identical fits. It is not a shortcut around the fresh-control contract.
- **`tools/precheck.py` has an argument-dispatch hole.** Anything that is not
  exactly `--file` falls through to the flag checker, which prints plausible
  `[OK]`/`[WARN]` lines and **exits 0 without opening the file**. Reproduced with
  `--bat`. Use `python tools/precheck.py --file <script>` and confirm the
  `N command(s) checked` line is present. A fix is queued but not applied.

### Queue after BND, in order

1. **Re-judge the 7 downgraded axes** on three clean boundaries once BND lands —
   under the fixed contract, not as a candidate pool.
2. **Tier 1 remainder**: `--feat-window + cells` (A100 6-seed +0.51 vs
   single-seed cross-machine −5.85), `--baseline-col` (n=1), `xgb-121-blend`
   (contaminated legs).
3. **Remaining rule-5 violations**: `--std-k 120` (upper +3.86),
   `--drop-cols li,wexp` (+4.06), `--skill-neutral-mode const` (+3.25).
4. **Tier 3**: of 20 wired-but-never-run flags at ledger=0, only `--grow`
   (Depthwise/Lossguide) and `--ptype` have no closed analogue. `--weight-mode`
   is already dead via the cross-season result.

### Review requested from Codex

1. Confirm or contest the four DROPs against the stored arrays. All 36 chain
   members are in `out/` locally and on the 5070; `row_id` and target are
   elementwise identical across all of them, verified before any BSS was read.
2. Confirm that the FMCOARSE result is read correctly as closing the family
   rather than as a bug — the taxonomy behaved as designed (4 cells, 3 success
   cells, recovery 99.85%) and the pre-registered `rms` kill-check was read
   first at 0.009634 against a 0.002 threshold, so the arms were genuinely
   distinct.
3. Decide nothing about `--std-k 120` from SK40. The closure is directional and
   the preregistration says so.
4. Do not infer any champion constant, league weight or blend coefficient from
   any number above. None of these ran on the submission path.

---

## Earlier: 2026-08-16 structural research — two axes closed on CPU, zero GPU spent.

`TM_DIST` trackman distributional arsenal embedding -- **FAIL**. The embedding is
genuinely new (champion's 112 numeric features reconstruct it at CV R^2 <= 0.18)
and genuinely persistent (consecutive-season CCA 0.95/0.93/0.89 at all five
transitions), and it is still worth an honest **+0.43** ceiling against a +3 bar.
Non-linear frozen transfer collapses from rho +0.06265 in-sample to **+0.00222**,
and **-0.00003** when the source and target boundaries are swapped. The useful
separation: what a pitcher throws is orthogonal to where this model is wrong.
Trackman distribution axis CLOSED -- no PCA dim change, no clustering sweep, no
feature-subset rescue.

`CROSS_SEASON_ROBUST_RISK` season-loss dispersion penalty -- **FAIL**. Premise
confirmed (out-of-fold per-season BSS spans 1005.61-2577.35, not monotone in
time distance, not a denominator artifact), mechanism refuted: upweighting the
worst season gives **-84.14** and the opposite weighting gives **-80.04**.
**Season reweighting itself is the cost, not its direction** -- which explains
the whole recency family (`W` LB -41.5, season decay, te-halflife,
`--min-season` -95.09) as one statement. CLOSED.

The CPU surrogate used for the second was admitted only after reproducing a
known answer: recency weighting at halflife 2 scored -48.97 against the real
`W` result of LB -41.5.

`SUCCESS_AUX_GRADIENT` (candidate C) -- **FAIL**. Premise confirmed: every
auxiliary gradient conflicts with the primary on both boundaries, sign stable
3/3, `cos(success, reverse)` -0.9868/-0.1902 at a **100%/100%** minibatch
conflict rate. But nothing survives the surgery -- `reverse` is nearly an
anti-parallel duplicate (16% survives projection), `ball` is nearly orthogonal
so it does not move the success loss at first order, and the projected
directions rotate between boundaries (cos +0.31 / -0.10 / +0.19). CLOSED.

`MULTIVARIATE_LATENT_STATE` (candidate D) -- **FAIL**. 3-d state persists only
modestly (CCA 0.622-0.762 top component, third component ~0.18 = noise), is
substantially reconstructible from the champion's own features (in-sample
R^2 0.26-0.73), covers 70-74% of rows, and its frozen transfer lands at the
**75.5th / 64.2nd percentile** of a matched null. CLOSED.

**The pre-registered chain A-D is exhausted: four axes, zero GPU hours, champion
untouched, nothing submitted.**

**The most reusable output of the session is a method correction.** A direction
drawn at random in the champion's own feature space already correlates with its
out-of-time residual at |rho| median 0.0035-0.0058 (p90 0.0084-0.0144, n=100k,
null SE 0.00316). **The +3 bar is rho = 0.0055, which is at or below that
median**, so a rho pre-screen must quote a matched null with the same
fit-and-freeze protocol and the same number of free parameters. Rho below the
null median is not weak evidence, it is no evidence. This strengthens the
TM_DIST and `--te pchh` FAILs and leaves every adopted result untouched, since
adoption always ran through paired 6-seed deltas in `tools/judge.py`.

No next axis is pre-registered. The four structural candidates the user supplied
are spent, and the remaining items in `docs/NEXT_CANDIDATES_20260816.md` are
representation-reuse rather than new information.


`CHAMPION VERIFIED — NEXT PLAN AWAITING APPROVAL`. Champion is
`submissions/gskdep_0816.zip`, LB **1111.3713632162**, SHA256
`87617a131498b1121c100668b965b57443abbfdbe8bef5c39b28976d4f29e70d`, rank **#48**.

**Champion safety: SAFE TO KEEP, verified against the ZIP rather than the
document.** 22 members; base 8 packs byte-identical (sha256) to
`b1s8_20260813.zip`; cell 6 packs share `fit_rowid_sha d69792676c50e1cf`,
`features_sha a109f03b48b72da7`, 123 features; `set(cell)-set(base)` is exactly
`['skill_hat','skill_hat_vs_std']`. `failmode.py` absent and not inlined (zero
`.diff(` anywhere in the package; every cross-row aggregation is confined to
fit-time builders and unreachable from `fpipe.transform`). Strong subset
independence proven empirically: half / single-pitcher / scattered / single-row
and order-reversal all **0.000e+00**. Only `./data/test.csv`,
`./data/sample_submission.csv` and `model/matchup_constants_2024.npz` are read.
Five documentation-vs-zip discrepancies were found and the document was
corrected; none of them affects the shipped arithmetic.

**GSK interpretation: the "13% reproduced / transfer-attenuated" reading is
retracted.** `+3.108` and `+19.517` were judging-surface numbers. The
surface-matched estimate was `+1.104, SE 1.086` (`GENERAL_SKILL_ADD`), and an
independent recomputation from the stored val2024 arrays gives **+1.342**. The
LB delta's own sampling SE is **±1.06**, so the observed `+0.391` is **0.9 SE**
away — consistent, not attenuated. Consequence: **the leaderboard cannot resolve
anything below about +2 to +3.**

**E-LB2 cancelled.** The offline `_W_CELL` curve peaks at `w* ≈ .575-.60` and is
worth `+0.07` over the shipped `.55`; signal-to-noise against the ±1.06 LB SE is
0.066. `_W_CELL` stays 0.55; no coefficient is transplanted.

**Leaderboard reality (read directly 2026-08-16).** Rank-1 **1,240.63302**,
rank-2 1,176.54904, rank-3 1,170.70438, rank-10 1,157.32319, rank-15
1,144.20504. Rank-1 stands 64 points clear of rank-2 while ranks 2-15 span 32
points; closing 129 points needs `rho = 0.0360`, larger than knowing the true
pitch type of every pitch (`rho = 0.0350`). Rank-1 is excluded as a target on
arithmetic. Planning band is **ranks 2-15 = +33 to +65**.

Full verdicts: `docs/SETTLED.md` FLAGs `general-skill cell-only (GSK)
deployment`, `lb-gap-arithmetic`, `E-LB2 blend-weight LB probe`.

### Review requested from Claude

1. Confirm the numerical interpretation: official GSK increment +0.390733 is
   positive but materially below both historical estimates; decide the final
   SETTLED wording (likely KEEP-WEAK / transfer attenuation, not a new sweep).
2. Confirm `gskdep_0816.zip` as the rollback champion and the exact recipe in
   `docs/CHAMPION_GSK_RECIPE_20260816.md`.
3. Decide the next submission plan. The existing E-LB2 `.45/.65` probes use
   E-LB1/B1S cells, not GSK cells. Options are to cancel them as superseded or
   preregister/rebuild a GSK-family weight experiment; do not silently transplant
   their eventual coefficient onto GSK.
4. Do not infer a league-F adjustment, GSK blend strength, or per-feature weight
   from the single public/private LB score.

`.deployed_commit` is stamped and, from 2026-08-15, every deploy also appends to
`.deploy_history` — the single-line stamp had been overwritten by a later
deploy, which is how the first RANK16 scout's source commit was lost.

**P3-C2 completed.** The implementation follows the frozen contract in
[docs/P3C2_PREREGISTRATION_20260815.md](docs/P3C2_PREREGISTRATION_20260815.md):
inference success `[9,10,11]` unchanged, only cells 9/10 balanced, cell 11 and
all failure cells weight 1, analytic deweight before the success sum. Tests
17/17 modules passed, the dedicated contract passed 11 checks, and all 14
champion members remained bit-identical (blend `0.5087694207`).

Fresh control and candidate ran on `DESKTOP-053T952`, seed 3,
`val2023->test2024`, with identical row_id and target arrays elementwise and
strong matching fit/feature hashes across the two new cell members. Selection
weights were `0.70376927/1.72687783/1`; refit weights
`0.70778679/1.70315634/1`, both mass error 0. New-process prediction parity and
subset/reversal/half/single-row drift were exactly 0 for both artifacts.

**Untouched 2024 result:** cell `880.957 -> 878.993` (**-1.965**); fixed core
`888.772 -> 887.550` (**-1.222**). Source 2023 core delta **-6.671**; first half
`+8.346`, second `-10.790`, R `-0.078`, F `-9.748`. Reliability
`0.00003310 -> 0.00003315`, resolution `0.00222055 -> 0.00221857`, RMS
`0.0065013`, correlation `0.9902601`. The pre-registered rule says delta <= 0
is **FAIL**, so no n=6 extension. P3-C/P3-C2 class-weight variants are closed;
do not try cell 9-only, cell 10-only, weight/temperature/cap sweeps, or change
the success set. Champion B1S8 remains unchanged and nothing was submitted.

### Codex first pass (provisional)

The weighting/deweighting algebra is functioning: candidate weighted mass moves
cell 9/10 to `0.2373/0.2542`, and analytic correction returns them to
`0.3367/0.1512`, close to control `0.3357/0.1516`. The failure is therefore not
an extraction or calibration-routing bug. It changes the tree split geometry
enough to reduce resolution, especially late-season and F, while buying no
overall reliability. Recommendation: accept the pre-registered FAIL, do not
extend or submit, and choose a genuinely different core-resolution axis.

**Review requested from Claude:** verify the fixed-core arithmetic and the
mechanism above, then decide the next pre-registered research axis. There is no
live GPU work and no pending submission action.

The five: RANK16 **FAIL** (−219.73), H1ADD base-only **DROP** (+0.387, CI upper
+2.61), D12 **FAIL** (−62.96), P3-A **CLOSED** (−0.23), P3-B **DROP** (+0.278,
CI [−1.28, +1.84]). Only RANK16, D12 and P3-B used the GPU at all. P3-B's
common budget **803** did halve the dispersion it targeted (base BSS sd 3.702 →
1.781, best_iter sd 50.5 → 0) and every segment came out positive, but the size
was ~+0.3 rather than +3 and one seed carried it — no further budget, quantile
or multiplier search.

**RANK16 is closed, FAIL on performance.** Seed-3 gate on the judging surface,
all three members on the same host with identical `row_id` and target arrays:
base 865.50, cell 881.02, **rank 28.73**; `CONTROL_CORE 888.81` vs
`RANK_CORE 669.08` = **−219.73**. Resolution 0.000364 against base's 0.002174.
Stages 2 and 25 were not run and the 6-seed extension is cancelled. Artifacts and
the gate output are preserved with sha256 under `out/scout_recovered/`. Two
claims were narrowed in the closure audit — see the `RANK16 closure audit` entry
in `docs/SETTLED.md`: the GPU early-stopping corruption is a **3/3 reproduction
on the RANK16 production stage-1 path with fixed-length two-pass as a working
avoidance route**, *not* a general property of the library (earlier sessions
recorded healthy positive controls on an ES path), and the synthetic 40k/60k OOM
was an unrepresentative control excluded from the performance verdict.

Champion is **B1S8, LB 1108.4333490288, rank #34** (`submissions/b1s8_20260813.zip`,
sha256 `c2771bfdbbd9d81f9e43632d57fea5befeb16ff59478af06fb86114a4c6e7332`).
Unchanged — nothing has cleared the bar since, and all 14 shipped members were
re-verified bit-identical through the current `fpipe` on 2026-08-15
(`tools/verify_champion_identical.py`: 14/14 max |diff| 0.00e+00, blend
0.5087694207). Ledger 708 rows after P3-C2 merge, SETTLED 169 FLAG lines.

### 0815 브랜치 팀원 공유 기록 — Codex 1차 분석(잠정)

Mac에서 실시한 B1S base-only 8시드 비교에서는 기존 121피처를 모두 유지하고
`h1_hand_delta` 한 개를 추가한 H1ADD가 중심화 앙상블 기준 **+5.502 BSS**,
시드별 대응 평균 기준 **+4.995 BSS**(t=2.437)를 기록했다. 이 결과로 만든
Mac 전체 재학습 ZIP은 구조, 245,789행 추론, 행 부분집합 독립성 검증을 통과했다.

다만 이 결과는 아래 최신 5070의 base+cell fresh-control 결과(core -0.304)를
뒤집는 팀 판정이 아니다. 머신, 시드 수, base-only와 전체 core, 판정 통계가 달라
직접 합칠 수 없다. 따라서 B1S8의 LB 1108.433보다 높다고 주장하지 않으며,
한국어 상세 보고서는 [docs/H1ADD_0815_KO.md](docs/H1ADD_0815_KO.md)에 남겼다.

**Claude/팀원 검토 요청:** Mac base-only 결과를 독립 참고 신호로만 보관할지,
추가 실험 없이 최신 5070 DROP을 최종 판정으로 유지할지 확인해 달라. 현재 권고는
후자이며 새 GPU 학습이나 H1 계수 탐색은 제안하지 않는다.

#### Claude's answer, 2026-08-15 — DROP stands; kept as a reference signal

Reviewed and **agreed with the recommendation**. The record is precise and the
handling of the conflict is right: two numbers from different machines were not
averaged, and no LB claim was made from an unsubmitted zip. Three points to add,
one of which the report understates.

**The two results are not actually in conflict about the feature.** They differ
about the *blend*. Both machines put the base arm on the same side of zero —
Mac base-only **+4.995** (t 2.437, 8 seeds), 5070 base **+2.161** (median
+3.761, 5/6 positive, 6 seeds). What kills it is `cell −1.235`, giving core
**−0.304**. So the honest summary is not "Mac says yes, the 5070 says no"; it is
"both say the base arm probably likes `h1_hand_delta`, and the cell arm does
not, and the champion ships their blend." That distinction matters for what the
record should say, and it is why this belongs in SETTLED as corroboration rather
than as a contradiction.

**It changes nothing, because base-only is closed by pre-registration, not by
the number.** Adopting the family that happens to be positive after seeing both
is exactly the post-hoc selection the bar exists to prevent, and the existing
`H1_ADDITIVE` entry already says so. A second machine agreeing does not convert
a post-hoc split into a pre-registered one. Reopening would need a *newly*
pre-registered base-only experiment on one machine with same-session fresh
controls on both arms — and that is currently a standing prohibition, not a
proposal on the table.

**The Mac interval is thinner than "KEEP" suggests.** 95% CI `[+0.148, +9.842]`
with 1 of 8 seeds negative: the lower bound sits 0.15 above zero, so the
experiment establishes "probably not harmful to base", not "+5". Reading it as a
+5.5 gain and comparing that to the champion would repeat the v12 mistake
(offline +3.174 → LB −18.271).

Nothing to merge in code: `h1_hand_delta` (`src/fpipe.py:460`), `--h1-additive`
(`src/train_gbdt2.py:1629`), `tests/test_h1_additive.py` and
`scripts/chain_h1add_5070.bat` are already on `main`. The `0815` commit is
documentation only. No GPU was spent on this review.

---

## What today closed

| Axis | Fresh-control verdict |
|---|---|
| `--te-k b:500` | DROP — core +0.17, 95% upper +2.21 |
| two-strike routing | CLOSED — source +33.87, next season −35.82 |
| `NOISE12` (3.873% row-random corruption) | DROP — core −1.725, 95% upper +0.96 |
| `--max-ctr-complexity 3` | DROP — core −0.277, 95% CI [−1.36, +0.81] |
| `GENERAL_SKILL_ADD` | **PARK** — core +1.104, 95% CI [−1.69, +3.90] |
| F league-relative representation | DUPLICATE, not run — see below |
| `H1_ADDITIVE` (`--feat-h1 --h1-additive`) | DROP — core −0.304, 95% CI [−3.25, +2.64] |

## The three things that will change how you read numbers

**1. Fresh controls flip signs.** A candidate measured against the champion's own
stored artifacts is not the same measurement as one against a control trained in
the same session. CTR3 base was **+0.540 historical / −0.187 fresh**; the CTR3 core
was **+0.289 historical / −0.277 fresh**. Both happened to land on the same verdict;
next time they may not. Use the fresh comparison as primary.

Reuse an existing control only when host, surface, seeds, fingerprint, features,
effective params and the relevant code path are all unchanged. Today's controls
are `CTRL_base` and `NULLC_cell`, both on DESKTOP-053T952, fingerprint
`0.5352282202778269`.

**2. Re-run variation is the early-stopping pick, not GPU noise.** `NULLC_cell`
re-ran `B1S_cell`'s exact command and reproduced it: per-seed core
`[+0.50, 0.00, +0.01, 0.00, −0.00, −0.00]`, with corr(|Δbest_iter|, |Δval|) = **0.9906**
and every seed at Δiter 0 landing on 0.000 exactly. The base arm is the opposite
case — it stops at 950–1800 where the eval curve is flat, and between two identical
runs its pick moved by up to **+459 iterations**, worth +2.62. So `base +0.57, SE 0.44`
is *not* a per-seed noise floor to discount against. Print `best_iter` in every
paired report; where it did not move, the delta is mechanism.

**3. A flag reaching one arm does not mean it reached the other.** The cell arm
builds its own `CatBoostClassifier`. `--max-ctr-complexity 3` never reached it: the
packaged model said 4 and one seed reproduced its own control to the cent, exit code
0 throughout. Fixed (`_cell_params`), and `_snap()` now records `get_all_params()`
for all four constructors into the pack. Scope audited: 158 prior cell runs, none
affected, **no old verdict reopened**.

## H1 is finished, both ways

The parked replacement arm bundled two changes: it added `h1_hand_delta` and
dropped `std_asof_pitcher_success_rate_delta`, and dropping delta families is
separately worth −4.99. `--h1-additive` keeps the column, so only the composed
prior moves — 121 → 122 features, nothing removed.

```
primary fresh-vs-fresh core  -0.304  SE 1.145  t -0.27  CI [-3.25, +2.64]
base  +2.161  SE 2.346  median +3.761  5/6 positive
cell  -1.235  SE 0.900  2/6 positive
```

Removing the confound did not rescue it. Base is still the only positive family
and still imprecise — its SE here is **the largest measured**, because the base
arm's stopping point moved by 255.8 iterations on average and 712 on seed 5,
which alone cost −8.42. **Base-only is not reopened by this**: choosing a family
after seeing the split is what pre-registration exists to prevent, and this base
estimate is looser than the one that already parked.

## Open signals, none scheduled

- **Legacy failure-mode labels**: base-fixed core **+4.15, SE 1.11, t 3.76,
  6/6 positive**. Statistically KEEP-grade, mechanism unisolated — `clean14` cannot
  be defined (the two legacy-only cells are 100% corruption-populated) and NOISE12
  showed row-random corruption does not reproduce it. **RESEARCH_ONLY, not scheduled.**
  No boundary-targeted rescue, no rate sweep, no legacy-mimic variants.
- **`GENERAL_SKILL_ADD` cell family**: +2.568, t 2.18, 6/6 positive while base is
  −0.077. A family sign split, the same shape as H1 with the signs reversed. A
  cell-only arm is a **new pre-registration**, not this experiment's verdict.
- **`--fm-multilabel`**: PROVENANCE INCOMPLETE / mechanism technically open / LOW.
  No same-era baseline exists, and the path calls `_pitch_labels` without a
  `fit_mask`. Recorded as an INFRA BUG that is deliberately not scheduled.

## PARK axes — never add them together

```
H1 replacement  +1.28    expected pitch mix  +1.64
anchor-last-pitch +0.88  GENERAL_SKILL_ADD   +1.10
```
(H1 *additive* is DROP, not PARK — it does not join this list.)
"1+1+1 is +3" reasoning is banned. No seed extension to rescue a PARK.

## Why F is not the next GPU job

`x − mean(game_type, season)` is arithmetically `--feat-v5`'s column plus a
per-(league, season) step, and that step averages **0.0781 sd on R rows** — 89.09%
of the frame — against 0.6368 sd on F. `--feat-v5` is CLOSED at −26. Rank,
percentile, robust-z and transport share the same eight columns and the same intent,
so they inherit the objection, and choosing among them is the sweep the plan bans.

Design input if the axis is ever reopened with a genuinely different mechanism: the
league gap is column-specific and does not survive the regime break uniformly. Over
2023–2024, `asof_pitcher_ball_rate` sits **+0.978 sd** and `asof_pitcher_middle_rate`
**−0.851 sd**, both stable across all six seasons, while the pitcher success-rate
family has **converged to ~0** (+1.282 in 2022 → +0.451 → −0.002) and batter success
is collapsing. The leagues differ in *how pitchers miss*, not in *how good they are*.

## skill-hand: measured, and it goes the wrong way

`MATCHUP` holding 6/6 across boundaries would, read carelessly, reopen the hand
axis. It splits into the platoon (`pitcher_hand`, `batter_hand`,
`matchup_same_hand`, `dom_same_*`) and the pitcher's own record by handedness
(`te_pitcher_batter_hand_*`) — **both closed**, so both are information.

```
HAND_TE / HAND_RAW   in-sample 1.039 (.779-1.236)   unseen 0.646 (.576-.761)
retention  2021->22  RAW .320  TE .279
           2022->23  RAW .400  TE .366
           2023->24  RAW .431-.493   TE .215-.236
```

The pitcher-specific hand record is worth about as much as the platoon
in-sample and about two thirds as much out of sample, on every boundary, worst
at the most recent. **MATCHUP's stability is the platoon.** Combined with the
paragraph below, the axis is now **DO NOT BUILD**, not merely deprioritised.

## skill-hand: open in provenance, overlapping in mechanism

`skill.py` implements `axis="hand"` and it has **never run** — `--skill-axes` 0
runs, `skill_hand` 0, `axis=hand` 0. But its three extra inputs
(`te_pitcher_batter_hand_ratio` / `_dev` / `_n`) are all already champion
features, `_dev` is the exact input H1 composes, and the axis's stated
motivation — "+31 in the cross-fit residual" — is the same +31 that
`FLAG pitcher-batter-hand-residual | CLOSED` rejected: year-over-year
correlation **+0.0108**, transfer **−191.952**. With H1 now DROP on the same
input family, a richer supervised fit on those inputs has no independent
rationale. Do not promote it without one.

## The season-transfer map is done — read it before proposing a feature

Full report: [docs/SEASON_TRANSFER_MAP_20260815.md](docs/SEASON_TRANSFER_MAP_20260815.md).
It cost no GPU and no training: the champion refits on fit+val and so has **no
unseen season**, but `MV21_*`, `MVB22_native`/`MVCELL22_s42` and
`MVN3_s3,4,5`/`MVCELL_s42` were run with `--test-season`, which splits the test
season out before the refit. Same host, feature lists identical to B1S8's,
every reconstruction pinned at `replay_max_abs_diff = 0`.

**Three things it establishes.**

1. **Permuting a family measures routing, not information.** The numeric frame
   is **rank 90 of 112**: `std = asof + delta`, `dev = level / te_pitcher_ratio`,
   `shr = (rate·n+p·k)/(n+k)` at k=200 — all exact. `SEASON_STD` leaks on
   **13/13** columns, so its first place in every family table says nothing about
   information. Use the six closed blocks for information questions; they test
   CLOSED. And no deletion follows from a zero: `STD_DELTA` reads ~0 next season
   while deleting it actually costs **−16.22**.
2. **What survives a boundary is the pitcher and the platoon.**
   `PITCHER_HISTORY` STABLE 6/6 with its share *rising* out of sample (50–65% →
   63–74%), `MATCHUP` STABLE 6/6. Against that `BATTER_HISTORY` 0.21–0.43,
   `GAME_STATE` 0.03–0.50, `RECENT` median 0.31, `CALENDAR` median 0.00.
   **Do not build new derived features on the batter, game-state or calendar
   axes** — that is the reason `--feat-v4` (−18.72), `--feat-count-cat` (−9.57)
   and `--feat-count` (−1.07) all failed.
3. **Base and cell use the same information.** Agreement is spearman **+0.678**
   by representation family but **+0.943** by information block. A family sign
   split is a candidate competing with a shape the other arm already uses. This
   is the standing explanation for base-only arms looking good and failing at
   the blend — it is not a licence to pick an arm after seeing the split.

**The scope limit is now closed.** The surrogate's cell half was suspect because
the MV taxonomy has 14 cells against B1S8's 12. `B1SMOKE_base`/`B1SMOKE_cell`
(2026-08-13, the 5070) turned out to be the B1S8 recipe on exactly the judging
surface at seed 3 — **12 classes, 3 success cells, feature list identical,
replay 0** — so the calibration cost no GPU. On the champion's own taxonomy:

```
block            base same->next (ret)   cell same->next (ret)   gap
PITCHER_HISTORY  48.92 -> 66.16  1.35    55.14 -> 64.15  1.16   -2.01
MATCHUP          15.36 -> 17.54  1.14    14.21 -> 15.88  1.12   -1.66
CALENDAR_ID       7.61 ->  5.30  0.70     7.68 ->  5.36  0.70   +0.06
COUNT             7.37 ->  4.93  0.67     8.23 ->  8.52  1.04   +3.59
BATTER_HISTORY   12.30 ->  4.59  0.37     8.98 ->  3.88  0.43   -0.71
GAME_STATE        8.42 ->  1.48  0.18     5.76 ->  2.21  0.38   +0.73
base-cell spearman +0.943   max gap 3.59pp   (pre-registered: >=.85 and <=10pp)
```

Same information, different routing — **confirmed on the champion itself**, not
just on a surrogate. The proxy's cell advantage on COUNT and GAME_STATE
reproduces but smaller (+0.37 and +0.21 against +0.47 and +0.36), and every
other block reproduces within ±0.02. So the MV surrogate was a good stand-in for
*transfer* all along; its poor PredictionValuesChange agreement measured
routing, which is exactly what the two taxonomies disagree about.

## The supervision-geometry night (2026-08-15) — both tracks closed

The attribution map said base and cell route the same information differently,
so the open question was the objective, not another feature family. The Murphy
decomposition agreed: on the judging surface a perfect recalibration of the core
is worth **+13.3 BSS** (reliability is 0.013% of uncertainty) while +0.001 of
absolute resolution is worth **+400**. Two supervision geometries were tried.

**Ranking (PairLogitPairwise, group16) — BLOCKED by one corrupt artifact, not
judged.** The stage-1 model loads cleanly, reports 1224 trees and 121 features,
dumps 18.7 MB of valid JSON, and dies with `0xC0000005` on a **one-row**
predict. **The configuration is fully exonerated** — every parameter was
reproduced fresh at the failing scale and all of them score in 0.0s:

```
(60k,1200) 5,067,284B   (870k,300) 1,277,572B   (870k,1200) 5,067,268B
300-tree matrix: full254 / border32 / ctr1 / ctr1_b32 / nocat   all OK
early stopping itself: iters3000+es500+eval_set -> 1476 trees, 6,281,140B  OK
plus l2_leaf_reg=10 on that path -> 1889 trees, 8,033,988B                 OK
```

After the last arm **no parameter separates the broken artifact from a working
one.** Ruled out: OOM (one row kills it), rows, trees, model size,
`border_count`, CTRs and one-hot (both dump `cat 0, ctrs 0, one_hot 0` — a
groupwise loss leaves the applier 112 float features), the hardware, and
CatBoost 1.2.10. What is left is unique to that run: after `save_model` returned
it spun on two threads for **8h36m** and was force-killed, and three earlier
attempts on the same path aborted with exit 255 and no traceback. The mechanism
needs a debugger; it is **not** predictable from configuration, which is exactly
why the defence is per-artifact: fixed-length fitting off the early-stopping
path, `_assert_scoreable()`, and **save → load in a new process → predict one
row** at every consumer stage. That last check is the only one the broken model
ever failed. Re-running is a normal ~1.5 h/seed job now, but it is a new
pre-registration, not a resumption.

**FM_MULTILABEL_V2 — FAIL at the single-seed gate.** Both label defects repaired
first: partition-safe recovery (113 rows) and complete-case training (99.849%
kept, 1,691 dropped) instead of `nan_to_num(..., 0)` asserting "did not happen"
for "unknown". Artifact sound, fingerprint identical to the controls. **ML_CORE
876.73 against CONTROL_CORE 888.81, delta −12.07** — the pre-registered FAIL
condition. Mechanism: the multilabel member's resolution is **0.002173, the base
arm's 0.002174 and below the cell arm's 0.002207**, so swapping it into the cell
slot costs exactly the resolution the cell geometry was contributing.

**Two packaging traps found, both fixed.** A CatBoost object drops custom
attributes through joblib, so `_rank_calib` came back MISSING from every saved
rank pkl and inference would have shipped a clipped raw PairLogit score as
P(success) — mean 0.2279 against a calibrated 0.5417. `_fm_multilabel` had the
same exposure: without it a reloaded multilabel model looks binary and column 1,
P(middle), is read as P(success). Both now travel in the pack dict, and the rank
path **raises** rather than degrading when the calibration is absent.

**`schtasks` on the 5070 is not broken** — it was registered `Interactive only`
and the console session belongs to a different account, so it waited forever
(`267011` = never started). `/ru SYSTEM` fires. See `tools/run5070.sh`.

## Next candidates

1. **FM_MULTILABEL_V2 — closed by measurement, not HOLD any more.** The calibration fired the branch that
   deprioritises it: family-specific feature admission is not supported, and it
   may only be revisited as *the same information under a different supervision
   geometry*, never as a family-specialisation play. It still needs the
   implementation repaired first (partition-safe label recovery, no
   `nan_to_num(...,0)` on unknown auxiliary labels), which is a rewrite.
2. **next-season latent skill — dead, do not build.** The obvious follow-up to
   `PITCHER_HISTORY` rising out of sample was that `skill.py` aims at the rest of
   the *current* season. Retargeting it at the next season is genuinely unbuilt
   and leakage-clean, but it is a duplicate (pearson +0.926 with the current
   estimator, R² 0.990 on the frame) and it is **worse on its own ground**: on
   the same 205,033 rows of 2024, rest-of-season target explains 63.7%, plain
   k=80 shrinkage 61.4%, next-season target **59.5%**. The right horizon costs a
   season of history and 20% of pitchers, and that costs more than it buys.

No GPU candidate is licensed. The GPU has been idle since 2026-08-14.

HOLD, low expected value: `--refit-mult 2.0`, `--loss RMSE`, `--te-halflife 2`.

---

# Claude independent review of the Codex first pass — 2026-08-15

Read-only. No code changed, no GPU run, `scripts/chain_rk16_s3_5070.bat`
untouched. Every verdict is backed by the named file and line plus a
reproduction on this tree at `0b06e60`.

## Confirmed facts

**A. Champion safety — CONFIRM.** `submissions/b1s8_20260813.zip` holds 22
entries, 14 model pkls (8 base + 6 cell), and **no `failmode.py`**. The shipped
`script.py:91` calls `fpipe.predict()`; both champion pack kinds go through it
correctly — base returns `proba[:,1]`, cell returns the success-cell sum because
`fm_success` is a **list** `[9, 10, 11]`, so the `if pack.get("fm_success"):`
truthiness test is safe (a numpy array there would raise). Measured on 300 real
2024 rows: base mean 0.519804, cell mean 0.516948. The defects below reach
neither: B1S8 runs `--val-season 2024` with no test season, so its fit partition
is ≤ 2023 and B1 cannot fire; it runs `--feat-k 200`, which equals
`features.K = 200`, so B6 cannot fire. **No evidence B1S8 is invalid.**

**B1. Temporal guard is incomplete — CONFIRM (code), no impact.**
`src/train_gbdt2.py:378-398` asserts that no season exceeds `test_season`, that
test rows are exactly the test season, and that fit contains neither val nor
test. It never asserts `max(fit season) < val_season`. Minimal reproduction:
`_assert_partitions` **passes** with `val=2022, test=None` and
`fit = {2019, 2020, 2021, 2023, 2024}`, printing `fit 5 (<= 2024) | val 1 @2022`;
and with `val=2022, test=2024`, leaving 2023 in fit. **A ledger audit over all
692 rows finds 0 affected**, so nothing is invalidated.

**B3. `fm_multilabel` never reaches the submission path — CONFIRM, most severe.**
`src/fpipe.py:529-553` handles xgboost, `baseline_col` and `fm_success`, then
falls through to `proba[:,1]`. It never reads `fm_multilabel`. Measured on the
real `model/cat_ML2_s3.pkl`, whose pack **does** carry `fm_multilabel: True`:
`fpipe.predict` returns **0.130428**, exactly head 1 `P(middle)`, against the
correct head 0 `P(success)` of **0.493087**. Silent — no exception.

**B4. Teacher distillation carries a temporal leak — CONFIRM.**
`src/teacher.py:104` is `fold = rng.integers(0, args.folds, len(train))`, a
row-random K-fold over every season at once. Row-self-target leakage is blocked;
**season-transfer leakage is not** — a 2019 row's OOF teacher comes from a model
fitted on 2020–2024 rows, and `--soft-target` feeds that into the student's
training target. 58 ledger rows used it.

**B5. RMSE refit uses the selection frame — CONFIRM (code), no impact.**
`src/train_gbdt2.py:635` builds `full = Pool(train[features], ...)` where the
normal base refit uses `train_dep[features]`, while the pack stores `art_dep`.
`CatBoostRegressor` also has no `predict_proba`, so `fpipe.predict` would raise.
**0 ledger rows use `--loss RMSE`.**

**C8. `surf_report` centres on the evaluation season's own mean — CONFIRM.**
`tools/surf_report.py:29` is `p = p - (p.mean() - r)` with `r = y.mean()` of the
scored season. That constant is unavailable at submission time.

**C10. The fingerprint is one float — CONFIRM.**
`fpipe["priors"]["asof_pitcher_success_rate"]`, a single mean. It cannot
distinguish a different row set, row order or feature order that shares that
mean, and `tools/build_submission.py:73` compares only within a tag.

**C11. The two judgement tools disagree — CONFIRM.** `tools/surf_report.py:82`
uses `1.96 * se` at n=6; `tools/arm_compare.py:38,79` uses
`t(.975, df=5) = 2.571`. Same n, different interval.

**D12. The cell checkpoint is chosen on MultiClass loss — CONFIRM.**
`src/train_gbdt2.py:460` fits `loss_function="MultiClass"` and takes
`get_best_iteration()` from that curve, while the shipped quantity is
`Brier(0.45*base + 0.55*sum(success cells))`.

**E. `build_submission` can be bypassed — CONFIRM.** `--skip-smoke`
(`tools/build_submission.py:84,113`) turns the checks off, the strong synthetic
audit is only *mentioned* in a message at line 208 rather than run, and the
fingerprint comparison never crosses base/cell tags.

**F. `unittest discover` reports success on zero tests — CONFIRM.**
`python -m unittest discover -s tests` prints `Ran 0 tests ... OK` and exits 0,
because the tests are `main()`-style. There is **no CI at all** — no
`.github/workflows` — so today this misleads a human rather than a pipeline.

## Refuted / qualified findings

**B2. Rank packaging — CONFIRM the disconnect, REFUTE the severity.** Codex is
right that `fpipe.predict()` ignores `rank_calib` / `rank_ntree_end`. But a
`CatBoostRanker` has **no `predict_proba`**, so the path raises
`AttributeError: 'CatBoostRanker' object has no attribute 'predict_proba'`
(reproduced with a mock pack carrying a real fpipe artifact and the real 121
features). It is a **loud** failure, not a silently wrong number — the opposite
of B3. It still blocks shipping, and a runtime error consumes a submission, so
rank must not be packaged until `fpipe.predict` routes it; the *performance*
measurement is unaffected because the trainer scores in-process through
`_rank_probability`.

**B6. `--feat-k` — CONFIRM the code defect, REFUTE the consequence.**
`src/fpipe.py:86` passes `k=args.feat_k` at fit; line 301 calls
`add_features(df, art["priors"])` with no `k`; `feat_k` occurs exactly **once**
in `fpipe.py`, so it is never persisted. But `features.py:35` defaults to `k=K`
and `features.py:23` is `K = 200`, and the ledger contains only `--feat-k 200`
(226 rows) or no flag at all (466 rows, default 200). **The mismatch has never
been exercised. There are no verdicts to invalidate.**

**C9. Artifact consistency — PARTIAL.** `tools/arm_compare.py:59` uses
`drop_duplicates("tag")` at the default `keep="first"`, and LEDGER is
append-only, so a re-run tag would pair the **oldest** metadata with the
**newest** predictions. Real hazard, but **0 tags currently have duplicate
ledger rows**, so nothing on record is affected.

**B7. Post-`fpipe` features — PARTIAL, not adjudicated.** I did not enumerate
`feat-v4 / feat-v5 / extra_feats / PCA / role / fatigue / rules / ABS / gap /
fill_prev` one by one, and will not classify them from reading alone. What is
settled: **every flag in the B1S8 command is fpipe-side**, so the champion is
unaffected. The rest needs the mechanical test below, not an opinion.

## Existing results invalidated

**None, on the evidence available.** B1 (0/692 rows), B6 (0 non-200 runs), B5
(0 RMSE runs) and C9 (0 duplicate tags) all have empty footprints. B4 is the
only one touching recorded results — 58 `--soft-target` rows — but
`FLAG --soft-target | CLOSED` already rests on **submission-surface −10.19 with
blend weight 0.00** and already attributes the judging-surface +38.43 to F rows
only the teacher saw. The temporal leak inflates the teacher, so it makes a
negative closure **more** conservative. No verdict flips.

What *is* wrong on record is a claim of mine: the 2026-08-15 packaging line
implied the path was repaired because the keys are stored. The keys are stored
and **no consumer reads them**. Corrected append-only, not by editing.

## Champion safety verdict

**SAFE.** B1S8 is unaffected by every confirmed defect, for reasons that are
properties of its own command rather than luck about the bugs
(`--val-season 2024` with no test season; `--feat-k 200 == K`; only fpipe-side
flags; binary and cell packs only). No re-submission or re-packaging is required.

## Required fixes, in order

1. **`fpipe.predict` must route by pack kind, or refuse.** Add `rank_calib` and
   `fm_multilabel` branches and make an unrecognised pack raise rather than fall
   through to `proba[:,1]`. This is the only fix that closes a *silent* wrong
   answer (B3), and it also removes B2 and half of B5.
2. **Temporal guard**: assert `max(fit season) < val_season` (B1).
3. **Store `feat_k` in the artifact and read it in `transform`** (B6).
4. **`build_submission`**: make the strong subset audit required, compare
   fingerprints across base and cell tags, and make `--skip-smoke` refuse to
   write a zip rather than skipping checks (E).
5. **Fingerprint**: sorted-row_id hash + row count + ordered feature hash + key
   params, replacing the single float (C10).
6. **RMSE refit**: use `train_dep` like the other refits (B5).
7. **Teacher**: rolling OOF — season S predicted by a teacher fitted on < S — or
   leave the axis closed (B4).
8. **Judgement tools**: one interval convention, and `keep="last"` in
   `arm_compare` (C9, C11).

## Regression tests required

- `fpipe.predict` on a rank pack, a multilabel pack and an unknown pack: the
  first two must equal the sanctioned path, the third must raise. **This test
  would have caught B2 and B3 and nothing currently does** —
  `test_rank_contract` exercises `rank_probability_from_pack`, which the
  submission never calls.
- `_assert_partitions` must reject `val=2022, test=None` with 2023/2024 in fit.
- fit/transform parity for `--feat-k != 200` on a small frame.
- `build_submission --skip-smoke` must not produce a zip.
- Replace `unittest discover` with a runner that **fails on zero tests
  collected**.

## Experiments allowed after repair

- RANK16 performance measurement may proceed **without** fix 1, because the
  trainer scores in-process; no rank pack may be packaged or submitted until fix
  1 lands. The seed-3 chain launched before this review is being allowed to
  finish; its numbers are held, not judged.
- FM_MULTILABEL_V2 stays closed on performance (−12.07) regardless.
- D12's cell-checkpoint experiment is a genuine single change and is **not**
  duplicated by `--eval-metric BrierScore` (that was the binary arm) or by the
  cell-5000 and multilabel axes. It is not licensed by anything measured today
  and belongs behind the current queue.

## Handoff back to Codex

Numbers and citations are reproducible on `0b06e60`. Two places where I differ
from the first pass and the difference matters: **B2 is loud, not silent** (so
B3 is the one to fix first), and **B6 has no footprint at all** (so no
re-adjudication of past runs is needed). B7 is left explicitly unadjudicated
rather than guessed.


## Adoption rule (unchanged)

```
KEEP  delta >= +3 AND t >= 2.4 AND n >= 6 paired AND same machine AND same surface
DROP  95% upper < +3
else  PARK          KEEP only ever triggers temporal stress
```

Local→LB for the B1S family: **debiased score + 139.03** (confirmed on two points).
Do not carry that offset to another structure — the previous family's +128.4 got a
prediction's sign wrong.

## Before starting anything

```bash
bash tools/ledger_sync.sh                  # the trainer writes its row on the runner
python tools/precheck.py <every flag>      # exit 2 means forbidden
```

Judging tools: `tools/arm_compare.py` (paired, prints best_iter),
`tools/cell_arm_delta.py` (holds a base family fixed),
`tools/orphan_audit.py` (lead generator only — its raw deltas pair baselines
naively and put distillation at +241).

## Independent local P3-C2 replication

After the primary 5070 FAIL was recorded, the same seed-3 experiment was run
independently on the current RTX 3070 Ti with a fresh local base and control.
Fixed core moved **890.294435 -> 885.486834 = -4.807601**; source 2023, both
2024 row halves, both calendar halves, and R/F were all negative. The machines'
point estimates are not pooled, but both fire the same `delta <= 0` FAIL branch.
Closure and champion status are unchanged. Details:
[docs/P3C2_LOCAL_REPLICATION_20260815.md](docs/P3C2_LOCAL_REPLICATION_20260815.md).

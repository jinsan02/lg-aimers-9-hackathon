@echo off
REM RANK16 -- the first honest judgement of the pairwise-ranking axis.
REM
REM Why this axis. The season-transfer map showed base and cell agree at
REM spearman +0.943 on which information blocks survive a boundary, with no gap
REM over 3.6pp: the two arms route the same information differently. So another
REM feature family is not the open question -- the objective is. PairLogitPairwise
REM optimises ordering, which is the resolution term of the Brier decomposition,
REM and resolution is where the remaining headroom sits.
REM
REM Why two processes. FLAG --model rank full-refit is BANNED: group16 crashed
REM natively on both the 4070 and the A100 when the deployment ranker was built
REM while the selection ranker's CUDA pair buffers were still alive, and deleting
REM the Python objects first did not help. These are two separate interpreter
REM invocations, so the CUDA context dies with stage 1 and only scalars in a JSON
REM file cross the boundary. Stage 2 refuses to run unless the frame it rebuilds
REM matches stage 1 on features, cat_cols, group size, seed, row counts and the
REM training-set fingerprint.
REM
REM group16 is FIXED. 64 is BANNED (OOM on the 4070, segfault on the A100) and
REM 8/32 are not swept -- the plan forbids it.
REM
REM Surface is the judging one, so it pairs with B1SMOKE_base / B1SMOKE_cell
REM seed 3, which are the B1S8 recipe on exactly this surface.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_s3.json

REM ---- stage 1: selection only. Writes scalars, scores unseen 2024 with the
REM      selection ranker (a legitimate no-refit reference), then exits.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 1 --no-refit --seed 3 --tag RANK16S1_s3 > out\RANK16S1_s3.log 2>&1
echo %ERRORLEVEL% > out\RANK16S1_s3.exit

REM ---- stage 2 only runs if stage 1 actually produced the handoff. The first
REM      attempt died with exit 255 and no traceback after early stopping, so
REM      stage 2 started anyway and failed on a missing file, which reads like a
REM      stage-2 bug when the fault was upstream.
if not exist out\rank_meta_s3.json (
  echo STAGE1_PRODUCED_NO_META > out\RANK16_s3.exit
  goto :end
)

REM ---- stage 2: fresh interpreter, fresh CUDA context, deployment ranker.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 2 --seed 3 --tag RANK16_s3 > out\RANK16_s3.log 2>&1
echo %ERRORLEVEL% > out\RANK16_s3.exit

:end
echo DONE > out\RANK16_s3.done

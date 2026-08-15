@echo off
REM RANK16 seed 3 -- stage 1 re-fit, ONCE, under the termination contract.
REM
REM Why this exists. The 2026-08-15 scout passed every pre-registered check
REM (best_iteration 880, curve 1381, 5,876,280 bytes, 121 features in order,
REM three MARKs, judging surface) and then died 0xC0000005 predicting ONE row --
REM on desktop-5070 itself, the machine that wrote it, with a byte-identical
REM sha256 on both hosts and a fresh control ranker of the same shape scoring
REM through the same probe in 0.0s. The earlier rank_stage1_RANK16S1_s3.cbm
REM fails identically. Two for two.
REM
REM So the reuse gate failed and the handoff is re-derived. Whatever
REM best_iteration this run reports is the one stage 12 uses. Choosing between
REM it and the old 880 is forbidden.
REM
REM NEW TAG AND NEW META PATH. RK16S3_scout / rank_meta_rk16s3.json are
REM preserved evidence and must not be overwritten -- AGENTS.md forbids
REM destroying a failed artifact, and the cause being written down does not make
REM it disposable while the axis is still open.
REM
REM --stage-run-id makes this announce out\handoff\rank1_RK16S3B_scout.ready.json
REM by atomic rename once the model and meta are on disk. A separate supervisor
REM task verifies both against the sha recorded there and ends THIS task from
REM outside: os._exit left a finished stage spinning 8h36m here, and
REM TerminateProcess and taskkill /f /pid both returned success while the
REM process stayed.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_rk16s3b.json

%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 1 --no-refit --seed 3 --stage-run-id 20260815b --tag RK16S3B_scout > out\RK16S3B_scout.log 2>&1
echo %ERRORLEVEL% > out\RK16S3B_scout.exit
echo DONE > out\RK16S3B_stage1.done

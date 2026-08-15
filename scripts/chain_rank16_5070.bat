@echo off
REM RANK16 -- the first honest judgement of the pairwise-ranking axis.
REM
REM Why this axis. The season-transfer map showed base and cell agree at
REM spearman +0.943 on which information blocks survive a boundary, with no gap
REM over 3.6pp: the two arms route the same information differently. So another
REM feature family is not the open question -- the objective is. And the Murphy
REM decomposition says a perfect recalibration of the core is worth only +13.3
REM BSS while +0.001 of absolute resolution is worth +400, so the headroom is
REM entirely in resolution. PairLogitPairwise optimises ordering, which is the
REM resolution term.
REM
REM Why four processes. A GPU ranker cannot be touched again inside the process
REM that fitted it. Three runs died with exit 255 and no traceback -- twice in
REM CatBoost's best-model shrink (best_iter 977 and 762), once at the very next
REM step with the shrink disabled. A fourth completed its work and then refused
REM to exit, sitting alive 8.6 hours on two spinning threads while the batch
REM waited on it. So every stage fits, saves natively, and terminates itself
REM with TerminateProcess; the scoring and packaging happen in CPU-only
REM interpreters that never open a CUDA context.
REM
REM   1   GPU   fit selection ranker, save .cbm, write handoff, hard exit
REM   1b  CPU   load, score val, fit sigmoid, complete handoff, score unseen
REM   2   GPU   fit deployment ranker on the full frame, save .cbm, hard exit
REM   2b  CPU   load, attach calibration, score unseen, package the pkl
REM
REM group16 is FIXED. 64 is BANNED (OOM on the 4070, segfault on the A100) and
REM 8/32 are not swept. Surface is the judging one, so it pairs with
REM B1SMOKE_base / B1SMOKE_cell seed 3, the B1S8 recipe on exactly this surface.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_s3.json

REM ---- 1b: complete the handoff from the stage-1 .cbm already on disk.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 15 --no-refit --device CPU --seed 3 --tag RANK16S1_s3 > out\RANK16S15_s3.log 2>&1
echo %ERRORLEVEL% > out\RANK16S15_s3.exit
findstr /B /C:"0" out\RANK16S15_s3.exit >nul 2>&1
if errorlevel 1 goto :end

REM ---- 2: deployment ranker on the full frame, GPU, then hard exit.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 2 --seed 3 --tag RANK16_s3 > out\RANK16_s3.log 2>&1
echo %ERRORLEVEL% > out\RANK16_s3.exit
findstr /B /C:"0" out\RANK16_s3.exit >nul 2>&1
if errorlevel 1 goto :end

REM ---- 2b: score the unseen season and package, CPU only.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 25 --device CPU --seed 3 --tag RANK16_s3 > out\RANK16S25_s3.log 2>&1
echo %ERRORLEVEL% > out\RANK16S25_s3.exit

:end
echo DONE > out\RANK16_s3.done

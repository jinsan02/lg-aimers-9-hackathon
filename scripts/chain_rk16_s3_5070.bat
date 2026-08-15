@echo off
REM RANK16 seed 3 -- the first valid performance measurement of the axis.
REM
REM Pre-registered as a NEW performance experiment, not a resumption. The prior
REM incident is closed: the configuration was fully exonerated (every parameter
REM of the failing fit reproduced fresh at the failing scale and scored in 0.0s,
REM including the early-stopping path and l2_leaf_reg=10), and the corruption
REM was isolated to one artifact produced by a run that spun for 8h36m after
REM save_model and had to be force-killed. No incident probe runs here.
REM
REM The question is the objective, nothing else: does PairLogitPairwise over the
REM same 121 features buy enough untouched-2024 resolution to replace the binary
REM base slot? Everything else is held at the B1S recipe.
REM
REM Five processes, because a GPU ranker must not be touched again inside the
REM process that fitted it, and because the only check the broken artifact ever
REM failed was save -> load in a NEW process -> predict one row.
REM
REM   1   GPU  scout: iters 3000 + es 500 + eval_set. Reports best_iteration.
REM              Its model is rank_scout_*.cbm and is NEVER scored or shipped.
REM   12  GPU  selection ranker at best_iteration+1, fixed length, no eval_set,
REM              no early stopping. _assert_scoreable then hard exit.
REM   15  CPU  fresh process: load, assert, one-row smoke, score val, calibrate.
REM   2   GPU  deployment ranker at _refit_trees(best_iter, 1.5), fixed length.
REM   25  CPU  fresh process: load, assert, one-row smoke, score unseen 2024,
REM              package with rank_calib and rank_ntree_end in the pack.
REM
REM group16 is FIXED and its definition is not touched. Surface is the judging
REM one, pairing with B1SMOKE_base / B1SMOKE_cell seed 3 on this host.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_rk16s3.json

REM ---- 1: scout (early stopping). Model not for consumption.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 1 --no-refit --seed 3 --tag RK16S3_scout > out\RK16S3_scout.log 2>&1
echo %ERRORLEVEL% > out\RK16S3_scout.exit
if not exist out\rank_meta_rk16s3.json goto :end

REM ---- 12: fixed-length selection ranker.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 12 --no-refit --seed 3 --tag RK16S3_sel > out\RK16S3_sel.log 2>&1
echo %ERRORLEVEL% > out\RK16S3_sel.exit
if not exist out\rank_select_RK16S3_sel.cbm goto :end

REM ---- 15: fresh CPU process. Smoke, score validation, calibrate.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 15 --no-refit --device CPU --seed 3 --tag RK16S3_sel > out\RK16S3_sel15.log 2>&1
echo %ERRORLEVEL% > out\RK16S3_sel15.exit
findstr /B /C:"0" out\RK16S3_sel15.exit >nul 2>&1
if errorlevel 1 goto :end

REM ---- 2: fixed-length deployment ranker.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 2 --seed 3 --tag RK16S3 > out\RK16S3_dep.log 2>&1
echo %ERRORLEVEL% > out\RK16S3_dep.exit
if not exist out\rank_stage2_RK16S3.cbm goto :end

REM ---- 25: fresh CPU process. Smoke, score unseen 2024, package.
%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 25 --device CPU --seed 3 --tag RK16S3 > out\RK16S3_dep25.log 2>&1
echo %ERRORLEVEL% > out\RK16S3_dep25.exit

:end
echo DONE > out\RK16S3.done

@echo off
REM RANK16 seed 3 -- stage 15: fresh CPU process. Load, assert, smoke, score
REM validation 2023, fit the calibration sigmoid, complete the handoff.
REM
REM No GPU and no fit, so no termination contract: the stage-12 worker exited on
REM its own (RK16S3B_sel.exit and .done both written), unlike every early-stopped
REM stage-1 run.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_rk16s3b.json

%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 15 --no-refit --device CPU --seed 3 --tag RK16S3B_sel > out\RK16S3B_sel15.log 2>&1
echo %ERRORLEVEL% > out\RK16S3B_sel15.exit
echo DONE > out\RK16S3B_stage15.done

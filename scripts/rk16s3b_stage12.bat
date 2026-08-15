@echo off
REM RANK16 seed 3 -- stage 12: the fixed-length selection ranker.
REM
REM iterations = scout best_iteration + 1 = 952, no eval_set, no early stopping.
REM The scout's own model is discarded; only its curve position is reused, and
REM it is the number this run's scout produced (951), not the first run's 880.
REM Choosing between them is forbidden.
REM
REM This stage is also the experiment the synthetic control could not run. All
REM three unscoreable artifacts so far were EARLY-STOPPED stage-1 models; the
REM fixed-length path has never been put through save -> fresh process -> one-row
REM predict on real data. Stage 15 is that check, and it runs next, on CPU, in
REM its own process. If stage 12's model fails there, the axis is blocked on
REM infrastructure rather than on its own merits.
REM
REM Stage 12 rewrites rank_meta_rk16s3b.json in place (stage: scout ->
REM selected). The scout record is copied to rank_meta_rk16s3b.scout_backup.json
REM before this runs, and to out/scout_recovered/ on the laptop.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8
set RANK=--model rank --rank-group-size 16 --rank-meta out\rank_meta_rk16s3b.json

%PYU% src\train_gbdt2.py %CORE% %RANK% --rank-stage 12 --no-refit --seed 3 --stage-run-id 20260815c --tag RK16S3B_sel > out\RK16S3B_sel.log 2>&1
echo %ERRORLEVEL% > out\RK16S3B_sel.exit
echo DONE > out\RK16S3B_stage12.done

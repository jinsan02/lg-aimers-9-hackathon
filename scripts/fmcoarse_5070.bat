@echo off
REM FMCOARSE stage 1 -- collapse the failure block to one class.
REM Pre-registered in docs/FMCOARSE_PREREGISTRATION_20260816.md.
REM
REM The cell arm trains with 12-way MultiClass CE and is scored on the
REM aggregated binary Brier of the success cells, so CE spends capacity on
REM distinctions the metric cannot see. Measured on a 120-bucket grid:
REM I(bucket; within-failure) = 0.028673 nats/row vs I(bucket; block) =
REM 0.003885 -- 7.4x. SUCCESS_AUX_GRADIENT then showed that supervision fights
REM the primary (100% minibatch conflict for reverse) and that nothing survives
REM PCGrad projection. That experiment KEPT the supervision; this deletes it.
REM
REM Judging surface, because it is the only one with an untouched season, which
REM is what a mechanism question needs. Nine fits: one base arm shared by both
REM cores, plus control and candidate cell arms, three seeds each.
REM
REM Kill-check fixed in advance: rms(candidate, control) < 0.002 on untouched
REM 2024 closes the axis regardless of sign.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --p1 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3,4,5

echo ==== FMCOARSE base (shared, fixes the core) ====
%PYU% src\train_gbdt2.py %COMMON% --depth 8 %SEEDS% --tag FMC_base
echo ==== FMCOARSE control cell (12 classes) ====
%PYU% src\train_gbdt2.py %COMMON% %CELL% %SEEDS% --tag FMCCTL_cell
echo ==== FMCOARSE candidate cell (4 classes) ====
%PYU% src\train_gbdt2.py %COMMON% %CELL% --fm-coarse failure %SEEDS% --tag FMC_cell
echo ==== FMCOARSE stage 1 done ====

@echo off
REM B0-JL -- judging, legacy-like. The new zero point for cheap screening.
REM
REM P0 is fixed here (future-season cutoff, split-safe cell labels); P1 is NOT
REM (TE global prior and skill first-season fallback keep their legacy behaviour).
REM So a B0 number is a paired delta inside one implementation, never "honest"
REM evidence. Adoption happens on B1-J, not here.
REM
REM One host, one config. Do not edit the core below for a candidate -- add the
REM candidate's flag to its own arm so the delta isolates one change.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024

set SEEDS=%1
if "%SEEDS%"=="" set SEEDS=3,4,5

%PY% src\train_gbdt2.py %CORE% --depth 8 --seeds %SEEDS% --tag B0JL_base > out\B0JL_base.log 2>&1
echo %ERRORLEVEL% > out\B0JL_base.exit

%PY% src\train_gbdt2.py %CORE% --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --seeds %SEEDS% --tag B0JL_cell > out\B0JL_cell.log 2>&1
echo %ERRORLEVEL% > out\B0JL_cell.exit

echo DONE > out\B0JL.done

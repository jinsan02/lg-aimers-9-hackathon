@echo off
REM Temporal stress case: val2022 -> unseen 2023. NOT a second statistical
REM sample of the 2023->2024 surface -- the audit is explicit that these are two
REM stress cases, not independent draws. A candidate is run here to see whether
REM its sign survives a different transition, never to pool the numbers.
REM
REM   run_judge2223_5070.bat B1S22 "3,4,5" --p1
REM
REM Why --drop-f-pre 2023 and not 2022. The F league changes regime exactly
REM between these two seasons: success .7087 in 2022, .4729 in 2023, while R
REM barely moves (.5037 -> .5031). Measured 2026-08-13 on the official train.
REM
REM   --drop-f-pre 2022  train/val R-only, test 2023 carries 25,686 F rows the
REM                      model has never seen. That folds the open F-resolution
REM                      hole into the temporal answer.
REM   --drop-f-pre 2023  R-only throughout. The transition question is isolated.
REM
REM The second is the one that answers the question being asked. Note this makes
REM the surface R-only, so its absolute score is not comparable to the 2023->2024
REM surface, which keeps new-regime F in both training and test.
REM
REM The old rolling runs on this surface used no --drop-f-pre at all and scored
REM val BSS 2396.60 -- old-regime F made 2022 trivially predictable. They are in
REM docs/INVALIDATED.tsv for a different reason (future seasons in the fit pool);
REM this is the second reason not to revive them.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2023 --max-train-season 2023 --val-season 2022 --test-season 2023

set TAG=%~1
if "%TAG%"=="" (
  echo usage: run_judge2223_5070.bat ^<TAG^> "^<SEEDS^>" [extra flags]
  exit /b 2
)
set SEEDS=%~2
if "%SEEDS%"=="" set SEEDS=3,4,5

set EXTRA=
shift
shift
:more
if "%~1"=="" goto go
set EXTRA=%EXTRA% %~1
shift
goto more
:go

echo running %TAG% seeds %SEEDS% extra:%EXTRA%

%PYU% src\train_gbdt2.py %CORE%%EXTRA% --depth 8 --seeds %SEEDS% --tag %TAG%_base > out\%TAG%_base.log 2>&1
echo %ERRORLEVEL% > out\%TAG%_base.exit

%PYU% src\train_gbdt2.py %CORE%%EXTRA% --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --seeds %SEEDS% --tag %TAG%_cell > out\%TAG%_cell.log 2>&1
echo %ERRORLEVEL% > out\%TAG%_cell.exit

echo DONE > out\%TAG%.done

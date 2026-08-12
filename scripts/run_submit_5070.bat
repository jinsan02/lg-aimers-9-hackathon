@echo off
REM B1-S -- submission shape. val2024, NO --drop-f-pre, no test season.
REM
REM   run_submit_5070.bat B1S "3,4,5,6,8,13" --p1
REM
REM This is the only surface whose training set matches the shipped champion,
REM so it is the only one where "did the rebuild beat v11" is a real question.
REM Verify with tools/member_fingerprint.py: the champion's
REM fpipe['priors']['asof_pitcher_success_rate'] is 0.5401750413.
REM
REM There is no unseen season here -- the data ends at 2024 and 2024 is the
REM validation season. The only score is val2024. Do not mix its absolute value
REM with a judging-surface number (SETTLED.md -> drop-f-pre-omitted | BANNED).
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024

set TAG=%~1
if "%TAG%"=="" (
  echo usage: run_submit_5070.bat ^<TAG^> "^<SEEDS^>" [extra flags]
  exit /b 2
)
set SEEDS=%~2
if "%SEEDS%"=="" set SEEDS=3,4,5,6,8,13

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

@echo off
REM Judging surface (val2023 -> unseen 2024), one host, one core config.
REM
REM   run_judge_5070.bat B0JL "3,4,5"
REM   run_judge_5070.bat B1J  "3,4,5" --p1
REM
REM Runs both arms (binary depth 8, 14-cell MultiClass depth 5) and writes
REM <TAG>_base / <TAG>_cell. Never edit CORE for a candidate -- pass the
REM candidate's flag as an extra argument so the delta isolates one change.
REM
REM This is NOT the submission surface. It uses --drop-f-pre 2022; the
REM submission does not, and mixing the two flag sets is what produced
REM best_iter 12-21 in P2'-B (SETTLED.md -> drop-f-pre-omitted | BANNED).
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe

REM -u so the log fills as the run goes. Buffered, B0-JL showed a 0-byte log for
REM 50 minutes and there was no way to tell a live run from a dead one -- which
REM is how two overlapping GPU jobs went unnoticed.
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024

set TAG=%~1
if "%TAG%"=="" (
  echo usage: run_judge_5070.bat ^<TAG^> "^<SEEDS^>" [extra flags]
  exit /b 2
)

REM cmd splits an unquoted argument on commas, so `... 3,4,5` arrives as
REM %2=3 %3=4 %4=5 and only seed 3 runs -- with no _s3 suffix on the tag, which
REM then breaks every downstream glob. Pass it quoted and take %~2.
set SEEDS=%~2
if "%SEEDS%"=="" set SEEDS=3,4,5

REM Everything after the first two arguments is forwarded to both arms.
set EXTRA=
shift
shift
:more
if "%~1"=="" goto go
REM %1 not %~1: cmd splits bat parameters on space, comma AND equals, so a
REM value like "strikes_before == 2" only survives inside its quotes. %~1
REM strips them and the next expansion shatters the token.
set EXTRA=%EXTRA% %1
shift
goto more
:go

echo running %TAG% seeds %SEEDS% extra:%EXTRA%

%PYU% src\train_gbdt2.py %CORE%%EXTRA% --depth 8 --seeds %SEEDS% --tag %TAG%_base > out\%TAG%_base.log 2>&1
echo %ERRORLEVEL% > out\%TAG%_base.exit

%PYU% src\train_gbdt2.py %CORE%%EXTRA% --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --seeds %SEEDS% --tag %TAG%_cell > out\%TAG%_cell.log 2>&1
echo %ERRORLEVEL% > out\%TAG%_cell.exit

echo DONE > out\%TAG%.done

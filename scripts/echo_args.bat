@echo off
REM Argument-forwarding harness. The block below between the BEGIN/END markers
REM must stay byte-identical to the one in run_judge_5070.bat,
REM run_submit_5070.bat and run_judge2223_5070.bat -- tests/test_runner_args.py
REM compares them and fails if they drift. Testing a copy that has quietly
REM diverged from the real runner is worse than not testing at all.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.venv\Scripts\python.exe

set TAG=%~1
if "%TAG%"=="" (
  echo usage: echo_args.bat ^<TAG^> "^<SEEDS^>" [extra flags]
  exit /b 2
)
set SEEDS=%~2
if "%SEEDS%"=="" set SEEDS=3,4,5

REM ---- FORWARD BEGIN ----
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
REM ---- FORWARD END ----

echo TAG=%TAG%
echo SEEDS=%SEEDS%
%PY% tools\runner_echo_args.py %EXTRA%

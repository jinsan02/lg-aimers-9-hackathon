@echo off
REM Two cell arms back to back, submission surface. No base arm on purpose.
REM
REM   NOISE12  the experiment: corrected 12-cell taxonomy + the pre-registered
REM            3.873% deterministic structured auxiliary-label corruption
REM   NULLC    the control: byte-identical to B1S_cell except for the tag
REM
REM Why no base arm. `--fm-noise-rate` is consumed inside `if args.failmode_cells`,
REM so it cannot reach a depth-8 binary run -- re-running base would only
REM resample GPU nondeterminism. Holding one base family fixed across both
REM comparisons removes that noise from `core` entirely, which is strictly
REM better than how LEGLBL was measured: there the base arm was re-run and
REM contributed +0.57 of pure run-to-run drift to a +3.97 headline.
REM
REM Why NULLC exists at all. 635 ledger rows and not once has the same config
REM been run twice under two tags on the same host, so the cell arm's
REM run-to-run floor has never been measured. LEGLBL's own base arm -- an
REM identical command with a flag that is a no-op for it -- moved +2.62 on seed
REM 4. Without NULLC there is no scale against which +3.97 or whatever NOISE12
REM returns can be read.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=3,4,5,6,8,13

%PYU% src\train_gbdt2.py %CORE% --p1 --fm-noise-rate 0.03873 %CELL% --seeds %SEEDS% --tag NOISE12_cell > out\NOISE12_cell.log 2>&1
echo %ERRORLEVEL% > out\NOISE12_cell.exit

%PYU% src\train_gbdt2.py %CORE% --p1 %CELL% --seeds %SEEDS% --tag NULLC_cell > out\NULLC_cell.log 2>&1
echo %ERRORLEVEL% > out\NULLC_cell.exit

echo DONE > out\NOISE12.done

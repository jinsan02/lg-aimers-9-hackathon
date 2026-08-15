@echo off
REM P3-C2 -- fresh seed-3 control and structured-success-only weighted cell CE.
REM Pre-registered in docs/P3C2_PREREGISTRATION_20260815.md. The only model
REM change is --p3c2-balanced on the candidate. Both arms dump full cell
REM probabilities for the frozen calibration/resolution report.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --dump-cell-proba --seed 3

%PYU% src\train_gbdt2.py %CORE% --tag P3C2CTL_cell > out\P3C2CTL_cell.log 2>&1
set RC=%ERRORLEVEL%
echo %RC% > out\P3C2CTL_cell.exit
if not "%RC%"=="0" goto :end

%PYU% src\train_gbdt2.py %CORE% --p3c2-balanced --tag P3C2CAND_cell > out\P3C2CAND_cell.log 2>&1
set RC=%ERRORLEVEL%
echo %RC% > out\P3C2CAND_cell.exit

:end
echo DONE > out\P3C2.done

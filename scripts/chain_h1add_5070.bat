@echo off
REM H1_ADDITIVE -- add h1_hand_delta, keep the delta the old arm removed.
REM
REM The parked H1 arm changed two things: it added h1_hand_delta and it dropped
REM std_asof_pitcher_success_rate_delta, a champion feature. precheck's own note
REM records that dropping delta families already costs -4.99, so base +3.88 /
REM cell -1.55 / core +0.63 measured the pair. --h1-additive keeps the column,
REM leaving one change: 121 -> 122 features.
REM
REM Controls are CTRL_base and NULLC_cell, both on this host from 2026-08-14.
REM Their reuse was asserted, not assumed: running fit() under the control flags
REM through the pre-change and post-change fpipe gives 76 identical added
REM columns, identical categoricals, identical row order, numeric max_abs_diff
REM 0.000e+00, and artifact keys differing only by h1_additive (False, unused).
REM
REM Whole recipe on purpose. base and cell have split signs on the same feature
REM change before, and picking a family after seeing the result is the post-hoc
REM selection the pre-registration forbids. Base does not gate the cell run.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=3,4,5,6,8,13

%PYU% src\train_gbdt2.py %CORE% --p1 --feat-h1 --h1-additive --depth 8 --seeds %SEEDS% --tag H1ADD_base > out\H1ADD_base.log 2>&1
echo %ERRORLEVEL% > out\H1ADD_base.exit

%PYU% src\train_gbdt2.py %CORE% --p1 --feat-h1 --h1-additive %CELL% --seeds %SEEDS% --tag H1ADD_cell > out\H1ADD_cell.log 2>&1
echo %ERRORLEVEL% > out\H1ADD_cell.exit

echo DONE > out\H1ADD.done

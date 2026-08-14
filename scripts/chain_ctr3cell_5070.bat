@echo off
REM CTR3 cell arm, re-run after the flag was found not to reach it.
REM
REM The first attempt (tag CTR3_cell, killed at 2 of 6 seeds) reported
REM max_ctr_complexity 4 in its packaged model and reproduced NULLC_cell to the
REM cent. The cell arm builds its own CatBoostClassifier and did not consume the
REM binary arm's params dict; _cell_params now feeds it.
REM
REM New tag so the invalid partial run can never be paired with this one by a
REM later glob. Control is NULLC_cell, same host, same day, argv identical to
REM B1S_cell apart from the tag.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005

%PYU% src\train_gbdt2.py %CORE% --p1 --max-ctr-complexity 3 %CELL% --seeds 3,4,5,6,8,13 --tag CTR3C_cell > out\CTR3C_cell.log 2>&1
echo %ERRORLEVEL% > out\CTR3C_cell.exit
echo DONE > out\CTR3C.done

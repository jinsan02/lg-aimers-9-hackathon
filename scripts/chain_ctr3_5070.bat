@echo off
REM CTR3 isolated, with a fresh same-session control for the base arm.
REM
REM   CTRL_base   B1S8's base command re-run today under a new tag
REM   CTR3_base   the same, plus --max-ctr-complexity 3
REM   CTR3_cell   the same, plus --max-ctr-complexity 3
REM
REM Hypothesis: max_ctr_complexity 4 -> 3 is a categorical-interaction
REM regularisation. 4 is what B1S8 actually applied -- read out of
REM model/cat_B1S_base_s3.pkl with get_all_params(), not assumed from the docs --
REM so this arm makes CatBoost combine fewer categorical features at once,
REM across the 9 cat columns (7 declared + month_cat/dow_cat from --feat-v2).
REM
REM Why CTRL_base exists. NULLC_cell showed the cell arm re-runs deterministically
REM and that what little moves is the early-stopping pick, not diffuse GPU noise:
REM corr(|d best_iter|, |d val|) = 0.9906, and seeds with d_iter 0 reproduce to
REM 0.000. The base arm is the opposite case -- it stops at 950-1800 where the
REM eval curve is flat, and between two identical runs its pick moved by up to
REM +459 iterations, worth +2.62 BSS. So a base-arm delta measured against the
REM historical champion run cannot be separated from stopping lottery. Against a
REM control trained in this same session it can.
REM
REM The cell control is NULLC_cell, run earlier today on this host with argv
REM identical to B1S_cell apart from the tag. Re-running a second cell control
REM would cost 35 minutes to re-demonstrate determinism that was just measured.
REM
REM Six seeds, not B1S8's eight: n>=6 paired is the adoption bar and every other
REM comparison today is on these six. The extra two base seeds are a blend
REM decision, not a training difference.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=3,4,5,6,8,13

%PYU% src\train_gbdt2.py %CORE% --p1 --depth 8 --seeds %SEEDS% --tag CTRL_base > out\CTRL_base.log 2>&1
echo %ERRORLEVEL% > out\CTRL_base.exit

%PYU% src\train_gbdt2.py %CORE% --p1 --max-ctr-complexity 3 --depth 8 --seeds %SEEDS% --tag CTR3_base > out\CTR3_base.log 2>&1
echo %ERRORLEVEL% > out\CTR3_base.exit

%PYU% src\train_gbdt2.py %CORE% --p1 --max-ctr-complexity 3 %CELL% --seeds %SEEDS% --tag CTR3_cell > out\CTR3_cell.log 2>&1
echo %ERRORLEVEL% > out\CTR3_cell.exit

echo DONE > out\CTR3.done

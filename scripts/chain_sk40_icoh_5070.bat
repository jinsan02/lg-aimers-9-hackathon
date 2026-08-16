@echo off
REM SK40 + ICOH -- two rule-violation recoveries, queued behind FMCOARSE.
REM Pre-registered in docs/SK40_ICOH_PREREGISTRATION_20260816.md.
REM
REM SK40: --std-k 40 was recorded CLOSED at submission-surface -1.954, SE 2.377.
REM   With n=8 the 95% upper is -1.954 + 2.365*2.377 = +3.67, so rule 5 says
REM   PARK. Worse, the number is a 4070 base-arm-only run and the champion is on
REM   the 5070 -- cross-machine comparison is BANNED, and std-k 40 is itself the
REM   axis that diverged 11 points between machines. k=40 is re-measured rather
REM   than the better-looking k=20 because the curve that favours k20 is the
REM   same contaminated one (k40 scored +4.27 on the 4070 and +10.31 on the
REM   A100). A clean DROP here closes the whole downward direction.
REM
REM ICOH: all 8 --feat-id-cohort ledger rows are single-seed, against a measured
REM   base seed spread of -7.46..+6.74, so every recorded number is inside the
REM   noise floor. Two of three boundaries were +13.93 and +8.13.
REM
REM WAIT FIRST: one GPU job per machine. Poll for FMCOARSE's completion marker
REM and abort after 60 minutes rather than start alongside it. Failing closed is
REM deliberate -- a crashed FMCOARSE must not silently become a concurrent run.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set /a WAITED=0
:waitloop
findstr /c:"stage 1 done" C:\aimers\out\fmcoarse.log >nul 2>&1
if not errorlevel 1 goto ready
set /a WAITED+=1
if %WAITED% GTR 120 goto aborted
REM ping, not timeout: timeout needs a console and this runs headless as SYSTEM.
ping -n 31 127.0.0.1 >nul
goto waitloop

:aborted
echo CHAIN ABORTED after 60 minutes -- FMCOARSE never wrote its completion marker.
echo Nothing was fitted. Investigate out\fmcoarse.log before rerunning.
exit /b 2

:ready
echo FMCOARSE finished, starting the chain.

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --val-season 2024 --p1
set CELL=--feat-skill --max-train-season 2024 --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3,4,5,6,8,13

REM ONE control pair serves both experiments. SK40's control and ICOH's control
REM are the SAME command -- champion recipe at --std-k 80 -- and they run in the
REM same session on the same host, which is exactly what the fresh-control
REM contract requires. Fitting them twice would be 8 identical wasted fits.
echo ==== shared control base ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 80 --depth 8 %SEEDS% --tag CHAINCTL_base
echo ==== shared control cell ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 80 %CELL% %SEEDS% --tag CHAINCTL_cell

echo ==== SK40 candidate base (std-k 40) ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 40 --depth 8 %SEEDS% --tag SK40_base
echo ==== SK40 candidate cell (std-k 40) ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 40 %CELL% %SEEDS% --tag SK40_cell
echo ==== SK40 done ====

echo ==== ICOH candidate base ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 80 --feat-id-cohort --depth 8 %SEEDS% --tag ICOH_base
echo ==== ICOH candidate cell ====
%PYU% src/train_gbdt2.py %COMMON% --std-k 80 --feat-id-cohort %CELL% %SEEDS% --tag ICOH_cell
echo ==== ICOH done ====
echo ==== CHAIN COMPLETE ====

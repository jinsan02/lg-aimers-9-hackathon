@echo off
REM P0 smoke. Uses val2022 -> test2023 on purpose: that is the surface the
REM future-season bug contaminated, so the new cutoff must announce itself.
REM Short iterations -- this checks correctness, not score.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set CORE=--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.05 --iters 150 --es 60 --l2 10 --refit-mult 1.5 --drop-f-pre 2022 --val-season 2022 --test-season 2023 --seed 3

%PY% src\train_gbdt2.py %CORE% --depth 8 --tag P0SMOKE_base > out\P0SMOKE_base.log 2>&1
echo %ERRORLEVEL% > out\P0SMOKE_base.exit

%PY% src\train_gbdt2.py %CORE% --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --tag P0SMOKE_cell > out\P0SMOKE_cell.log 2>&1
echo %ERRORLEVEL% > out\P0SMOKE_cell.exit

echo DONE > out\P0SMOKE.done

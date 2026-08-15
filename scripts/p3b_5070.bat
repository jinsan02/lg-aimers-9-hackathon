@echo off
REM P3-B -- one common tree budget for every base seed.
REM Pre-registered in docs/P3B_PREREGISTRATION_20260815.md. Budget 803 = floor
REM of the median of B1J6_base's val2023 stopping points (600,782,799,807,854,
REM 874). It is NOT recomputed from this session: the control arm's new picks
REM are a diagnostic only.
REM
REM Twelve fits in one session on one host, because B1J6_base cannot be reused
REM as the control -- season_std.add_std could not be shown numerically
REM identical to the commit it ran under, and one unproven link is enough.
REM
REM CONTROL   B1J6_base's exact recipe under the current code, normal early
REM           stopping, refit-mult 1.5.
REM CANDIDATE identical except the deployment refit is scaled from 803 for
REM           every seed -> _refit_trees(803, 1.5) = 1206 trees.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8 --seeds 3,4,5,6,8,13

%PYU% src\train_gbdt2.py %CORE% --tag P3BCTL_base > out\P3BCTL_base.log 2>&1
echo %ERRORLEVEL% > out\P3BCTL_base.exit

%PYU% src\train_gbdt2.py %CORE% --common-budget 803 --tag P3BCAND_base > out\P3BCAND_base.log 2>&1
echo %ERRORLEVEL% > out\P3BCAND_base.exit

echo DONE > out\P3B.done

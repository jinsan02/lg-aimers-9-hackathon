@echo off
REM BND21/BND22 -- clean bounded boundary arrays for 2021->2022 and 2022->2023.
REM Pre-registered in docs/BND_PREREGISTRATION_20260816.md.
REM
REM These are ARTIFACTS, not a candidate. FLAG `human-baseball three-transition
REM gate` downgraded seven verdicts because both of these boundaries rest on
REM arrays in docs/INVALIDATED.tsv and the clean alternatives never existed.
REM One GPU investment serves seven axes.
REM
REM --max-train-season goes on BOTH arms here, unlike the 2024-boundary scripts
REM where it is a no-op. On the 2022 boundary it removes 2023 and 2024 from the
REM deployment refit; omitting it is the defect that invalidated ten rolling
REM runs on 2026-08-13.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --p1
set CELL=--feat-skill --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3,4,5,6,8,13

echo ==== BND22 base (val 2022, train ^<= 2022) ====
%PYU% src/train_gbdt2.py %COMMON% --val-season 2022 --max-train-season 2022 --depth 8 %SEEDS% --tag BND22_base
echo ==== BND22 cell ====
%PYU% src/train_gbdt2.py %COMMON% --val-season 2022 --max-train-season 2022 %CELL% %SEEDS% --tag BND22_cell
echo ==== BND22 done ====

echo ==== BND23 base (val 2023, train ^<= 2023) ====
%PYU% src/train_gbdt2.py %COMMON% --val-season 2023 --max-train-season 2023 --depth 8 %SEEDS% --tag BND23_base
echo ==== BND23 cell ====
%PYU% src/train_gbdt2.py %COMMON% --val-season 2023 --max-train-season 2023 %CELL% %SEEDS% --tag BND23_cell
echo ==== BND23 done ====
echo ==== BND COMPLETE ====

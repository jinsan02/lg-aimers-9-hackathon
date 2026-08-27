@echo off
REM LEAFIT n=6 confirmation. Runs ONLY if the seed-3 gate returns >= +3.
REM Pre-registered in docs/NEXT_GPU_PREREGISTRATION_20260828.md.
REM
REM Fresh tags for all three arms rather than reusing the seed-3 artifacts:
REM the overnight contract allows reuse only when lineage proves exact parity,
REM and a single-seed run names its artifacts without the _s<seed> suffix, so
REM the two families cannot be mixed in one paired series without ambiguity.
REM
REM Same judging surface, same host, one session. 18 fits.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PYU=C:\aimers\.venv\Scripts\python.exe -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --p1 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024
set CELL=--feat-skill --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3,4,5,6,8,13

echo ==== LEAFIT6 shared base ====
%PYU% src/train_gbdt2.py %COMMON% --depth 8 %SEEDS% --tag LEAFIT6_base
echo ==== LEAFIT6 control cell ====
%PYU% src/train_gbdt2.py %COMMON% %CELL% %SEEDS% --tag LEAFIT6CTL_cell
echo ==== LEAFIT6 candidate cell (--cell-leaf-iters 10) ====
%PYU% src/train_gbdt2.py %COMMON% %CELL% --cell-leaf-iters 10 %SEEDS% --tag LEAFIT6_cell
echo ==== LEAFIT6 COMPLETE ====

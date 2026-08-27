@echo off
REM LEAFIT seed-3 gate: cell-arm leaf_estimation_iterations 1 -> 10.
REM Pre-registered in docs/NEXT_GPU_PREREGISTRATION_20260828.md (see the
REM AMENDMENT block: judging surface, laptop host).
REM
REM Judging surface -- fit <= 2022, val 2023, UNTOUCHED 2024 -- because that is
REM the only surface with a season the model has never seen, and the overnight
REM contract makes the untouched-season transition the primary adoption
REM evidence. --drop-f-pre 2022 is required here: BND23 omitted it and its base
REM members showed the banned 11-19 best-iteration collapse.
REM
REM ONE shared base arm fixes the core. --cell-leaf-iters cannot reach a depth-8
REM binary run, so refitting the base for the candidate would fold GPU
REM nondeterminism into the headline -- the defect cell_arm_delta.py exists to
REM remove.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PYU=C:\aimers\.venv\Scripts\python.exe -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --p1 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024
set CELL=--feat-skill --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
set SEEDS=--seeds 3

echo ==== LEAFIT shared base (fixes the core) ====
%PYU% src/train_gbdt2.py %COMMON% --depth 8 %SEEDS% --tag LEAFIT_base
echo ==== LEAFIT control cell (leaf_estimation_iterations = CatBoost default) ====
%PYU% src/train_gbdt2.py %COMMON% %CELL% %SEEDS% --tag LEAFITCTL_cell
echo ==== LEAFIT candidate cell (--cell-leaf-iters 10) ====
%PYU% src/train_gbdt2.py %COMMON% %CELL% --cell-leaf-iters 10 %SEEDS% --tag LEAFIT_cell
echo ==== LEAFIT COMPLETE ====

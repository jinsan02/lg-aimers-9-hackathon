@echo off
REM D12 seed 3 -- cell checkpoint chosen by the fixed-core Brier.
REM Pre-registered in docs/D12_PREREGISTRATION_20260815.md. Grid step 25, tie
REM inside 0.05 BSS takes the smaller iteration. Selection on 2023 only; the
REM untouched 2024 season is scored once, afterwards, and never used to choose.
REM Single change: only the iteration the refit is scaled from. Base predictions
REM are B1SMOKE_base's, reused byte-for-byte and joined by row_id.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1
set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005

%PYU% src\train_gbdt2.py %CORE% %CELL% --d12-base-preds out\cat_B1SMOKE_base_val_preds.npz --d12-step 25 --seed 3 --tag D12_cand > out\D12_cand.log 2>&1
echo %ERRORLEVEL% > out\D12_cand.exit
echo DONE > out\D12_cand.done

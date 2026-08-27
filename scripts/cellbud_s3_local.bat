@echo off
REM CELLBUD: let early stopping actually select the cell arm's length.
REM Pre-registered in docs/OVERNIGHT_FALLBACK_PREREGISTRATION_20260828.md.
REM
REM Across 263 cell rows in LEDGER.tsv, best_iter piles against the --iters 3000
REM ceiling -- 2999 appears 37 times and the 2990-2999 band holds 164. --es 500
REM has never fired on this arm. The only two runs given more budget reached
REM 3928 and 4533, and both are pre-B1S single-seed rows on other hosts.
REM
REM 8000 rather than 5000 because 4533 + 500 > 5000, so 5000 may itself truncate.
REM The validity condition is that early stopping FIRES: best_iter + 500 < 8000.
REM
REM ONE fit. Base and control are reused from the LEAFIT session on this host
REM today; lineage parity is asserted before scoring.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PYU=C:\aimers\.venv\Scripts\python.exe -u

set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --p1 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024
set CELL=--feat-skill --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005

echo ==== CELLBUD candidate cell (--iters 8000) ====
%PYU% src/train_gbdt2.py %COMMON% --iters 8000 %CELL% --seeds 3 --tag CELLBUD_cell
echo ==== CELLBUD COMPLETE ====

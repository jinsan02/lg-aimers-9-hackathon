@echo off
REM FM_MULTILABEL_V2 -- the first modern-valid measurement of the multilabel
REM supervision geometry. One seed, diagnostic gate only.
REM
REM Why it is second and not first. The attribution map showed base and cell
REM route the same information differently (block spearman +0.943 on the
REM champion's own taxonomy), so the open question is the objective, not another
REM feature family. The ranking objective was the primary attempt and is blocked
REM by a CatBoost defect, not by its merits. This is the other supervision
REM geometry: MultiLogloss over [control_success, middle, ball, reverse], four
REM independent sigmoids sharing one tree ensemble, P(success) read off head 0.
REM
REM What V2 repairs, both label defects, neither swept:
REM   * partition-safe recovery -- the old path called _pitch_labels on the whole
REM     frame so the last fit rows differenced into the validation season
REM     (113 rows on this surface).
REM   * complete-case training -- the old path ran unrecoverable auxiliary labels
REM     through nan_to_num(..., 0), asserting "did not happen" for "unknown".
REM     MultiLogloss has no per-head mask, so a row trains only when all three
REM     auxiliary labels are observed: 99.849% of rows, 1,691 dropped.
REM Head 0 is exact on every row either way -- it is read from the target column,
REM never differenced -- and every validation row is scored, including the ones
REM held out of training, so the delta shares its denominator with base and cell.
REM
REM Old CZ_ml numbers are not used: no same-era/same-host/same-surface baseline
REM exists for them. This pairs with B1SMOKE_base / B1SMOKE_cell seed 3, the B1S8
REM recipe on exactly this surface, on this host.
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set PY=C:\aimers\.conda\python.exe
set PYU=%PY% -u

set CORE=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 5

%PYU% src\train_gbdt2.py %CORE% --fm-multilabel --seed 3 --tag ML2_s3 > out\ML2_s3.log 2>&1
echo %ERRORLEVEL% > out\ML2_s3.exit

echo DONE > out\ML2_s3.done

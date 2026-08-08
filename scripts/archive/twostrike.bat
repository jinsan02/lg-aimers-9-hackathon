@echo off
REM E166 -- 2-strike segment model + routing.
REM Loss map: 0-2 MSE .24925 / 1-2 .24842 vs overall .24761 -> ceiling +73.
REM Filter is applied AFTER feature build, so TE/std/skill tables stay full-data;
REM only the fitted rows change. Routing key is the row's own column -> row-independent.
REM TS_base is the same config unfiltered, re-run so both sides carry row_id.
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --tag TS_base
echo TSBASE_DONE
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --row-filter "strikes_before==2" --tag TS_only
echo TS_DONE

@echo off
REM 미학습 표면(val 2023 -> 미학습 2024) 재판정 A조 — 4070
REM 기준선: RN1.5 = 868.56 (6시드 평균)
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --drop-cols season --tag SA_nosea
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --drop-cols season,season_progress --tag SA_nosea2
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 40 --tag SA_k40
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --depth 7 --tag SA_d7
echo SURF_A_DONE

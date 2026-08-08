@echo off
REM 미학습 표면 축 재판정 — A100 chain.sh 와 **겹치지 않는** 축만.
REM chain: SC_nosea SC_nosea2 SC_k40 SC_k120 SC_tek100 SC_d7
REM 기준선 RN1.5 = 868.56 (6시드)
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --lr 0.02 --tag SD_lr02
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --depth 9 --tag SD_d9
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --te-k 25 --tag SD_tek25
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 200 --tag SD_k200
echo SURF4070_DONE

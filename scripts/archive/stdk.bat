@echo off
REM std-k 축 정밀 스윕 — SC_k40 이 +15.05 (t=+11.88) 로 채택됐고 단조 경향이다.
REM   k 40 : +15.05   /   k 80 (현행) : 기준   /   k 200 : -19.98
REM 더 낮은 쪽에 최적이 있을 수 있어 10/20/30/60 을 본다.
REM std_k 는 E99 시즌내 복원의 수축 강도다. k 가 작을수록 당해 시즌 추정을 더 믿는다.
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 10 --tag SK_k10
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 20 --tag SK_k20
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 30 --tag SK_k30
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 60 --tag SK_k60
echo STDK_DONE

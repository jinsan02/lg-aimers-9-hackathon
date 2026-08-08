@echo off
REM E158 — v16 이 -6.15 로 떨어진 원인 검정: **구체제 F리그 데이터**
REM
REM 편향 시계열(BI/RN)은 전부 --drop-f-pre 2022 로 쟀다. 조기종료가 F 체제변화를
REM 가로지르면 15 iter 에서 죽기 때문이었다. 그런데 **제출 모델은 그 플래그를 안 쓴다.**
REM 즉 편향을 잰 모델과 제출한 모델의 학습 데이터가 다르다 — 어제 스스로 적어둔
REM 결함인데 그대로 상수를 박았다.
REM
REM F는 전체의 11%이고 구체제(2019~2022) 성공률이 0.71, 신체제가 0.47 이다.
REM 구체제 F 를 학습에 넣으면 신체제 F 행을 0.03 만 과대예측해도 전역 편향이
REM 0.11 x 0.03 = +0.0033 이 된다. LB 역산으로 나온 2025 편향 +0.0041 과 맞는다.
REM
REM 조기종료 인공물을 피하려고 **고정 반복수**로 두 조건을 공평하게 비교한다.
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --depth 8 --l2 10 --refit-mult 1.5
set V=--val-season 2023 --test-season 2024 --iters 1200 --es 5000
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --drop-f-pre 2022 --tag FL_drop
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --tag FL_keep
echo FL_DONE

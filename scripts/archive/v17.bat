@echo off
REM E158b — 구체제 F리그를 학습에서 빼고 제출 구조 그대로 재빌드.
REM
REM FL 실험: 구체제 F 를 넣으면 미학습 시즌 편향이 +0.0027 -> +0.0124 로 벌어진다.
REM 우리 제출은 그 F 를 넣고 있고, LB 역산 편향 +0.0041 이 그것으로 설명된다.
REM SHIFT 로 가리는 대신 **원인을 뺀다**. 성공하면 SHIFT=0 이 비로소 옳아진다.
REM
REM 구조는 제출과 동일: val 2024 로 best_iter -> 전체(<=2024) 재학습 x1.5
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --depth 8 --tag v17f --seeds 42,7,13,3,4,5,6,8
echo V17F_DONE
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --depth 5 --iters 3000 --failmode-cells --tag v17c --seeds 42,7,13,3,4,5
echo V17C_DONE

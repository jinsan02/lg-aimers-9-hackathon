@echo off
REM (1) F리그 전용 모델 — 손실 지도에서 F 가 R 보다 -301 이다.
REM     라우팅 검정: 2024 F 행에서 공유 모델 vs F 전용 모델 중 누가 나은가.
REM     --drop-f-pre 2022 로 구체제 F 를 빼면 학습은 2023 F 만(약 6만행) 남는다.
REM     제출 구조(val 2024)라 그대로 제출에 옮길 수 있다.
REM (2) refit 배수 2.0 — 어제 +2.44 (t=1.91) 로 보류. 시드 6개 추가해 판정한다.
cd /d C:\aimers
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --league F --drop-f-pre 2022 --tag FO_f --seeds 42,7,13,3,4,5
echo FONLY_DONE
set V=--val-season 2023 --test-season 2024 --drop-f-pre 2022
set S=--seeds 31,32,33,34,35,36
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --refit-mult 1.5 --tag RS1.5
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --refit-mult 2.0 --tag RS2.0
echo NIGHT2_DONE

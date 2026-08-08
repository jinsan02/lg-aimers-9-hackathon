@echo off
REM F리그 전용 모델은 죽었다 — --league F --drop-f-pre 2022 로 학습이 55,696행만
REM 남고 TE 결측 71.5%, skill 추정기가 적합할 시즌이 없어 matmul 오류로 종료.
REM 데이터가 없어서 못 하는 것이라 재시도해도 같다. 대신 확실한 두 개를 돌린다.
REM
REM (1) 셀 멤버 시드 6 -> 8. 제출 구조(val 2024). 가중 0.55 로 셀이 절반 이상을
REM     지고 있는데 시드가 base(8)보다 적다. 앙상블 분산만 줄이는 확실한 이득.
REM (2) te-k 100 — A100 에서 +4.27 (t=1.53) 로 보류였다. **같은 머신** 4070 에서
REM     재현되는지 본다. 기준선 RN1.5 878.83.
cd /d C:\aimers
:wait
tasklist /fi "imagename eq python.exe" | find "python.exe" > nul
if not errorlevel 1 (timeout /t 60 /nobreak > nul & goto wait)
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --l2 10 --refit-mult 1.5
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% --depth 5 --iters 3000 --failmode-cells --tag ZD5 --seeds 6,8
echo ZD5SEED_DONE
set V=--val-season 2023 --test-season 2024 --drop-f-pre 2022
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% --depth 8 --te-k 100 --tag TQ_k100 --seeds 3,4,5,6,8,13
echo NIGHT3_DONE

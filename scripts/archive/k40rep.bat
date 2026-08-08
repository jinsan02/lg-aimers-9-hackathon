@echo off
REM ★ 최우선 — SC_k40 의 +15.05 를 **같은 머신에서** 재현한다.
REM
REM std-k 응답곡선이 매끄럽지 않다:
REM   k10 +2.86 / k20 +3.21 / k30 -0.83 / **k40 +15.05** / k120 +0.13 / k200 -19.98
REM 수축 상수는 매끄럽게 반응해야 하는데 k40 만 홀로 솟았다.
REM 결정적 차이: **SC_k40 은 A100, 나머지 SK_* 와 기준선 RN1.5 는 4070** 에서 돌았다.
REM 같은 시드라도 GPU 아키텍처가 다르면 CatBoost 결과가 달라진다(기록된 교훈).
REM 여기서 4070 으로 k40 을 다시 돌려 +15 가 남는지 본다. 안 남으면 v19 게이트도
REM 못 넘을 것이고, std-k 이득은 없던 일이 된다.
cd /d C:\aimers
:wait
tasklist /fi "imagename eq python.exe" | find "python.exe" > nul
if not errorlevel 1 (timeout /t 60 /nobreak > nul & goto wait)
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --std-k 40 --tag SK_k40
echo K40REP_DONE
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --te-k b:500,*:50 --tag TK_b500
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --te-k b:500,pc:110,ph:75,p:90,pi:90 --tag TK_rel
echo TEK_DONE

@echo off
REM E161 — TE 축별 수축(te_k)을 신뢰도 값으로 바꿔본다.
REM
REM 신뢰도로 재본 최적 k (적률법/반분법):
REM   투수 91/381 · 투수x카운트 112/209 · 투수x타자손 75/206 · 타자 556/2414
REM 우리는 전 축에 50 을 쓴다. **타자 축이 6~11배 덜 수축돼 있다** —
REM 제구는 투수의 일이라 타자 정체성에 실린 신호가 거의 없다는 뜻이다.
REM 기준선은 RN1.5(te-k 50, std-k 80) = 878.83.
REM
REM 앞 작업(stdk)이 끝날 때까지 기다린다. schtasks 는 동시 발화 사고가 있었으므로
REM 예약이 아니라 **폴링**으로 줄을 세운다.
cd /d C:\aimers
:wait
tasklist /fi "imagename eq python.exe" | find "python.exe" > nul
if not errorlevel 1 (timeout /t 60 /nobreak > nul & goto wait)
set B=--feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --drop-f-pre 2022
set V=--val-season 2023 --test-season 2024
set S=--seeds 3,4,5,6,8,13
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --te-k b:500,*:50 --tag TK_b500
.venv\Scripts\python.exe src\train_gbdt2.py --model cat %B% %V% %S% --te-k b:500,pc:110,ph:75,p:90,pi:90 --tag TK_rel
echo TEK_DONE

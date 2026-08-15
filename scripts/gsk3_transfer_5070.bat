@echo off
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
set SHARED=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --drop-f-pre 2021 --max-train-season 2023 --val-season 2022 --test-season 2023 --p1 --seeds 3,4,5,6,8,13

%PY% src\train_gbdt2.py %SHARED% --depth 8 --tag GSK3CTL_base > out\GSK3CTL_base.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\GSK3CTL_base.exit
if not "%RC%"=="0" exit /b %RC%

set CELL=--depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005
%PY% src\train_gbdt2.py %SHARED% %CELL% --tag GSK3CTL_cell > out\GSK3CTL_cell.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\GSK3CTL_cell.exit
if not "%RC%"=="0" exit /b %RC%

%PY% src\train_gbdt2.py %SHARED% %CELL% --feat-skill --tag GSK3CAND_cell > out\GSK3CAND_cell.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\GSK3CAND_cell.exit
if not "%RC%"=="0" exit /b %RC%

(echo DONE)>out\GSK3.done
exit /b 0


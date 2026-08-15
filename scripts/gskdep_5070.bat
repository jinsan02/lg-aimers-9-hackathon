@echo off
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --feat-skill --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --loss Logloss --eval-metric Logloss --refit-mult 1.5 --max-train-season 2024 --val-season 2024 --p1 --depth 5 --failmode-cells --fm-modes middle,ball,reverse --fm-min-share 0.005 --seeds 3,4,5,6,8,13

%PY% src\train_gbdt2.py %COMMON% --tag GSKDEP_cell > out\GSKDEP_cell.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\GSKDEP_cell.exit
if not "%RC%"=="0" exit /b %RC%
(echo DONE)>out\GSKDEP.done
exit /b 0

@echo off
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
set COMMON=--model cat --feat-v2 --feat-k 200 --te p,pc,ph,b,pi --te-k 50 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --iters 3000 --es 500 --l2 10 --border-count 254 --refit-mult 1.5 --drop-f-pre 2022 --max-train-season 2024 --val-season 2023 --test-season 2024 --p1 --depth 8 --seeds 3,4,5,6,8,13

%PY% src\train_gbdt2.py %COMMON% --loss Logloss --eval-metric Logloss --tag RMSE2CTL_base > out\RMSE2CTL_base.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\RMSE2CTL_base.exit
if not "%RC%"=="0" exit /b %RC%

%PY% src\train_gbdt2.py %COMMON% --loss RMSE --eval-metric RMSE --tag RMSE2CAND_base > out\RMSE2CAND_base.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\RMSE2CAND_base.exit
if not "%RC%"=="0" exit /b %RC%

(echo DONE)>out\RMSE2.done
exit /b 0

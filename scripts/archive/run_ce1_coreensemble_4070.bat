@echo off
cd /d C:\aimers
set PYTHONIOENCODING=utf-8
set BASE=--model cat --feat-v2 --te p,pc,ph,b,pi --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --es 500 --depth 8 --l2 10 --refit-mult 1.5 --val-season 2024 --seeds 7,13,3,4,5,6,8
.venv\Scripts\python.exe tools\precheck.py %BASE% --lr 0.005 --tag CE1V_lr5
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe src\train_gbdt2.py %BASE% --lr 0.005 --tag CE1V_lr5
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe tools\precheck.py %BASE% --lr 0.01 --random-strength 0.5 --tag CE1V_rs05
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe src\train_gbdt2.py %BASE% --lr 0.01 --random-strength 0.5 --tag CE1V_rs05
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe tools\precheck.py %BASE% --lr 0.01 --random-strength 2 --tag CE1V_rs2
if errorlevel 1 exit /b %errorlevel%
.venv\Scripts\python.exe src\train_gbdt2.py %BASE% --lr 0.01 --random-strength 2 --tag CE1V_rs2

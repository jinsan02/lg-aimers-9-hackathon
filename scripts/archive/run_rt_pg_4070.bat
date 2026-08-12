@echo off
cd /d C:\aimers
set PYTHONUTF8=1
set P=.venv\Scripts\python.exe
set C=--model cat --feat-v2 --te-dev --feat-std --std-k 80 --std-to-prior --std-season-prior --feat-domain --feat-skill-pc --lr 0.01 --es 500 --depth 8 --l2 10 --refit-mult 1.5 --val-season 2024 --seed 42

%P% src\train_gbdt2.py %C% --te p,pc,ph,b,pi --feat-roster --tag RT1V_roster
if errorlevel 1 exit /b %errorlevel%
%P% src\train_gbdt2.py %C% --te p,pc,ph,b,pi,pg --tag PG1V_pg
if errorlevel 1 exit /b %errorlevel%

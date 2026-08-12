@echo off
cd /d C:\aimers
if not exist out mkdir out
if not exist model mkdir model
set PYTHONIOENCODING=utf-8
C:\aimers\.conda\python.exe src\train_tabdecoder.py --tag TDEC1 --seed 42 --dim 96 --layers 3 --heads 8 --batch 2048 --eval-batch 8192 --epochs 15 --patience 3 > out\TDEC1_5070.log 2>&1
echo %ERRORLEVEL% > out\TDEC1_5070.exit

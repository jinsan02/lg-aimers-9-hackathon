@echo off
cd /d C:\aimers
C:\aimers\.conda\python.exe tools\f_adapter_gate.py --base B1SMOKE_base --cell B1SMOKE_cell > out\f1_gate.log 2>&1
set RC=%errorlevel%
(echo %RC%)>out\f1_gate.exit
exit /b %RC%

@echo off
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
%PY% tools\member_fingerprint.py --verbose GSK2CTL_cell
if errorlevel 1 exit /b %errorlevel%
%PY% tools\member_fingerprint.py --verbose GSK2CAND_cell
if errorlevel 1 exit /b %errorlevel%
%PY% tools\gsk2_gate.py
exit /b %errorlevel%

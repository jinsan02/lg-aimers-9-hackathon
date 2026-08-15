@echo off
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
%PY% tools\member_fingerprint.py --verbose GSK3CTL_base GSK3CTL_cell GSK3CAND_cell
if errorlevel 1 exit /b %errorlevel%
%PY% tools\gsk2_gate.py --control GSK3CTL_cell --candidate GSK3CAND_cell --base GSK3CTL_base --report out/gsk3_gate.json
exit /b %errorlevel%

@echo off
cd /d C:\aimers
C:\aimers\.conda\python.exe tools\member_fingerprint.py --verbose RMSE2CTL_base RMSE2CAND_base
if errorlevel 1 exit /b %errorlevel%
C:\aimers\.conda\python.exe tools\rmse2_gate.py --control RMSE2CTL_base --candidate RMSE2CAND_base --cell B1J6_cell
exit /b %errorlevel%

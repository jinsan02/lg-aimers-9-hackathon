@echo off
cd /d C:\aimers
C:\aimers\.conda\python.exe tools\member_fingerprint.py --verbose B1SMOKE_base P3C2CTL_cell P3C2CAND_cell
if errorlevel 1 exit /b %errorlevel%
C:\aimers\.conda\python.exe tools\p3c2_artifact_audit.py P3C2CTL_cell P3C2CAND_cell
exit /b %errorlevel%

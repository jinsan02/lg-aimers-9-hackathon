@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
.venv\Scripts\python.exe tools\verify_submission.py submissions\v10_middle_fix_0809.zip > out\verify_v10fix.log 2>&1
echo EXITCODE=%ERRORLEVEL%
type out\verify_v10fix.log
exit /b %ERRORLEVEL%

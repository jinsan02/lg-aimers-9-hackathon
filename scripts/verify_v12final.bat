@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
.venv\Scripts\python.exe tools\verify_submission.py submissions\v12_career_pb_0809.zip > out\verify_v12final.log 2>&1
echo VERIFY_EXIT=%ERRORLEVEL%
type out\verify_v12final.log
exit /b %ERRORLEVEL%

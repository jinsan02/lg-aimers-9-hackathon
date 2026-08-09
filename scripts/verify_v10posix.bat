@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
.venv\Scripts\python.exe tools\verify_submission.py submissions\v10_middle_posix_0809.zip > out\verify_v10posix.log 2>&1
echo EXITCODE=%ERRORLEVEL%
type out\verify_v10posix.log
exit /b %ERRORLEVEL%

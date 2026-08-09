@echo off
cd /d C:\aimers
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
.venv\Scripts\python.exe tools\verify_submission.py submissions\v11_pb_posix_0809.zip > out\verify_v11pb.log 2>&1
echo VERIFY_EXIT=%ERRORLEVEL%
type out\verify_v11pb.log
exit /b %ERRORLEVEL%

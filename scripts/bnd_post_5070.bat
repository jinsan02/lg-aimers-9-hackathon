@echo off
REM CPU-only post-BND supervisor. It never starts a trainer and waits for the
REM pre-registered runner's explicit completion marker before touching packs.
cd /d C:\aimers
set PY=C:\aimers\.conda\python.exe
set MARKER===== BND COMPLETE ====

:WAIT_BND
findstr /c:"%MARKER%" C:\aimers\out\bnd.log >nul 2>&1
if errorlevel 1 (
  timeout /t 30 /nobreak >nul
  goto WAIT_BND
)

%PY% tools\bnd_integrity_report.py --report out\bnd_integrity.json > out\bnd_integrity.log 2>&1
if errorlevel 1 (
  echo INTEGRITY_FAILED>out\bnd_post.failed
  exit /b 2
)

%PY% tools\bnd_materialize_next.py > out\bnd_materialize.log 2>&1
if errorlevel 1 (
  echo MATERIALIZE_FAILED>out\bnd_post.failed
  exit /b 2
)

echo DONE>out\bnd_post.done
exit /b 0

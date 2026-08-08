@echo off
REM Use Git Bash explicitly because bash.exe may resolve to WSL on Windows.
setlocal
set "GB=C:\Program Files\Git\bin\bash.exe"
if not exist "%GB%" set "GB=C:\Program Files\Git\usr\bin\bash.exe"
if not exist "%GB%" (
  echo Git Bash was not found. Check the configured paths.
  exit /b 1
)
"%GB%" "%~dp0agent_sync.sh" %*

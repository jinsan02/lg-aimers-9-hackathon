@echo off
REM Windows 에서 `bash` 가 WSL 로 잡히면 agent_sync.sh 가 E_ACCESSDENIED 로 죽는다
REM (2026-08-08 Codex 세션에서 발생). Git Bash 를 직접 지정한다.
setlocal
set GB=C:\Program Files\Git\bin\bash.exe
if not exist "%GB%" set GB=C:\Program Files\Git\usr\bin\bash.exe
if not exist "%GB%" (
  echo Git Bash 를 못 찾았다. 경로를 확인할 것.
  exit /b 1
)
"%GB%" "%~dp0agent_sync.sh" %*

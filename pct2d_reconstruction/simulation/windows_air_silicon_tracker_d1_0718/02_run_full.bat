@echo off
setlocal
cd /d "%~dp0"
set WORKERS=%~1
if "%WORKERS%"=="" set WORKERS=12
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_parallel.ps1" -Workers %WORKERS%
if errorlevel 1 (
  echo Some angles failed. Run the same command again to retry.
  exit /b 1
)
echo D1 full simulation PASS.

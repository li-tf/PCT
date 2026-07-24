@echo off
setlocal
cd /d "%~dp0"
set WORKERS=%~1
if "%WORKERS%"=="" set WORKERS=4
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_parallel.ps1" -Workers %WORKERS%
if errorlevel 1 (
  echo.
  echo Some angles failed. Run this command again to retry incomplete angles.
  pause
  exit /b 1
)
echo.
echo Full simulation and integrated completeness check finished.
pause

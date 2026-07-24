@echo off
setlocal
cd /d "%~dp0"
python check_environment.py
if errorlevel 1 (
  echo.
  echo Environment check failed.
  pause
  exit /b 1
)
echo.
echo Environment check passed.
pause

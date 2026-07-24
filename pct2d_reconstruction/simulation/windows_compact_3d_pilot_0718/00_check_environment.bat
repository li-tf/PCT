@echo off
setlocal
cd /d "%~dp0"
python check_environment.py
if errorlevel 1 exit /b 1
echo Environment check PASS.

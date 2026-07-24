@echo off
setlocal
cd /d "%~dp0"
python check_environment.py
exit /b %errorlevel%

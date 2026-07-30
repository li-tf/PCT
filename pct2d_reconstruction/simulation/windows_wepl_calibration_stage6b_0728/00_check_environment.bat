@echo off
setlocal
python "%~dp0check_environment.py"
exit /b %errorlevel%

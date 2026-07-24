@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_smoke_tests.ps1"
exit /b %errorlevel%

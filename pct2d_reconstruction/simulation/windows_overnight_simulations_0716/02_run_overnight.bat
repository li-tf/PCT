@echo off
setlocal
cd /d "%~dp0"
set WORKERS=%~1
if "%WORKERS%"=="" set WORKERS=12
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_overnight.ps1" -Workers %WORKERS%
exit /b %errorlevel%

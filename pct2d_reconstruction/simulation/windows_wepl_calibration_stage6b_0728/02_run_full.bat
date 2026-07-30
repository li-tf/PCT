@echo off
setlocal
set WORKERS=%1
if "%WORKERS%"=="" set WORKERS=12
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_parallel.ps1" -Workers %WORKERS%
exit /b %errorlevel%

@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_status.ps1"
exit /b %errorlevel%

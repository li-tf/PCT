@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_smoke_test.ps1"
if errorlevel 1 exit /b 1
echo Stage 6A virtual MLIC smoke test PASS.

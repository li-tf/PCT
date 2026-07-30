@echo off
setlocal
cd /d "%~dp0"
set WORKERS=%~1
if "%WORKERS%"=="" set WORKERS=12
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch_parallel.ps1" ^
  -Workers %WORKERS% ^
  -ConfigPath "%~dp0simulation_config_highstat_200mev.json" ^
  -QcRoot "%~dp0qc\highstat_200mev"
if errorlevel 1 (
  echo Some high-statistics MLIC tasks failed. Run the same command again to retry.
  exit /b 1
)
echo Stage 6A high-statistics 200 MeV simulation PASS.

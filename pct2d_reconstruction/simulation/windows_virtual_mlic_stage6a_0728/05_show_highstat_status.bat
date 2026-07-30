@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0show_status.ps1" ^
  -ConfigPath "%~dp0simulation_config_highstat_200mev.json" ^
  -QcRoot "%~dp0qc\highstat_200mev"

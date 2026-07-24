@echo off
setlocal
cd /d "%~dp0"
if exist smoke_test rmdir /s /q smoke_test
if exist smoke_qc rmdir /s /q smoke_qc
python run_angle.py --angle 0 --protons-per-projection 10000 --output-dir smoke_test --qc-dir smoke_qc
if errorlevel 1 (
  echo.
  echo Smoke test failed.
  pause
  exit /b 1
)
echo.
echo Smoke test completed in smoke_test\ with metadata in smoke_qc\
pause

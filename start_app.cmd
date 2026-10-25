@echo off
setlocal
cd /d "%~dp0"
rem Prefer the native Windows environment; WSL registration is not required.
where py >nul 2>nul
if errorlevel 1 (
  python "scripts\start_windows.py" %*
) else (
  py -3 "scripts\start_windows.py" %*
)
if errorlevel 1 (
  echo.
  echo Startup failed. See submission\START_HERE.md for environment setup.
  pause
)

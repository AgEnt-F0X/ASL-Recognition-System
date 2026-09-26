@echo off
title ASL Recognition System - Setup for New PC
echo ========================================================
echo Setting up ASL Recognition System on this computer...
echo ========================================================
cd /d "%~dp0"

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed on this system!
    echo Please install Python (3.10 - 3.12) from https://www.python.org/downloads/
    echo (Be sure to check "Add Python to PATH" during installation)
    echo.
    pause
    exit /b 1
)

echo [1/3] Creating Python virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
)

echo [2/3] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/3] Installing dependencies...
".venv\Scripts\pip.exe" install -r requirements.txt

echo.
echo ========================================================
echo SETUP COMPLETE!
echo You can now double-click:
echo   - Run_ASL_Desktop.bat (Desktop App with instant webcam)
echo   - Run_ASL_Web.bat     (Browser interface on localhost)
echo ========================================================
pause

@echo off
title ASL Recognition System - Desktop App
echo ========================================================
echo Starting ASL Real-Time Fingerspelling Recognition...
echo ========================================================
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" gui_app.py
) else (
    python gui_app.py
)
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error.
    pause
)

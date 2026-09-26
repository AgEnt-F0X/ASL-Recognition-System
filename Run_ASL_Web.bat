@echo off
title ASL Recognition System - Local Web App
echo ========================================================
echo Starting ASL Real-Time Web App at http://localhost:8501...
echo NOTE: On localhost, live camera video works 100%% with no TURN needed!
echo ========================================================
cd /d "%~dp0"

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run web_app.py
) else (
    streamlit run web_app.py
)
if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error.
    pause
)

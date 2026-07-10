@echo off
title Emotion-to-Art AI Runner
echo ===================================================
echo   Emotion-to-Art AI - Setup and Launcher
echo ===================================================
echo.

:: Get the directory of the batch script
cd /d "%~dp0depi project"

:: Ensure virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Creating Python 3.12 virtual environment...
    py -3.12 -m venv .venv
)

:: Activate the virtual environment
call .venv\Scripts\activate.bat

echo [1/2] Installing/updating project requirements...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements!
    pause
    exit /b %errorlevel%
)

echo.
echo [2/2] Launching Streamlit web application...
echo.
python -m streamlit run app.py

pause

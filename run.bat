@echo off
title Emotion-to-Art AI Runner
echo ===================================================
echo   Emotion-to-Art AI - Setup and Launcher
echo ===================================================
echo.

:: Get the directory of the batch script
cd /d "%~dp0depi project"

echo [1/3] Installing/updating project requirements...
py -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing requirements!
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Installing torchvision...
py -m pip install torchvision
if %errorlevel% neq 0 (
    echo Error installing torchvision!
    pause
    exit /b %errorlevel%
)

echo.
echo [3/3] Launching Streamlit web application...
echo.
py -m streamlit run app.py

pause

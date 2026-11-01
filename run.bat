@echo off
REM ==============================================================================
REM Multi-Camera Vehicle Tracking & Re-Identification System (MCVT)
REM Final Year Project (FYP) Launcher Script for Windows
REM ==============================================================================

title MCVT Vision System

echo ==================================================================
echo    MCVT VISION: Multi-Camera Vehicle Tracking ^& Re-ID System      
echo ==================================================================
echo.

set SCRIPT_DIR=%~dp0
set SYSTEM_DIR=%SCRIPT_DIR%mcvt-system

REM Check PHP
where php >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] PHP detected.
    echo [*] Starting PHP Built-in Server on http://localhost:8080...
    start /b php -S 0.0.0.0:8080 -t "%SYSTEM_DIR%\frontend"
    echo [OK] PHP Dashboard available at: http://localhost:8080/index.php
) else (
    echo [!] PHP is not found in PATH.
    echo     You can install PHP or access the dashboard directly at:
    echo     http://localhost:5000/ui
)

echo.
echo [*] Starting Python AI Vision Core (Flask + YOLOv8 + ReID Engine)...
cd /d "%SYSTEM_DIR%\backend"
python app.py

pause

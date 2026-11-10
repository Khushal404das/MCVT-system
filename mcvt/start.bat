@echo off
setlocal
REM Starts the three MCVT processes in separate windows:
REM   1) PHP WebSocket hub   2) PHP web UI   3) Python AI engine
REM Any MCVT processes already running are stopped first, so the ports are free.
cd /d "%~dp0"

REM ---- find PHP (PATH first, then the winget install) ----
set "PHP="
for /f "delims=" %%i in ('where php 2^>nul') do if not defined PHP set "PHP=%%i"
if not defined PHP if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\PHP.PHP.8.2_Microsoft.Winget.Source_8wekyb3d8bbwe\php.exe" set "PHP=%LOCALAPPDATA%\Microsoft\WinGet\Packages\PHP.PHP.8.2_Microsoft.Winget.Source_8wekyb3d8bbwe\php.exe"
if not defined PHP (
  echo PHP was not found. Install PHP 8.2 or add it to PATH.
  pause & exit /b 1
)

REM ---- find Python (the real install, not the Microsoft Store shortcut) ----
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY for /f "delims=" %%i in ('where python 2^>nul ^| findstr /v /i WindowsApps') do if not defined PY set "PY=%%i"
if not defined PY (
  echo Python was not found. Install Python 3.11 or add it to PATH.
  pause & exit /b 1
)

echo PHP:    %PHP%
echo Python: %PY%

REM ---- stop MCVT processes left over from an earlier start ----
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'ws-server\.php|serve\.php|web/public|run_engine\.py|prepare_streams\.py' -and $_.ProcessId -ne $PID } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host ('Stopped old process ' + $_.ProcessId) }"
ping -n 2 127.0.0.1 >nul

REM ---- start (windows stay open, so any error stays readable) ----
start "MCVT WebSocket hub" cmd /k ""%PHP%" web\bin\ws-server.php"
ping -n 2 127.0.0.1 >nul
start "MCVT web UI" cmd /k ""%PHP%" web\bin\serve.php"
REM prepare_streams.py only converts videos that were not converted before.
start "MCVT AI engine" cmd /k ""%PY%" engine\prepare_streams.py && "%PY%" engine\run_engine.py"

ping -n 5 127.0.0.1 >nul
start "" http://127.0.0.1:8080
echo MCVT is starting: http://127.0.0.1:8080
endlocal

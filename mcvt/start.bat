@echo off
REM Starts the three MCVT processes in separate windows:
REM   1) PHP WebSocket hub   2) PHP web UI   3) Python AI engine
cd /d "%~dp0"
start "MCVT WebSocket hub" php web\bin\ws-server.php
timeout /t 1 /nobreak >nul
start "MCVT web UI" php web\bin\serve.php
REM prepare_streams.py only converts videos that were not converted before.
start "MCVT AI engine" cmd /k "python engine\prepare_streams.py && python engine\run_engine.py"
echo MCVT is starting. Open the URL shown in the "MCVT web UI" window.

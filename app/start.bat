@echo off
rem Start the CATMuS HTR web app, then open the browser.
cd /d "%~dp0.."
.venv\Scripts\python.exe app\app.py
pause

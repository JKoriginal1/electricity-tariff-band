@echo off
REM One-time setup: creates .venv, installs packages, runs tests, then starts the app.
cd /d "%~dp0"
set PY=
where python >nul 2>&1 && set PY=python
if not defined PY where py >nul 2>&1 && set PY=py
if not defined PY (
  echo Python was not found. Install Python 3.11+ from python.org and run this again.
  pause & exit /b 1
)
if not exist .venv\Scripts\python.exe (
  echo [1/3] Creating virtual environment...
  %PY% -m venv .venv
)
echo [2/3] Installing packages (first run takes several minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt pytest httpx
echo [3/3] Running tests...
".venv\Scripts\python.exe" -m pytest -q tests
echo.
echo Setup finished. Starting the app...
call "%~dp0start_app.bat"

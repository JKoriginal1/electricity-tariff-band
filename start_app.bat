@echo off
REM Starts the tariff-band app (run setup_and_run.bat once first).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Packages are not installed yet. Please run setup_and_run.bat first.
  pause & exit /b 1
)
echo Starting the app... your browser will open at http://localhost:8501 in a few seconds.
echo Keep this window open while using the app. Close it to stop the app.
start "" cmd /c "timeout /t 10 /nobreak >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.headless true --browser.gatherUsageStats false --server.port 8501
echo.
echo The app stopped. If you see an error above, take a screenshot.
pause

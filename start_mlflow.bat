@echo off
REM Opens the MLflow dashboard (experiment tracking + model registry) at http://127.0.0.1:5000
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run setup_and_run.bat once first.
  pause
  exit /b 1
)
set MLFLOW_DISABLE_TELEMETRY=true
set MLFLOW_DISABLE_AGENT_HINT=1
echo Installing MLflow if needed...
.venv\Scripts\python -m pip install -q mlflow==3.17.2
if not exist mlops\mlflow.db (
  echo Logging all experiments to MLflow...
  .venv\Scripts\python mlops\track_experiments.py
)
echo Opening the dashboard in your browser in a few seconds. Keep this window open; close it to stop MLflow.
start "" cmd /c "timeout /t 10 >nul & start http://127.0.0.1:5000/#/experiments"
.venv\Scripts\python -m mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db --port 5000

# Next-Month Electricity Tariff Band Forecaster (Sri Lanka)

Applied Machine Learning — individual assignment, MSc in AI.

Predicts which **domestic tariff band** next month's electricity bill will fall in: **Low (≤60 units)**, **Medium (61–180)** or **High (>180)**. The inputs are a household's last 12 bills, the month being predicted and optional home details. The model is an **Explainable Boosting Machine (EBM)**, a glass-box GA²M from InterpretML that was not covered in the lectures and is not deep learning.

| Test set (807 unseen households, 7,249 forecasts) | Accuracy | Macro-F1 | Macro ROC-AUC |
|---|---|---|---|
| **EBM (selected)** | **0.896** | **0.884** | **0.972** |
| CatBoost (best black box) | 0.897 | 0.886 | 0.972 |
| Naive: band of last-3-bill average | 0.877 | 0.865 | 0.890 |

The EBM is statistically tied with RF, XGBoost, LightGBM and CatBoost (paired household bootstrap). It beats the naive rule overall (+1.8 points), and by **+8 points for households within 10 units of a band edge**, which are the households a warning helps most.

## Data
**RECON-SL** — LIRNEasia & LECO (2026), *Residential Electricity Consumption Dataset for Sri Lanka*, IEEE DataPort, doi:10.21227/n1dk-q860. The dataset covers 4,063 LECO households with 25 months of bills (Oct 2022 – Oct 2024) and a household survey.

The dataset's licence is not stated, so the raw files are **not** in the public repo. To get them:
1. Sign in at IEEE DataPort and download RECON-SL.
2. Copy `consumption_data/non_smart_meter/monthly_consumption.csv` and the `survey_data/` wave-1 files into `data/recon_sl/`, keeping the same folder layout. Alternatively, set the environment variable `RECON_SL_DIR`.

## Run (Windows)
```
setup_and_run.bat     # first time: creates .venv, installs requirements, starts the app
start_app.bat         # afterwards: opens http://localhost:8501
```
Manual setup:
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
uvicorn api.main:app            # REST API, docs at http://127.0.0.1:8000/docs
pytest tests                    # 11 tests
jupyter nbconvert --to notebook --execute notebooks/Electricity_Tariff_Band_Forecasting.ipynb   # full retrain (~1.5 h on 2 cores)
```

## MLOps
| Stage | How |
|---|---|
| Experiment tracking + registry | `python mlops/track_experiments.py`, then `mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db` (http://127.0.0.1:5000). Logs every model's params and CV/validation/test metrics, and registers the EBM as `tariff-band-ebm@champion` |
| Serving | Streamlit app + FastAPI service, both loading the same pipeline and the same preprocessing (`src/elec.py`) |
| Deployment | Streamlit Community Cloud (live app link below); Docker image for the API |
| CI | GitHub Actions: tests + Docker build and smoke test on every push |
| Monitoring | PSI data-drift check against the training baseline (`models/drift_reference.json`): in the app's batch tab, or `python mlops/monitor.py batch.csv` (exit 0/1/2 = stable/moderate/major) |
| Retraining | `python mlops/retrain.py` — champion/challenger: promotes the retrained EBM only if it is at least as good, archives the old model, registers a new version (`--dry-run` to only compare) |

**Live app:** _(link)_

## Structure
```
src/elec.py                     shared cleaning, feature engineering and input validation (notebook, app and API)
notebooks/                      main notebook (executed); built from cells.py + interpretations.py
models/                         tariff_band_model.pkl (EBM pipeline), new_household_model.pkl, model_meta.json
app/streamlit_app.py            front end: household forecast, new household, batch CSV, explanations
api/main.py                     FastAPI /predict and /health
mlops/                          track_experiments.py (MLflow), monitor.py (drift), retrain.py (champion/challenger)
src/monitoring.py               PSI drift detection
tests/                          pytest
reports/                        notebook HTML and figures
Dockerfile, .github/workflows/  container and CI (GitHub Actions: tests + Docker smoke test)
```

## Method in brief
- **No leakage:** each sample uses only the 12 bills before the forecast date (Jan–Sep 2024). The split is by household (60/20/20), so test customers are never seen in training.
- **Models:** 9 tuned models plus the naive baseline. Tuning uses RandomizedSearchCV with GroupKFold(3). The EBM is selected if it is statistically tied with the best model.
- **Explainability:** EBM shape functions and importances, PDP/ICE, SHAP (ρ = 0.82 with EBM importance), and LIME vs exact EBM local explanations.
- **Checks:** overfitting, a temporal hold-out, a history-only model, and fairness by socio-economic class, house type and solar.

## Limitations
Only one year of forecast dates is available: on unseen months the model roughly equals persistence. The survey is a single snapshot. The usage what-if is not causal. The model predicts units, not rupees, so the band edges must follow PUCSL tariff revisions.

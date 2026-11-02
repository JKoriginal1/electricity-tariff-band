"""Log every model of the experiment to MLflow and register the selected EBM in the model registry.

Run from the project root:
    python mlops/track_experiments.py
    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db      # then open http://127.0.0.1:5000

Reads the results the notebook saved in models/ (tuning_results.csv, validation_results.csv,
model_comparison.csv, model_meta.json) so the tracked numbers are exactly the reported ones.
"""
from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elec  # noqa: E402

MODELS = ROOT / "models"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlops' / 'mlflow.db').as_posix()}")
EXPERIMENT = "tariff-band-forecasting"
REGISTERED = "tariff-band-ebm"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlops" / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    meta = json.loads((MODELS / "model_meta.json").read_text())
    tune = pd.read_csv(MODELS / "tuning_results.csv").set_index("model")
    val = pd.read_csv(MODELS / "validation_results.csv").set_index("model")
    test = pd.read_csv(MODELS / "model_comparison.csv").set_index("model")
    data_tags = {"dataset": "RECON-SL (LIRNEasia & LECO, 2026)", "doi": "10.21227/n1dk-q860",
                 "split": "by household 60/20/20, seed 42", "horizon_months": str(meta["horizon_months"])}

    for name in test.index:
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**data_tags, "model_family": name, "selected": str(name == meta["model_name"])})
            if name in tune.index:
                params = tune.loc[name, "best_params"]
                params = ast.literal_eval(params) if isinstance(params, str) else {}
                mlflow.log_params({k: v for k, v in params.items()})
                for col in ["cv_accuracy", "cv_sd", "train_accuracy", "fit_time_s"]:
                    if pd.notna(tune.loc[name, col]):
                        mlflow.log_metric(col, float(tune.loc[name, col]))
            for prefix, table in [("val", val), ("test", test)]:
                if name in table.index:
                    mlflow.log_metrics({f"{prefix}_{k}": float(v) for k, v in table.loc[name].items()})
            if name == meta["model_name"]:
                model = joblib.load(MODELS / "tariff_band_model.pkl")
                X_ex = elec.make_input_row([150, 160, 175, 190, 185, 170, 160, 158, 165, 172, 178, 182], 4, {})
                mlflow.sklearn.log_model(model, name="model", signature=infer_signature(X_ex, model.predict_proba(X_ex)),
                                         input_example=X_ex, registered_model_name=REGISTERED,
                                         skops_trusted_types=["interpret.glassbox._ebm._ebm.ExplainableBoostingClassifier"])
                for f in ["model_meta.json", "drift_reference.json"]:
                    if (MODELS / f).exists():
                        mlflow.log_artifact(str(MODELS / f), artifact_path="metadata")
                for f in sorted((ROOT / "reports" / "figures").glob("*.png")):
                    mlflow.log_artifact(str(f), artifact_path="figures")
                ebm_run = run.info.run_id
        print(f"logged {name}")

    client = mlflow.MlflowClient()
    version = max(client.search_model_versions(f"name='{REGISTERED}'"), key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED, "champion", version.version)
    client.set_model_version_tag(REGISTERED, version.version, "test_accuracy", str(meta["test_metrics"]["accuracy"]))
    print(f"registered {REGISTERED} v{version.version} as 'champion' (run {ebm_run})")
    print(f"view: mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()

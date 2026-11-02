"""Champion / challenger retraining.

Retrains the EBM on the latest RECON-SL files with the tuned hyperparameters, evaluates it on the same
household-grouped test split and **promotes it only if it is at least as good as the current model**
(test accuracy not more than 0.2 points lower and macro AUC not lower than champion - 0.005).
Every attempt is logged to MLflow; a promoted model becomes a new registry version with alias 'champion'.

Run from the project root:
    python mlops/retrain.py                 # full retrain (slow: ~10-60 min depending on CPU)
    python mlops/retrain.py --quick         # smoke test with fewer bags (never promoted)
    python mlops/retrain.py --dry-run       # full retrain + comparison, never promoted
    python mlops/retrain.py --data D:/path/to/recon_sl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elec  # noqa: E402
import monitoring  # noqa: E402

MODELS = ROOT / "models"
ACC_TOL, AUC_TOL = 0.002, 0.005


def evaluate(model, X, y) -> dict:
    p = model.predict_proba(X)
    return {"accuracy": round(accuracy_score(y, p.argmax(1)), 4),
            "macro_f1": round(f1_score(y, p.argmax(1), average="macro"), 4),
            "macro_auc": round(roc_auc_score(y, p, multi_class="ovr"), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("RECON_SL_DIR", ROOT / "data" / "recon_sl"))
    ap.add_argument("--quick", action="store_true", help="fast smoke test, never promoted")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="full retrain and comparison, but never promote")
    a = ap.parse_args()

    meta = json.loads((MODELS / "model_meta.json").read_text())
    champion = joblib.load(MODELS / "tariff_band_model.pkl")

    S = elec.load_dataset(a.data)
    train, val, test = elec.split_by_household(S)
    trval = pd.concat([train, val])
    X_test, y_test = test[elec.FEATURES], test["band"].astype(int)
    print(f"data: {len(S):,} samples | refit on train+validation {len(trval):,} | test {len(test):,}")

    challenger = clone(champion)
    if a.quick:
        challenger.set_params(model__outer_bags=2, model__interactions=3, model__max_rounds=500)
    t0 = time.time()
    challenger.fit(trval[elec.FEATURES], trval["band"].astype(int))
    fit_s = round(time.time() - t0)

    champ_m, chall_m = evaluate(champion, X_test, y_test), evaluate(challenger, X_test, y_test)
    promote = (not a.quick and not a.dry_run and chall_m["accuracy"] >= champ_m["accuracy"] - ACC_TOL
               and chall_m["macro_auc"] >= champ_m["macro_auc"] - AUC_TOL)
    print(f"champion   {champ_m}\nchallenger {chall_m}  (fit {fit_s}s)\ndecision: {'PROMOTE' if promote else 'keep champion'}")

    if promote:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        archive = MODELS / "archive"; archive.mkdir(exist_ok=True)
        shutil.copy(MODELS / "tariff_band_model.pkl", archive / f"tariff_band_model_{meta['version']}_{stamp}.pkl")
        major, minor, patch = (int(x) for x in meta["version"].split("."))
        meta.update(version=f"{major}.{minor + 1}.0", retrained_at=stamp,
                    test_metrics={**meta["test_metrics"], **chall_m})
        joblib.dump(challenger, MODELS / "tariff_band_model.pkl")
        (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2))
        monitoring.save_reference(trval[elec.FEATURES], trval["band"], MODELS / "drift_reference.json")
        print(f"promoted as v{meta['version']} (previous model archived)")

    if not a.no_mlflow:
        import mlflow
        from track_experiments import EXPERIMENT, REGISTERED, TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlops" / "mlartifacts").as_uri())
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"retrain {'quick' if a.quick else 'dry-run' if a.dry_run else 'full'}"):
            mlflow.set_tags({"type": "retrain", "decision": "promote" if promote else "keep"})
            mlflow.log_params({k.replace("model__", ""): v for k, v in challenger.get_params().items()
                               if k.startswith("model__") and k[7:] in {*meta["best_params"], "max_rounds"}})
            mlflow.log_metrics({f"challenger_{k}": v for k, v in chall_m.items()} |
                               {f"champion_{k}": v for k, v in champ_m.items()} | {"fit_time_s": fit_s})
            if promote:
                mlflow.sklearn.log_model(challenger, name="model", registered_model_name=REGISTERED,
                                         skops_trusted_types=["interpret.glassbox._ebm._ebm.ExplainableBoostingClassifier"])
                client = mlflow.MlflowClient()
                v = max(client.search_model_versions(f"name='{REGISTERED}'"), key=lambda m: int(m.version))
                client.set_registered_model_alias(REGISTERED, "champion", v.version)
    sys.exit(0)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()

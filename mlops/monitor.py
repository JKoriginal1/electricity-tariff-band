"""Drift check for a batch of households (same CSV format as the app's batch mode).

    python mlops/monitor.py new_households.csv [--out drift_report.csv]

Builds the model features for every valid row, compares them with the training baseline
(models/drift_reference.json) using PSI, and also compares the predicted band mix with the training band mix.
Exit code 0 = stable, 1 = moderate shift, 2 = major shift (retrain recommended) — usable from a scheduler or CI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elec  # noqa: E402
import monitoring  # noqa: E402


def batch_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hh_cols = [c for c in df.columns if c in elec.HOUSEHOLD_DEFAULTS]
    for r in df.to_dict("records"):
        bills = [r[f"bill_{i}"] for i in range(1, 13)]
        home = {c: r[c] for c in hh_cols if pd.notna(r[c])}
        if elec.validate_inputs(bills, home):
            continue
        rows.append(elec.make_input_row(bills, int(r["target_month"]), home))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=elec.FEATURES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    X = batch_features(df)
    if X.empty:
        sys.exit("no valid rows")
    ref = monitoring.load_reference(ROOT / "models" / "drift_reference.json")
    rep = monitoring.drift_report(X, ref, supplied=[c for c in df.columns if c in elec.HOUSEHOLD_DEFAULTS])
    model = joblib.load(ROOT / "models" / "tariff_band_model.pkl")
    shift = monitoring.prediction_shift(model.predict(X), ref)
    print(rep.to_string(index=False))
    print(f"\nrows checked: {len(X)} | predicted-band-mix PSI: {shift:.3f} ({monitoring.status(shift)})")
    if len(X) < 200:
        print("note: fewer than 200 rows — PSI is noisy for small batches")
    if a.out:
        rep.to_csv(a.out, index=False)
    worst = max(rep.PSI.max(), shift)
    sys.exit(2 if worst >= monitoring.MODERATE else 1 if worst >= monitoring.STABLE else 0)


if __name__ == "__main__":
    main()

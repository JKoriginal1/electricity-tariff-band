import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import elec  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)
LOW = [40, 42, 45, 50, 48, 44, 41, 40, 43, 47, 45, 42]
HIGH = [260, 270, 300, 330, 320, 290, 270, 265, 275, 280, 300, 310]


def test_bands():
    assert list(elec.band_of([0, 60, 61, 180, 181])) == [0, 0, 1, 1, 2]


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_low_and_high_households():
    lo = client.post("/predict", json={"bills_last_12": LOW, "target_month": 6}).json()
    hi = client.post("/predict", json={"bills_last_12": HIGH, "target_month": 4, "n_ac": 2}).json()
    assert lo["predicted_band"] == "Low" and hi["predicted_band"] == "High"
    assert abs(sum(lo["probabilities"].values()) - 1) < 1e-3


@pytest.mark.parametrize("payload", [
    {"bills_last_12": LOW[:6], "target_month": 6},                 # too few bills
    {"bills_last_12": LOW, "target_month": 13},                    # bad month
    {"bills_last_12": [-5] + LOW[1:], "target_month": 6},          # negative bill
    {"bills_last_12": LOW, "target_month": 6, "house_type": "Castle"},
    {"bills_last_12": LOW, "target_month": 6, "household_members": 0},
])
def test_invalid_inputs_rejected(payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_validation_messages():
    assert elec.validate_inputs([1] * 12, {}) == []
    assert elec.validate_inputs([np.nan] * 3 + [1] * 9, {})          # too many missing
    assert elec.validate_inputs([1] * 12, {"n_ac": 99})


def test_usage_cut_lowers_high_risk():
    import joblib
    model = joblib.load(ROOT / "models" / "tariff_band_model.pkl")
    row = elec.make_input_row([190] * 12, 5, {})
    ph = [model.predict_proba(elec.scale_usage(row, -c))[0][2] for c in (0, .1, .2, .3)]
    assert ph[0] > ph[-1]


def test_drift_detects_shift():
    import pandas as pd
    import monitoring
    ref = monitoring.load_reference(ROOT / "models" / "drift_reference.json")
    rng = np.random.default_rng(0)
    rows = [elec.make_input_row(list(rng.uniform(60, 180, 12)), int(m), {}) for m in rng.integers(1, 13, 300)]
    X = pd.concat(rows, ignore_index=True)
    base = monitoring.drift_report(X, ref).set_index("feature").PSI
    hot = monitoring.drift_report(elec.scale_usage(X, 0.8), ref).set_index("feature").PSI
    assert hot["avg_last_12"] > base["avg_last_12"]
    assert monitoring.psi([0.5, 0.5], [0.5, 0.5]) == 0

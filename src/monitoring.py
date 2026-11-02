"""Data-drift monitoring: compare incoming households with the training data.

Uses the Population Stability Index (PSI) per feature:
    PSI = sum_b (p_new,b - p_ref,b) * ln(p_new,b / p_ref,b)
over decile bins of the training distribution.  Common rule of thumb:
    PSI < 0.10 stable | 0.10-0.25 moderate shift | > 0.25 major shift (investigate / retrain).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_MONITORED = ["last_1", "avg_last_3", "avg_last_12", "std_last_12", "max_last_12", "min_last_12", "trend_3"]
HOUSEHOLD_MONITORED = ["household_members", "floor_area_sqft", "n_ac", "solar_panels"]
MONITORED = HISTORY_MONITORED + HOUSEHOLD_MONITORED
# The forecast month is not monitored: a batch is usually one billing month, which is expected, not drift.
EPS = 1e-4
STABLE, MODERATE = 0.10, 0.25


def _edges(x: np.ndarray, n_bins: int = 10) -> list[float]:
    q = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return [-np.inf, *q.tolist(), np.inf]


def _shares(x: np.ndarray, edges: list[float]) -> np.ndarray:
    x = x[~np.isnan(x)]
    counts = np.histogram(x, bins=edges)[0].astype(float)
    return np.clip(counts / max(counts.sum(), 1), EPS, None)


def build_reference(X: pd.DataFrame, y: np.ndarray | None = None) -> dict:
    """Bin edges and bin shares of the training data for each monitored feature."""
    ref = {"n_rows": int(len(X)), "features": {}}
    for f in MONITORED:
        x = X[f].astype(float).to_numpy()
        e = _edges(x)
        ref["features"][f] = {"edges": [None if np.isinf(v) else v for v in e], "shares": _shares(x, e).tolist()}
    if y is not None:
        ref["band_shares"] = np.bincount(np.asarray(y, dtype=int), minlength=3).astype(float).tolist()
        tot = sum(ref["band_shares"]); ref["band_shares"] = [v / tot for v in ref["band_shares"]]
    return ref


def save_reference(X: pd.DataFrame, y, path: Path) -> None:
    Path(path).write_text(json.dumps(build_reference(X, y), indent=1))


def load_reference(path: Path) -> dict:
    ref = json.loads(Path(path).read_text())
    for f in ref["features"].values():
        f["edges"] = [(-np.inf if i == 0 else np.inf) if v is None else v for i, v in enumerate(f["edges"])]
    return ref


def psi(ref_shares, new_shares) -> float:
    p, q = np.asarray(ref_shares), np.asarray(new_shares)
    return float(np.sum((q - p) * np.log(q / p)))


def status(value: float) -> str:
    return "stable" if value < STABLE else ("moderate shift" if value < MODERATE else "MAJOR SHIFT")


def drift_report(X_new: pd.DataFrame, ref: dict, supplied: list[str] | None = None) -> pd.DataFrame:
    """One row per monitored feature: PSI, status, new median.

    Bill-history features are always checked. Household features are checked only if they were actually
    `supplied` in the batch (otherwise the app fills typical values, which would look like drift)."""
    rows = []
    for f, r in ref["features"].items():
        if f not in X_new or f not in MONITORED:
            continue
        if f in HOUSEHOLD_MONITORED and (supplied is None or f not in supplied):
            continue
        x = X_new[f].astype(float).to_numpy()
        v = psi(r["shares"], _shares(x, r["edges"]))
        rows.append({"feature": f, "PSI": round(v, 3), "status": status(v), "new_median": float(np.nanmedian(x))})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def prediction_shift(pred_bands: np.ndarray, ref: dict) -> float:
    """PSI between the predicted band mix of a batch and the training band mix (a label-free performance proxy)."""
    if "band_shares" not in ref:
        return float("nan")
    new = np.clip(np.bincount(np.asarray(pred_bands, dtype=int), minlength=3) / max(len(pred_bands), 1), EPS, None)
    return psi(ref["band_shares"], new)

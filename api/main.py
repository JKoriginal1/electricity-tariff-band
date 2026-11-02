"""FastAPI service: next-month tariff-band forecast for a household.

Run:  uvicorn api.main:app --reload      Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elec  # noqa: E402

MODEL = joblib.load(ROOT / "models" / "tariff_band_model.pkl")
META = json.loads((ROOT / "models" / "model_meta.json").read_text())
SHORT = ["Low", "Medium", "High"]

app = FastAPI(title="Electricity Tariff Band Forecaster",
              description="Predicts which domestic tariff band (≤60, 61–180, >180 units) next month's bill will fall in.",
              version=META.get("version", "1.0.0"))


class Household(BaseModel):
    bills_last_12: list[float | None] = Field(..., min_length=12, max_length=12,
                                              description="Last 12 monthly bills in units (kWh), oldest first")
    target_month: int = Field(..., ge=1, le=12)
    household_members: int = Field(4, ge=1, le=20)
    floor_area_sqft: float = Field(1200, ge=100, le=20000)
    house_type: Literal[tuple(elec.HOUSE_TYPES)] = "Single House - Single Floor"  # type: ignore[valid-type]
    n_ac: int = Field(0, ge=0, le=15)
    solar_panels: int = Field(0, ge=0, le=1)
    grid_water_heating: int = Field(0, ge=0, le=1)
    electric_cooking: int = Field(0, ge=0, le=1)
    other: dict[str, float] = Field(default_factory=dict, description="Optional extra household fields (e.g. n_fridge)")


class Forecast(BaseModel):
    predicted_band: str
    probabilities: dict[str, float]
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": META["model_name"]}


@app.post("/predict", response_model=Forecast)
def predict(h: Household) -> Forecast:
    home = h.model_dump(exclude={"bills_last_12", "target_month", "other"}) | {
        k: v for k, v in h.other.items() if k in elec.HOUSEHOLD_DEFAULTS}
    bills = [float("nan") if b is None else b for b in h.bills_last_12]
    errors = elec.validate_inputs(bills, home)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    p = MODEL.predict_proba(elec.make_input_row(bills, h.target_month, home))[0]
    return Forecast(predicted_band=SHORT[int(p.argmax())],
                    probabilities={b: round(float(x), 4) for b, x in zip(SHORT, p)}, model=META["model_name"])

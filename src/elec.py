"""Shared data preparation for the Sri Lankan household electricity tariff-band project (RECON-SL).

Used by the notebook, the Streamlit app and the API so the same cleaning rules and
feature definitions apply everywhere.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
BANDS = ["Low (≤60 units)", "Medium (61–180 units)", "High (>180 units)"]
BAND_EDGES = (60, 180)          # Sri Lankan domestic tariff break points (units = kWh per month)
HORIZON = 1                     # predict NEXT month's bill band
HISTORY = 12                    # use the last 12 monthly bills
ORIGINS = pd.date_range("2024-01-31", "2024-09-30", freq="ME")   # forecast dates (after the wave-1 survey)

# Appliances recorded in the survey that matter for electricity use
APPLIANCES = {
    "Refrigerator": "n_fridge",
    "Washing Machine": "n_washing_machine",
    "Electric Iron including electric steam iron": "n_iron",
    "Rice cooker": "n_rice_cooker",
    "TV": "n_tv",
    "Electric Kettle": "n_kettle",
    "Electric Oven": "n_oven",
    "Microwave": "n_microwave",
    "Electric Water pump": "n_water_pump",
    "Geyser / Hot water systems for bathrooms which operate from electricity": "n_geyser",
    "Laptops": "n_laptop",
    "Computers": "n_computer",
    "Electric cook tops (induction cookers, Infra-red cookers, hot plates)": "n_induction_cooker",
    "Electric water heater to heat water for drinking purposes": "n_water_heater",
    "Clothes dryer": "n_clothes_dryer",
    "Electric Vacuum Cleaner": "n_vacuum",
}

HOUSE_TYPES = ["Single House - Single Floor", "Single House - Double Floor",
               "Single House - More than 2 floors", "Flat / apartment", "Other"]
SEC_CLASSES = ["SEC A", "SEC B", "SEC C", "SEC D", "SEC E"]

HISTORY_FEATURES = ["last_1", "avg_last_3", "avg_last_6", "avg_last_12", "std_last_12",
                    "max_last_12", "min_last_12", "trend_3", "same_season_last_year", "zero_months_12"]
HOUSEHOLD_NUMERIC = ["household_members", "floor_area_sqft", "monthly_expenditure_lkr",
                     "n_rooms", "n_bulbs", "n_fans", "n_ac", "solar_panels", "electric_cooking",
                     "grid_water_heating", "home_business"] + list(APPLIANCES.values())
HOUSEHOLD_CATEGORICAL = ["house_type"]   # socio_economic_class kept only for fairness checks
CALENDAR = ["target_start_month"]
FEATURES = HISTORY_FEATURES + HOUSEHOLD_NUMERIC + HOUSEHOLD_CATEGORICAL + CALENDAR
CATEGORICAL = HOUSEHOLD_CATEGORICAL
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
# "New household" model: home & appliance details only (no bill history), for customers without bills
PROFILE_FEATURES = HOUSEHOLD_NUMERIC + HOUSEHOLD_CATEGORICAL + CALENDAR


def band_of(kwh) -> np.ndarray:
    """0 = Low (≤60), 1 = Medium (61–180), 2 = High (>180)."""
    kwh = np.asarray(kwh, dtype=float)
    return np.where(kwh <= BAND_EDGES[0], 0, np.where(kwh <= BAND_EDGES[1], 1, 2))


# --------------------------------------------------------------------------- loading
def load_bills(root: Path) -> pd.DataFrame:
    m = pd.read_csv(Path(root) / "consumption_data/non_smart_meter/monthly_consumption.csv")
    m["kwh"] = pd.to_numeric(m["consumption"].astype(str).str.replace(",", ""), errors="coerce")
    m["month"] = pd.to_datetime(m["month"]) + pd.offsets.MonthEnd(0)
    return m[["household_ID", "month", "kwh"]]


def clean_bills(bills: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a household x month matrix and a log of what was cleaned."""
    log = {}
    b = bills.copy()
    log["raw_rows"] = len(b)
    log["non_numeric_fixed"] = int(bills["kwh"].isna().sum())  # after comma removal, true missing
    # implausible single-month values (> 3,000 units is beyond any household connection)
    b.loc[b.kwh > 3000, "kwh"] = np.nan
    P = b.pivot(index="household_ID", columns="month", values="kwh").sort_index(axis=1)
    log["households"] = P.shape[0]
    log["months"] = P.shape[1]
    log["missing_cells"] = int(P.isna().sum().sum())
    # fill isolated single missing months by the average of neighbours (meter-reading gaps)
    P = P.T.interpolate(limit=1, limit_area="inside").T
    log["missing_after_interpolation"] = int(P.isna().sum().sum())
    return P, log


def household_profile(root: Path) -> pd.DataFrame:
    """Static household characteristics from the wave-1 survey (one row per household)."""
    s = Path(root) / "survey_data/wave_1"
    hh = pd.read_csv(s / "w1_household_information_and_history.csv").set_index("household_ID")
    gen = pd.read_csv(s / "w1_electricity_generation_water_heating_cooking.csv").set_index("household_ID")
    app = pd.read_csv(s / "w1_appliances.csv")
    rooms = pd.read_csv(s / "w1_room_roster.csv")
    ac = pd.read_csv(s / "w1_ac_roster.csv")

    H = pd.DataFrame(index=hh.index)
    H["household_members"] = pd.to_numeric(hh["no_of_household_members"], errors="coerce")
    H["floor_area_sqft"] = pd.to_numeric(hh["floor_area"], errors="coerce")
    H["monthly_expenditure_lkr"] = pd.to_numeric(hh["total_monthly_expenditure_of_last_month"], errors="coerce")
    ht = hh["type_of_house"].fillna("Other")
    ht = ht.replace({"Flat": "Flat / apartment", "Condominium/ Luxury apartments": "Flat / apartment"})
    H["house_type"] = ht.where(ht.isin(HOUSE_TYPES), "Other")
    H["socio_economic_class"] = hh["socio_economic_class"].where(hh["socio_economic_class"].isin(SEC_CLASSES))
    H["home_business"] = (hh["is_there_business_carried_out_in_the_household"] == "Yes").astype(int)

    H["solar_panels"] = gen["generate_electicity_using_solar_energy"].astype(str).str.startswith("Y").astype(int)
    H["electric_cooking"] = (gen["electricity_from_national_grid_used_for_cooking"] == "Yes").astype(int)
    H["grid_water_heating"] = gen["water_heating_method_for_bathing"].astype(str).str.contains(
        "grid supply|electric kettle/heater", regex=True).astype(int)

    counts = pd.crosstab(app["household_ID"], app["appliance_type"])
    for name, col in APPLIANCES.items():
        H[col] = counts[name] if name in counts else 0
    R = rooms.groupby("household_ID").agg(n_rooms=("room_ID", "size"),
                                          n_bulbs=("no_of_bulbs_in_the_room", "sum"),
                                          n_fans=("no_of_fans_in_the_room", "sum"))
    H = H.join(R)
    H["n_ac"] = ac.groupby("household_ID").size()
    fill0 = list(APPLIANCES.values()) + ["n_rooms", "n_bulbs", "n_fans", "n_ac"]
    H[fill0] = H[fill0].fillna(0)
    return H


def history_features(window: np.ndarray, target_start_month: int, same_season: float) -> dict:
    """Features from the last 12 monthly bills (oldest first)."""
    w = np.asarray(window, dtype=float)
    return {
        "last_1": w[-1],
        "avg_last_3": np.nanmean(w[-3:]),
        "avg_last_6": np.nanmean(w[-6:]),
        "avg_last_12": np.nanmean(w),
        "std_last_12": np.nanstd(w),
        "max_last_12": np.nanmax(w),
        "min_last_12": np.nanmin(w),
        "trend_3": np.nanmean(w[-3:]) - np.nanmean(w[-6:-3]),
        "same_season_last_year": same_season,
        "zero_months_12": int(np.nansum(w == 0)),
        "target_start_month": target_start_month,
    }


def build_samples(P: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    """One row per household x forecast date: last 12 bills + household profile -> band of next 3 months."""
    months = list(P.columns)
    rows = []
    for origin in ORIGINS:
        i = months.index(origin)
        hist = P.iloc[:, i - HISTORY + 1: i + 1]
        fut = P.iloc[:, i + 1: i + 1 + HORIZON]
        same = P.iloc[:, i + 1 - 12: i + 1 - 12 + HORIZON].mean(axis=1)
        ok = (hist.notna().sum(axis=1) >= 10) & (fut.notna().sum(axis=1) == HORIZON)
        for hid in P.index[ok]:
            f = history_features(hist.loc[hid].to_numpy(), (origin + pd.offsets.MonthEnd(1)).month, same.loc[hid])
            f.update(household_ID=hid, origin=origin, target_kwh=fut.loc[hid].mean())
            rows.append(f)
    S = pd.DataFrame(rows).merge(H, left_on="household_ID", right_index=True, how="inner")
    S["band"] = band_of(S["target_kwh"])
    return S


HOUSEHOLD_DEFAULTS = {
    "household_members": 4, "floor_area_sqft": 1200, "monthly_expenditure_lkr": 60000, "n_rooms": 8,
    "n_bulbs": 12, "n_fans": 3, "n_ac": 0, "solar_panels": 0, "electric_cooking": 0, "grid_water_heating": 0,
    "home_business": 0, "house_type": "Single House - Single Floor",
    **{c: 0 for c in APPLIANCES.values()}, "n_fridge": 1, "n_iron": 1, "n_rice_cooker": 1, "n_tv": 1,
}
LIMITS = {  # valid input ranges, taken from the survey data
    "household_members": (1, 20), "floor_area_sqft": (100, 20000), "monthly_expenditure_lkr": (0, 2_000_000),
    "n_rooms": (1, 40), "n_bulbs": (0, 150), "n_fans": (0, 40), "n_ac": (0, 15),
}


def validate_inputs(bills_last_12, household: dict) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    errors = []
    try:
        w = np.asarray(bills_last_12, dtype=float)
    except (TypeError, ValueError):
        return ["Monthly bills must be numbers (units / kWh)."]
    if w.shape != (HISTORY,):
        errors.append(f"Exactly {HISTORY} monthly bills are needed (got {w.size}).")
    elif np.isnan(w).sum() > 2:
        errors.append("At most 2 of the 12 monthly bills can be missing.")
    elif (w[~np.isnan(w)] < 0).any() or (w[~np.isnan(w)] > 3000).any():
        errors.append("Monthly bills must be between 0 and 3,000 units.")
    for k, (lo, hi) in LIMITS.items():
        v = household.get(k, HOUSEHOLD_DEFAULTS[k])
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{k} must be a number."); continue
        if not lo <= v <= hi:
            errors.append(f"{k} must be between {lo} and {hi} (got {v:g}).")
    for k in list(APPLIANCES.values()):
        v = household.get(k, 0)
        try:
            if not 0 <= float(v) <= 20:
                errors.append(f"{k} must be between 0 and 20.")
        except (TypeError, ValueError):
            errors.append(f"{k} must be a number.")
    if household.get("house_type", HOUSEHOLD_DEFAULTS["house_type"]) not in HOUSE_TYPES:
        errors.append(f"house_type must be one of: {', '.join(HOUSE_TYPES)}.")
    return errors


def make_input_row(bills_last_12, target_month: int, household: dict) -> pd.DataFrame:
    """Build one model-ready row from a user's last 12 bills (oldest first) and household details."""
    errors = validate_inputs(bills_last_12, household)
    if errors:
        raise ValueError("; ".join(errors))
    w = np.asarray(bills_last_12, dtype=float)
    w = pd.Series(w).interpolate(limit_direction="both").to_numpy()
    f = history_features(w, int(target_month), float(w[0]))  # same month last year = oldest bill
    f.update({**HOUSEHOLD_DEFAULTS, **household})
    return pd.DataFrame([f])[FEATURES]


def make_profile_row(household: dict, target_month: int) -> pd.DataFrame:
    """Model-ready row for the new-household (no bill history) model."""
    errors = validate_inputs([0.0] * HISTORY, household)
    if errors:
        raise ValueError("; ".join(errors))
    f = {**HOUSEHOLD_DEFAULTS, **household, "target_start_month": int(target_month)}
    return pd.DataFrame([f])[PROFILE_FEATURES]


def units_to_stay_below(edge: float, recent_avg: float) -> float:
    """Units the household would need to cut from its recent monthly average to stay at or below an edge."""
    return max(0.0, recent_avg - edge)


KWH_FEATURES = ["last_1", "avg_last_3", "avg_last_6", "avg_last_12", "std_last_12", "max_last_12",
                "min_last_12", "trend_3", "same_season_last_year"]


def scale_usage(rows: pd.DataFrame, change: float) -> pd.DataFrame:
    """What-if: the household uses `change` (e.g. -0.10) more/less in every month. All unit-based history
    features are scaled together so the input stays internally consistent."""
    r = rows.copy()
    r[KWH_FEATURES] = r[KWH_FEATURES].astype(float) * (1 + change)
    return r


def split_by_household(S: pd.DataFrame, seed: int = 42):
    """Reproduce the notebook's split: 3 demo households held out, then 60/20/20 train/validation/test by household."""
    from sklearn.model_selection import GroupShuffleSplit
    rng = np.random.default_rng(seed)
    demo = rng.choice(S.household_ID.unique(), 3, replace=False)
    data = S[~S.household_ID.isin(demo)].reset_index(drop=True)
    tv, te = next(GroupShuffleSplit(1, test_size=0.20, random_state=seed).split(data, groups=data.household_ID))
    trval = data.iloc[tv]
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(trval, groups=trval.household_ID))
    return trval.iloc[tr], trval.iloc[va], data.iloc[te]


def load_dataset(data_dir) -> pd.DataFrame:
    """Raw RECON-SL files -> modelling table (one row per household x forecast date)."""
    P, _ = clean_bills(load_bills(data_dir))
    return build_samples(P, household_profile(data_dir))

"""Streamlit front end: next-month electricity tariff-band forecast with explanations and advice.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import elec  # noqa: E402
import monitoring  # noqa: E402

st.set_page_config(page_title="Tariff Band Forecaster", page_icon="⚡", layout="wide")
SHORT = ["Low", "Medium", "High"]
COLORS = ["#2a9d8f", "#e9c46a", "#d1495b"]


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "tariff_band_model.pkl")
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text())
    comp = pd.read_csv(ROOT / "models" / "model_comparison.csv")
    profile = joblib.load(ROOT / "models" / "new_household_model.pkl")
    ref = monitoring.load_reference(ROOT / "models" / "drift_reference.json")
    return model, meta, comp, profile, ref


model, meta, comp, profile_model, drift_ref = load()
EST = model.named_steps["model"]
IS_EBM = hasattr(EST, "eval_terms")

FRIENDLY = {
    "last_1": "last month's bill", "avg_last_3": "average of last 3 bills", "avg_last_6": "average of last 6 bills",
    "avg_last_12": "12-month average", "std_last_12": "bill variability", "max_last_12": "highest bill (12 m)",
    "min_last_12": "lowest bill (12 m)", "trend_3": "recent trend", "same_season_last_year": "same month last year",
    "zero_months_12": "zero bills", "target_start_month": "month being predicted", "household_members": "family size",
    "floor_area_sqft": "floor area", "monthly_expenditure_lkr": "monthly expenditure", "n_rooms": "rooms",
    "n_bulbs": "bulbs", "n_fans": "fans", "n_ac": "air conditioners", "solar_panels": "solar panels",
    "electric_cooking": "electric cooking", "grid_water_heating": "electric water heating", "home_business": "home business",
    "house_type": "house type",
    **{v: k.split(" including")[0].split(" /")[0].split(" (")[0].lower() for k, v in elec.APPLIANCES.items()},
}


def nice(term: str) -> str:
    return " × ".join(FRIENDLY.get(t, t) for t in term.split(" & "))


def explain(row: pd.DataFrame, cls: int, est=None) -> pd.Series:
    est = est or EST
    c = pd.DataFrame(est.eval_terms(row)[0], index=est.term_names_)
    s = c.iloc[:, cls] if c.shape[1] == 3 else c.iloc[:, 0]
    s = s.loc[s.abs().sort_values().index[-10:]].sort_values()
    s.index = [nice(t) for t in s.index]
    return s


def barh(series, xlabel):
    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(series) + 1))
    ax.barh(series.index, series.values, color=["#2a7ab9" if v < 0 else "#d1495b" for v in series.values])
    ax.axvline(0, color="grey", lw=.8); ax.set_xlabel(xlabel); fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model")
    st.write(f"**{meta['algorithm']}**")
    tm = meta["test_metrics"]
    st.metric("Test accuracy", f"{tm['accuracy']:.1%}", help=f"Naive 'same as recent bills' rule: {meta['naive_test_accuracy']:.1%}")
    st.metric("Macro ROC-AUC", f"{tm['macro_auc']:.3f}")
    st.metric("Macro F1", f"{tm['macro_f1']:.3f}")
    st.caption("Trained on 4,000+ LECO households (RECON-SL, LIRNEasia & LECO, 2026). "
               "Bands: Low ≤60, Medium 61–180, High >180 units/month.")

st.title("⚡ Next-Month Electricity Tariff Band Forecaster")
st.write("Enter your **last 12 monthly bills** and a few facts about your home. The model predicts which "
         "**tariff band** next month's bill will fall in, explains **why**, and shows which changes could keep "
         "you in a cheaper band.")
st.caption("Forecast aid for an MSc assignment. It estimates units, not rupees, and tariffs change over time.")

tab1, tab5, tab2, tab3, tab4 = st.tabs(["My household (with bills)", "New household (no bills yet)", "Batch (CSV, for utilities)", "What the model learned", "Model comparison"])

# ---------------------------------------------------------------- single household
with tab1:
    left, right = st.columns([1.05, 1.25])
    with left:
        st.subheader("1 · Last 12 monthly bills (units / kWh)")
        today = date.today()
        months = pd.date_range(end=pd.Timestamp(today.year, today.month, 1) - pd.offsets.MonthEnd(1), periods=12, freq="ME")
        default = [118, 125, 140, 165, 172, 160, 140, 132, 128, 130, 138, 150]
        bills_df = st.data_editor(pd.DataFrame({"month": months.strftime("%b %Y"), "units": default}),
                                  hide_index=True, width="stretch", disabled=["month"],
                                  column_config={"units": st.column_config.NumberColumn(min_value=0, max_value=3000, step=1)})
        target_month = st.selectbox("Month to predict", list(range(1, 13)), index=today.month - 1,
                                    format_func=lambda m: date(2000, m, 1).strftime("%B"))
        st.subheader("2 · Your home (optional)")
        st.caption("With 12 months of bills, home details change the forecast very little: your bills already reflect your appliances.")
        exp = st.expander("Home details", expanded=False)
        c1, c2 = exp.columns(2)
        members = c1.number_input("People living here", 1, 20, 4)
        area = c2.number_input("Floor area (sq ft)", 100, 20000, 1200, step=50)
        house = c1.selectbox("House type", elec.HOUSE_TYPES)
        spend = c2.number_input("Monthly household spending (LKR)", 0, 2_000_000, 60000, step=5000)
        rooms = c1.number_input("Rooms", 1, 40, 7); bulbs = c2.number_input("Light bulbs", 0, 150, 12)
        fans = c1.number_input("Ceiling/stand fans", 0, 40, 3); acs = c2.number_input("Air conditioners", 0, 15, 0)
        solar = c1.checkbox("Rooftop solar panels"); ecook = c2.checkbox("Cook with electricity (rice cooker/induction)")
        heat = c1.checkbox("Electric water heater for bathing"); biz = c2.checkbox("Business run from home")
        with exp:
            st.markdown("**Appliances (counts)**")
            cols = st.columns(3); appl = {}
            defaults = {"n_fridge": 1, "n_iron": 1, "n_rice_cooker": 1, "n_tv": 1}
            for i, (label, key) in enumerate(elec.APPLIANCES.items()):
                appl[key] = cols[i % 3].number_input(FRIENDLY[key].capitalize(), 0, 20, defaults.get(key, 0), key=key)
        go = st.button("Predict next month's band", type="primary", width="stretch")

    if go:
        bills = pd.to_numeric(bills_df["units"], errors="coerce").tolist()
        home = {"household_members": members, "floor_area_sqft": area, "house_type": house, "monthly_expenditure_lkr": spend,
                "n_rooms": rooms, "n_bulbs": bulbs, "n_fans": fans, "n_ac": acs, "solar_panels": int(solar),
                "electric_cooking": int(ecook), "grid_water_heating": int(heat), "home_business": int(biz), **appl}
        problems = elec.validate_inputs(bills, home)
        with right:
            if problems:
                for p in problems: st.error(p)
            else:
                row = elec.make_input_row(bills, target_month, home)
                pr = model.predict_proba(row)[0]; cls = int(pr.argmax())
                st.subheader("Forecast")
                st.markdown(f"### Next month: <span style='color:{COLORS[cls]}'>{elec.BANDS[cls]}</span>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6.5, 1.3))
                ax.barh(SHORT, pr, color=COLORS); ax.set_xlim(0, 1)
                for i, p in enumerate(pr): ax.text(min(p + .01, .9), i, f"{p:.0%}", va="center")
                ax.set_xlabel("probability"); fig.tight_layout(); st.pyplot(fig)
                avg3 = float(np.nanmean(bills[-3:]))
                edge = 60 if avg3 <= 120 else 180
                st.info(f"Your last-3-month average is **{avg3:.0f} units**: {abs(avg3 - edge):.0f} units "
                        f"{'below' if avg3 <= edge else 'above'} the {edge}-unit band edge.")

                if IS_EBM:
                    st.markdown(f"**Why '{SHORT[cls]}'? — the model's exact explanation** (red pushes towards it, blue away)")
                    st.pyplot(barh(explain(row, cls), f"contribution to '{SHORT[cls]}' (log-odds)"))

                st.markdown("**What could keep you in a cheaper band?**")
                if cls == 2:
                    need = elec.units_to_stay_below(180, avg3)
                    st.write(f"To stay out of the **High** band, keep next month at or below **180 units**: about "
                             f"**{need:.0f} units less** than your recent average ({need / max(avg3, 1):.0%}, ≈{need / 30:.1f} units a day).")
                elif cls == 1:
                    st.write(f"You have about **{max(180 - avg3, 0):.0f} units of headroom** before the expensive **High** band "
                             f"(180 units). Reaching the cheapest **Low** band would need about {max(avg3 - 60, 0):.0f} units less a month.")
                else:
                    st.write(f"You are in the cheapest band, with about **{max(60 - avg3, 0):.0f} units of headroom** below the 60-unit edge.")
                out = []
                for cut in [0, .05, .10, .20]:
                    p = model.predict_proba(elec.scale_usage(row, -cut))[0]
                    out.append({"If monthly usage were": f"{'-' if cut else ''}{int(cut*100)}%", "P(Low)": f"{p[0]:.0%}",
                                "P(Medium)": f"{p[1]:.0%}", "P(High)": f"{p[2]:.0%}", "Likely band": SHORT[int(p.argmax())]})
                st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
                st.caption("Biggest savings usually come from air conditioners, electric water heaters, water pumps and "
                           "old refrigerators — see the 'New household' tab for how much each appliance type matters.")

# ---------------------------------------------------------------- new household
with tab5:
    st.write("No bill history yet (new connection, just moved, planning a home)? Describe the home and its appliances "
             "to estimate its **typical tariff band**. Less accurate than the bill-based forecast, because usage habits are unknown.")
    st.caption(f"New-household model test accuracy: {meta['new_household_model']['test_accuracy']:.1%} "
               f"(bill-based model: {meta['test_metrics']['accuracy']:.1%}).")
    a, b = st.columns([1.05, 1.25])
    with a:
        c1, c2 = st.columns(2)
        nh = {"household_members": c1.number_input("People", 1, 20, 4, key="nh_m"),
              "floor_area_sqft": c2.number_input("Floor area (sq ft)", 100, 20000, 1200, step=50, key="nh_a"),
              "house_type": c1.selectbox("House type", elec.HOUSE_TYPES, key="nh_h"),
              "monthly_expenditure_lkr": c2.number_input("Monthly spending (LKR)", 0, 2_000_000, 60000, step=5000, key="nh_s"),
              "n_rooms": c1.number_input("Rooms", 1, 40, 7, key="nh_r"), "n_bulbs": c2.number_input("Bulbs", 0, 150, 12, key="nh_b"),
              "n_fans": c1.number_input("Fans", 0, 40, 3, key="nh_f"), "n_ac": c2.number_input("Air conditioners", 0, 15, 0, key="nh_ac"),
              "solar_panels": int(c1.checkbox("Rooftop solar", key="nh_sol")),
              "electric_cooking": int(c2.checkbox("Electric cooking", key="nh_ec")),
              "grid_water_heating": int(c1.checkbox("Electric water heater", key="nh_wh")),
              "home_business": int(c2.checkbox("Home business", key="nh_biz"))}
        cols = st.columns(3); dflt = {"n_fridge": 1, "n_iron": 1, "n_rice_cooker": 1, "n_tv": 1}
        for i, (label, key) in enumerate(elec.APPLIANCES.items()):
            nh[key] = cols[i % 3].number_input(FRIENDLY[key].capitalize(), 0, 20, dflt.get(key, 0), key="nh_" + key)
        nh_month = st.selectbox("Month", list(range(1, 13)), index=date.today().month - 1,
                                format_func=lambda m: date(2000, m, 1).strftime("%B"), key="nh_month")
        go2 = st.button("Estimate typical band", type="primary", width="stretch")
    if go2:
        with b:
            errs = elec.validate_inputs([0.0] * 12, nh)
            if errs:
                for e in errs: st.error(e)
            else:
                r = elec.make_profile_row(nh, nh_month); pr = profile_model.predict_proba(r)[0]; c = int(pr.argmax())
                st.markdown(f"### Typical band: <span style='color:{COLORS[c]}'>{elec.BANDS[c]}</span>", unsafe_allow_html=True)
                st.write(" · ".join(f"{s_}: {p:.0%}" for s_, p in zip(SHORT, pr)))
                st.markdown("**Which home features drive this estimate** (towards High band)")
                st.pyplot(barh(explain(r, 2, profile_model), "contribution to 'High' (log-odds)"))

# ---------------------------------------------------------------- batch
REQ = ["household_ID", "target_month"] + [f"bill_{i}" for i in range(1, 13)]
with tab2:
    st.write("Upload a CSV with columns `household_ID, target_month, bill_1 … bill_12` (bill_1 = oldest). "
             "Optional household columns (e.g. `household_members, n_ac, solar_panels, house_type`) improve accuracy; "
             "missing ones use typical values.")
    ex = pd.DataFrame([{"household_ID": "H001", "target_month": 4, **{f"bill_{i}": v for i, v in enumerate([150, 160, 175, 190, 185, 170, 160, 158, 165, 172, 178, 182], 1)}, "n_ac": 1, "household_members": 5},
                       {"household_ID": "H002", "target_month": 4, **{f"bill_{i}": v for i, v in enumerate([40, 42, 45, 50, 48, 44, 41, 40, 43, 47, 52, 58], 1)}, "household_members": 2}])
    st.download_button("Download example CSV", ex.to_csv(index=False), "example_households.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        miss = [c for c in REQ if c not in df.columns]
        if miss:
            st.error("Missing columns: " + ", ".join(miss))
        else:
            res, feats = [], []
            hh_cols = [c for c in df.columns if c in elec.HOUSEHOLD_DEFAULTS]
            for r in df.itertuples(index=False):
                r = r._asdict(); bills = [r[f"bill_{i}"] for i in range(1, 13)]; home = {c: r[c] for c in hh_cols if pd.notna(r[c])}
                errs = elec.validate_inputs(bills, home)
                try:
                    tmonth = int(r["target_month"]); assert 1 <= tmonth <= 12
                except Exception:
                    errs.append("target_month must be 1–12.")
                if errs:
                    res.append({"household_ID": r["household_ID"], "status": "ERROR: " + "; ".join(errs)}); continue
                xrow = elec.make_input_row(bills, tmonth, home); feats.append(xrow)
                p = model.predict_proba(xrow)[0]
                res.append({"household_ID": r["household_ID"], "status": "ok", "predicted_band": SHORT[int(p.argmax())],
                            "P_low": round(p[0], 3), "P_medium": round(p[1], 3), "P_high": round(p[2], 3)})
            out = pd.DataFrame(res)
            n_err = (out.status != "ok").sum()
            (st.warning if n_err else st.success)(f"{len(out) - n_err} households scored, {n_err} rejected (see status column).")
            st.dataframe(out, hide_index=True, width="stretch")
            if "predicted_band" in out:
                st.bar_chart(out.predicted_band.value_counts().reindex(SHORT).fillna(0)
                             .rename(index={"Low": "1 · Low", "Medium": "2 · Medium", "High": "3 · High"}))
            st.download_button("Download results", out.to_csv(index=False), "band_forecasts.csv")

            if feats:
                st.markdown("#### Data-drift check (monitoring)")
                st.caption("Compares this batch with the households the model was trained on, using the Population "
                           "Stability Index (PSI): < 0.10 stable, 0.10–0.25 moderate shift, > 0.25 major shift → retrain "
                           "(`python mlops/retrain.py`). PSI is noisy for batches under ~200 households.")
                Xb = pd.concat(feats, ignore_index=True)
                rep = monitoring.drift_report(Xb, drift_ref, supplied=hh_cols)
                shift = monitoring.prediction_shift(model.predict(Xb), drift_ref)
                worst = max(rep.PSI.max(), shift)
                msg = f"Worst feature PSI {rep.PSI.max():.2f} ({rep.feature.iloc[0]}); predicted-band-mix PSI {shift:.2f}."
                (st.success if worst < monitoring.STABLE else st.warning if worst < monitoring.MODERATE else st.error)(
                    {True: "Stable — ", False: "Shift detected — "}[worst < monitoring.STABLE] + msg)
                st.dataframe(rep, hide_index=True, width="stretch")

# ---------------------------------------------------------------- learned
with tab3:
    if IS_EBM:
        st.write("An EBM is a sum of one learned curve per feature (plus a few interactions). These are its actual components.")
        imp = pd.Series(EST.term_importances(), index=[nice(t) for t in EST.term_names_]).sort_values().tail(15)
        st.pyplot(barh(imp, "importance (mean |score|)"))
    for f, cap in [("ebm_shapes.png", "Shape functions"), ("pdp.png", "Partial dependence"), ("shap_beeswarm.png", "SHAP (P(High band))")]:
        p = ROOT / "reports" / "figures" / f
        if p.exists(): st.image(str(p), caption=cap)

with tab4:
    st.write(f"Test-set results on **unseen households** (n = {meta['n_rows']['test']:,} household-months).")
    st.dataframe(comp, hide_index=True, width="stretch")
    p = ROOT / "reports" / "figures" / "roc_curves.png"
    if p.exists(): st.image(str(p), caption="One-vs-rest ROC curves (test set)")

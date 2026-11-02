"""Builds Electricity_Tariff_Band_Forecasting.ipynb from cells.py (+ interpretations.py if present).

Run:  python notebooks/build_notebook.py
Then: jupyter nbconvert --to notebook --execute --inplace notebooks/Electricity_Tariff_Band_Forecasting.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = []
INTERP = {}
if (HERE / "interpretations.py").exists():
    exec((HERE / "interpretations.py").read_text())  # defines INTERP dict


def md(s):
    s = s.strip()
    s = re.sub(r"<<INTERP:(\w+)>>", lambda m: INTERP.get(m.group(1), f"*(interpretation '{m.group(1)}' pending)*").strip(), s)
    cells.append(nbf.v4.new_markdown_cell(s))


# Visible "Figure N / Table N" labels above key outputs (numbers match the written report).
LABELS = [
    ('fig, axes = plt.subplots(1, 2, figsize=(14, 4))', "Figure 1 — Distribution of monthly bills and average consumption over time"),
    ('fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))', "Figure 1b — Data overview"),
    ('cols = ["household_members", "floor_area_sqft"', "Figure 1c — Correlation of household features with next-month units"),
    ('by_month = S.groupby("target_start_month")', "Figure 2 — Band mix by forecast month (seasonality)"),
    ("from sklearn.model_selection import GroupShuffleSplit", "Table 1 — Train / validation / test split (by household)"),
    ("from sklearn.metrics import (accuracy_score", "Table 2 — Naive baseline on the validation set"),
    ("from sklearn.model_selection import RandomizedSearchCV", "Hyperparameter tuning log (RandomizedSearchCV + GroupKFold)"),
    ("# Effect of tuning: default vs tuned", "Table 3 — Effect of tuning: default vs tuned accuracy"),
    ("# Stacking baseline", "Table 4 — Tuned models: CV accuracy and best hyperparameters"),
    ("p_val = {n: m.predict_proba(X_val)", "Table 5 — Validation results (all models)"),
    ("# Paired bootstrap over HOUSEHOLDS", "Table 6 — Paired bootstrap: EBM vs every other model"),
    ("top = val_table[val_table.model", "Table 7 — Final test results (unseen households)"),
    ("p_test = best.predict_proba(X_test)", "Figure 3 — Confusion matrices (test set)"),
    ("from sklearn.metrics import roc_curve", "Figure 4 — ROC curves (test set)"),
    ("# Households whose recent usage is close to a band edge", "Table 8 — Near-edge analysis: EBM vs naive rule"),
    ("from sklearn.base import clone", "Robustness checks — overfitting, temporal hold-out, history-only model"),
    ("# Fairness:", "Table 9 — Fairness: accuracy by socio-economic class, house type and solar"),
    ("imp = pd.Series(ebm_model.term_importances()", "Figure 5 — EBM term importance"),
    ("def shape(term):", "Figure 6 — EBM shape functions (what the model learned)"),
    ("appl_terms = [", "Table 10 — Appliance effects on the High-band score"),
    ("from sklearn.inspection import partial_dependence", "Figure 7 — Partial dependence (PDP) and ICE"),
    ("cats = {c: sorted(X_train[c]", "Figure 8 — SHAP values for P(High) + SHAP vs EBM ranking"),
    ("from lime.lime_tabular import LimeTabularExplainer", "Figure 9 — Local explanations: EBM (top) vs LIME (bottom)"),
    ("cand = tp[(tp.avg_last_3", "Table 11 — Usage scenarios: what if the household used less?"),
    ("prof_params = {k: v", "Figure 10 — New-household (no bills) model and appliance effects"),
    ("import joblib", "Saved model and metadata"),
    ("loaded = joblib.load(", "Inference demo — reload the saved model and predict"),
]
BANNER = ('<div id="{}" style="background:#0b4f6c;color:#fff;padding:7px 14px;border-radius:5px;'
          'font-weight:600;font-size:15px;margin-top:6px">{}</div>')


def anchor(label):
    return re.sub(r"[^a-z0-9]+", "-", label.split(" — ")[0].lower()).strip("-")


def code(s):
    s = s.strip()
    for prefix, label in LABELS:
        if s.startswith(prefix):
            cells.append(nbf.v4.new_markdown_cell(BANNER.format(anchor(label), label)))
            break
    cells.append(nbf.v4.new_code_cell(s))


exec((HERE / "cells.py").read_text())

index = "\n".join(f"- [{label}](#{anchor(label)})" for _, label in LABELS
                  if label.startswith(("Figure", "Table")))
cells.insert(1, nbf.v4.new_markdown_cell("### Figures and tables in this notebook\n"
                                         "Figure numbers match the written report.\n\n" + index))
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "Electricity_Tariff_Band_Forecasting.ipynb")
print(f"wrote {len(cells)} cells")

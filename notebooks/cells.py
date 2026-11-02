# ruff: noqa
# Cell definitions for Electricity_Tariff_Band_Forecasting.ipynb (executed by build_notebook.py)
# Cells containing <<INTERP:key>> are filled from interpretations.py after a run.

md(r"""
# Forecasting Household Electricity Tariff Bands in Sri Lanka with an Explainable Boosting Machine
### Applied Machine Learning — Individual Assignment

**Problem.** Sri Lanka's domestic electricity tariff is *block-based*: the price per unit rises steeply as a
household's monthly consumption moves past **60** and **180** units (kWh). After the 2022–2024 tariff
revisions, crossing a boundary can raise a family's bill sharply, yet most households only find out when the
bill arrives. Lanka Electricity Company (LECO) and the Ceylon Electricity Board would also like to know in
advance which customers are drifting into higher bands, so that energy-saving advice, solar promotion and
demand planning can be targeted.

**Goal.** Predict, from a household's **last 12 monthly bills** and **simple facts about the home**
(family size, house type, appliances), which tariff band its **next month's** consumption will fall in, and
**explain why**, so the household can act (e.g. reduce use of the appliances that push it into a higher band).

| Band | Monthly consumption | Meaning |
|---|---|---|
| Low | ≤ 60 units | Cheapest block |
| Medium | 61–180 units | Middle blocks |
| High | > 180 units | Most expensive blocks |

* **ML task:** 3-class classification
* **Data:** RECON-SL, a residential electricity consumption dataset for Sri Lanka (LIRNEasia with LECO,
  2026): monthly bills for 4,063 households (Oct 2022 – Oct 2024) and a household survey
* **New algorithm (not in lectures):** Explainable Boosting Machine (EBM), from InterpretML
* **Baselines:** naive "same band as recent bills", Logistic Regression, Decision Tree, KNN, Random Forest,
  XGBoost, LightGBM, CatBoost and Stacking
""")

code(r"""
import sys, os, json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import elec  # shared preprocessing module (also used by the app and API)

SEED = 42
QUICK = os.environ.get("QUICK") == "1"  # fast smoke-test mode
DATA = Path(os.environ.get("RECON_SL_DIR", ROOT / "data" / "recon_sl"))
FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 40)
BAND_SHORT = ["Low", "Medium", "High"]
BAND_COLORS = ["#2a9d8f", "#e9c46a", "#d1495b"]

import sklearn, xgboost, lightgbm, catboost, shap, interpret
print("scikit-learn", sklearn.__version__, "| interpret (EBM)", interpret.__version__, "| xgboost", xgboost.__version__,
      "| lightgbm", lightgbm.__version__, "| catboost", catboost.__version__, "| shap", shap.__version__)
""")

# ------------------------------------------------------------------ 1. data
md(r"""
## 1. Data source and first look

**Source:** *Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)*, LIRNEasia in
collaboration with Lanka Electricity Company (LECO), IEEE DataPort (doi:10.21227/n1dk-q860) and
arXiv:2609.28783 (2026). The work was funded by the LACUNA Fund.

**How it was collected:** households were randomly sampled from LECO's customer base, stratified by meter
type and LECO branch, in the western coastal service area (Moratuwa, Panadura, Kalutara, Galle, Maharagama
and others). Monthly consumption comes from LECO's billing system. Household characteristics come from
face-to-face surveys in Sinhala and Tamil (wave 1, mostly December 2023 – February 2024). Households are
identified only by an anonymous ID such as `ID0001`.

This project uses:
* `monthly_consumption.csv`: 25 monthly bills per household (Oct 2022 – Oct 2024)
* Wave-1 survey tables: household information, appliances, rooms, air conditioners, water heating and cooking
""")

code(r"""
bills = elec.load_bills(DATA)
print("Bill records:", bills.shape, "| households:", bills.household_ID.nunique(), "| months:", bills.month.nunique())
print("Raw 'consumption' values that were text with thousands separators (e.g. '1,159'):",
      int(pd.read_csv(DATA / 'consumption_data/non_smart_meter/monthly_consumption.csv').consumption.astype(str).str.contains(',').sum()))
bills.head()
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
k = bills.kwh.dropna()
axes[0].hist(k.clip(upper=600), bins=120, color="#2a7ab9")
for e in elec.BAND_EDGES: axes[0].axvline(e, color="k", ls="--", lw=1)
axes[0].set(xlabel="monthly units (kWh), clipped at 600", ylabel="bills", title="Distribution of monthly bills (dashed = tariff band edges)")
nat = bills.groupby("month").kwh.mean()
axes[1].plot(nat.index, nat.values, "o-", color="#2a7ab9")
axes[1].set(ylabel="mean units per household", title="Average monthly consumption over time")
fig.tight_layout(); fig.savefig(FIG / "eda_bills.png", dpi=120); plt.show()
print(k.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(1))
""")

md(r"""
**Observations**
* Consumption is **right-skewed**. The median household uses about 96 units a month, but the top 1% use more
  than 560, and one record shows 3,025.
* About 2.3% of bills are **zero** (vacant homes, meter faults, or solar homes exporting more than they import).
* There is a clear **seasonal pattern**: consumption peaks in the hot months (March–May) and dips mid-year.
* About 590 values were stored as **text with a thousands separator** ("1,159"). Read naively, these become
  missing values, so they were converted.
""")

# ------------------------------------------------------------------ 2. preprocessing
md(r"""
## 2. Data preprocessing

1. **Type repair:** text values such as "1,159" converted to numbers.
2. **Implausible values:** a single month above 3,000 units is beyond any domestic connection, so it is set
   to missing.
3. **Gaps:** an isolated missing month between two known months is filled by linear interpolation (a
   meter-reading gap). Longer gaps stay missing, and samples with more than 2 missing bills in the window are
   excluded.
4. **Household profile** from the wave-1 survey: family size, floor area, monthly expenditure, house type,
   home business, solar panels, electric cooking, electric water heating, number of rooms, bulbs, fans and air
   conditioners, and counts of 16 energy-relevant appliances.
5. **Encoding and scaling:** for the baseline models, numeric features are imputed (median) and scaled, and
   house type is one-hot encoded, all inside an sklearn `Pipeline`. The EBM handles missing values,
   categories and binning natively.
""")

code(r"""
P, clean_log = elec.clean_bills(bills)
H = elec.household_profile(DATA)
display(pd.Series(clean_log, name="value").to_frame())
print("Household profile:", H.shape)
H.describe().T[["count", "mean", "50%", "max"]].round(1)
""")

md(r"""
### 2.1 Building the modelling table (no look-ahead)

Each row is a **household at a forecast date** (the end of a month, January–September 2024, after the
wave-1 survey). It contains **only information available on that date**:

* **Bill history features** from the last 12 bills: last bill; averages of the last 3, 6 and 12 bills;
  standard deviation; minimum and maximum; recent trend (last 3 vs the 3 before); the bill for the same month
  last year (seasonality); and the number of zero bills.
* **Calendar:** the month being predicted.
* **Household profile** from the survey.

The **target** is the tariff band of the **following month's** bill.
""")

code(r"""
S = elec.build_samples(P, H)
print("Modelling table:", S.shape, "| households:", S.household_ID.nunique(), "| forecast dates:", S.origin.nunique())
dist = S.band.value_counts(normalize=True).sort_index().rename(index=dict(enumerate(elec.BANDS)))
display(dist.round(3).to_frame("share"))
S[["household_ID", "origin"] + elec.HISTORY_FEATURES[:5] + ["n_ac", "household_members", "target_kwh", "band"]].head()
""")

# ------------------------------------------------------------------ 3. EDA
md("## 3. Exploratory data analysis")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
S.band.map(dict(enumerate(BAND_SHORT))).value_counts().reindex(BAND_SHORT).plot.bar(ax=axes[0], color=BAND_COLORS, rot=0)
axes[0].set_title("Class balance (next-month band)")
for b in range(3):
    axes[1].hist(S.loc[S.band == b, "avg_last_3"].clip(upper=500), bins=60, alpha=.6, color=BAND_COLORS[b], label=BAND_SHORT[b])
for e in elec.BAND_EDGES: axes[1].axvline(e, color="k", ls="--", lw=1)
axes[1].set(xlabel="average of last 3 bills (units)", title="Recent usage vs next month's band"); axes[1].legend()
g = S.groupby("n_ac").band.apply(lambda s: (s == 2).mean()).loc[:4]
axes[2].bar(g.index.astype(str), g.values * 100, color="#d1495b")
axes[2].set(xlabel="number of air conditioners", ylabel="% in High band", title="Air conditioners vs High band")
fig.tight_layout(); fig.savefig(FIG / "eda_overview.png", dpi=120); plt.show()
""")

code(r"""
cols = ["household_members", "floor_area_sqft", "monthly_expenditure_lkr", "n_ac", "n_fridge", "n_washing_machine",
        "n_geyser", "n_water_pump", "n_bulbs", "solar_panels", "electric_cooking", "grid_water_heating", "avg_last_12"]
corr = S.drop_duplicates("household_ID")[cols + ["target_kwh"]].corr(method="spearman")["target_kwh"].drop("target_kwh").sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
corr.plot.barh(ax=ax, color=["#2a7ab9" if v < 0 else "#d1495b" for v in corr])
ax.set(title="Spearman correlation with next-month units (one row per household)", xlabel="ρ")
fig.tight_layout(); fig.savefig(FIG / "eda_correlations.png", dpi=120); plt.show()
corr.round(3).to_frame("spearman_rho")
""")

code(r"""
by_month = S.groupby("target_start_month").band.value_counts(normalize=True).unstack().reindex(columns=[0, 1, 2])
fig, ax = plt.subplots(figsize=(9, 3.8))
by_month.plot.bar(stacked=True, ax=ax, color=BAND_COLORS, rot=0); ax.legend(BAND_SHORT, loc="upper right")
ax.set(xlabel="month being predicted (2024)", ylabel="share", title="Band mix by month (seasonality)")
fig.tight_layout(); fig.savefig(FIG / "eda_season.png", dpi=120); plt.show()
""")

md(r"""
**EDA findings**
* **Classes:** Medium is the majority (about 58%), with Low (about 23%) and High (about 19%). Because the
  classes are imbalanced, **balanced accuracy and macro-F1** are reported alongside accuracy.
* **Recent usage is by far the strongest signal.** Households rarely jump bands, so the most informative
  cases are those **near the 60 and 180 boundaries**.
* **Air conditioners** sharply increase the chance of being in the High band. Floor area, monthly
  expenditure, number of bulbs, refrigerators, washing machines and electric water heaters are also positively
  correlated with consumption, while **solar panels** reduce billed units.
* **Seasonality:** the share of High-band months rises in the hot season (April–May).
""")

# ------------------------------------------------------------------ 4. leakage / baseline
md(r"""
## 4. Leakage check and the naive baseline

**Leakage control.** Every feature is computed from bills up to the forecast date and from the survey. The
target bill is never used as an input, and the train, validation and test sets contain **different
households** (Section 5). The survey was carried out mostly in December 2023 – February 2024, close to the
first forecast date. Appliance stock changes slowly, so it is treated as known.

**Naive baseline.** A useful model must beat the simple rule: *"next month's band = the band of the
average of the last 3 bills"*. This is how a household or utility would guess without machine learning.
""")

# ------------------------------------------------------------------ 5. split
md(r"""
## 5. Train / validation / test split (by household)

| Set | Share of households | Used for |
|---|---|---|
| Training | 60% | Fitting and hyperparameter tuning (3-fold **GroupKFold**: a household never appears in two folds) |
| Validation | 20% | Comparing tuned models, choosing the final model |
| Test | 20% | Reported once, at the end |

Splitting **by household** (not by row) is essential. Each household contributes up to 9 rows (one per
forecast date), so a row-level split would let the model memorise households and inflate the scores. Three
households are also held back completely as "new customers" for the inference demonstration.
""")

code(r"""
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
rng = np.random.default_rng(SEED)
all_hh = S.household_ID.unique()
demo_hh = rng.choice(all_hh, 3, replace=False)
data = S[~S.household_ID.isin(demo_hh)].reset_index(drop=True)

g1 = GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
trval_idx, test_idx = next(g1.split(data, groups=data.household_ID))
trval = data.iloc[trval_idx]
g2 = GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
tr_idx, val_idx = next(g2.split(trval, groups=trval.household_ID))
train, val, test = trval.iloc[tr_idx], trval.iloc[val_idx], data.iloc[test_idx]

def XY(d): return d[elec.FEATURES], d["band"].astype(int)
X_train, y_train = XY(train); X_val, y_val = XY(val); X_test, y_test = XY(test)
assert not set(train.household_ID) & set(test.household_ID) and not set(val.household_ID) & set(test.household_ID)
split = pd.DataFrame({"households": [train.household_ID.nunique(), val.household_ID.nunique(), test.household_ID.nunique()],
                      "rows": [len(train), len(val), len(test)],
                      "Low": [(d.band == 0).mean() for d in (train, val, test)],
                      "Medium": [(d.band == 1).mean() for d in (train, val, test)],
                      "High": [(d.band == 2).mean() for d in (train, val, test)]}, index=["train", "validation", "test"])
display(split.round(3))
cv3 = GroupKFold(n_splits=3)
groups_train = train.household_ID.to_numpy()
""")

code(r"""
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, log_loss)
def metrics(name, y, proba):
    yhat = proba.argmax(1)
    return {"model": name, "accuracy": accuracy_score(y, yhat), "balanced_acc": balanced_accuracy_score(y, yhat),
            "macro_f1": f1_score(y, yhat, average="macro"), "weighted_f1": f1_score(y, yhat, average="weighted"),
            "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"), "log_loss": log_loss(y, proba)}

def naive_proba(X):
    b = elec.band_of(X["avg_last_3"]); p = np.full((len(X), 3), 0.01); p[np.arange(len(X)), b] = 0.98
    return p  # hard rule expressed as (near-certain) probabilities that sum to 1

naive_val = metrics("Naive: band of last-3-bill average", y_val, naive_proba(X_val))
pd.DataFrame([naive_val]).round(4)
""")

# ------------------------------------------------------------------ 6. EBM + models
md(r"""
## 6. The new algorithm: Explainable Boosting Machine (EBM)

### 6.1 How it works
An EBM (Lou et al., 2013; Nori et al., 2019) is a **Generalised Additive Model with pairwise interactions
(GA²M)** trained by gradient boosting. For each class *k*:

$$\log\frac{P(y=k)}{P(y=\text{ref})} \;=\; \beta_{0k} + \sum_j f_{jk}(x_j) + \sum_{(i,j)} f_{ijk}(x_i, x_j)$$

* $f_j$ is a **shape function**: a learned curve showing how one feature (e.g. number of ACs) changes the score.
* $f_{ij}$ is a small 2-D surface for an **automatically detected pair** (e.g. recent usage × season).
* **Training (cyclic boosting):** with a very small learning rate, EBM visits the features in turn and fits a
  tiny tree on *one feature only* to the current residuals, adding it to that feature's curve. Repeating this
  thousands of times builds each curve separately, so the effects stay **additive and readable**.
* **Bagging:** the whole process is repeated on bootstrap samples (`outer_bags`) and averaged, which reduces
  variance and gives confidence bands.

### 6.2 Why it was selected
1. **Glass-box explanations, which the app depends on.** Because the prediction is a sum of per-feature
   terms, the app can tell a household *exactly* how much each part of its bill history, the season and its home
   contributes to the forecast. A black-box model can only approximate this afterwards with SHAP or LIME.
2. **It fits the problem's structure.** Thresholded, non-linear effects (e.g. usage near 60 or 180 units)
   and interactions with season are captured by shape functions and pairwise terms.
3. **Accuracy comparable to boosted trees** on tabular data, tested here against XGBoost, LightGBM and CatBoost.
4. **Not covered in the lectures**, and not deep learning.

### 6.3 How it differs from the standard models

| | Logistic regression | Decision tree | k-NN | RF / XGBoost / LightGBM | **EBM** |
|---|---|---|---|---|---|
| Feature effects | Linear | Steps, features mixed | No explicit model | Many deep trees mixing all features | **One flexible curve per feature** |
| Interactions | Only if added by hand | Implicit | Implicit | Implicit, hidden | **Explicit, selected pairs** |
| Explanation | Coefficients | Only if shallow | None | Needs SHAP/LIME | **Exact, built in** |
| Training | Convex optimisation | Greedy splitting | Memorises data | Bagging/boosting over all features | **Cyclic one-feature boosting + bagging** |

### 6.4 Hyperparameter tuning
None of the models use default settings. Each is tuned with `RandomizedSearchCV` using **GroupKFold (3
folds) on the training households**, scored by accuracy. The EBM search covers `interactions` (0–15),
`learning_rate` (0.005–0.05), `max_bins` (64–512), `min_samples_leaf` (2–50), `max_leaves` (2–4) and
`outer_bags` (8–14).
""")

code(r"""
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from scipy.stats import randint, uniform, loguniform

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), elec.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), elec.CATEGORICAL)])

EBM_NAME = "EBM (new algorithm)"
def make_pipeline(name, est):
    return Pipeline([("pre", "passthrough" if name == EBM_NAME else make_preprocessor()), ("model", est)])

search_spaces = {
    EBM_NAME: (ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1), 6, {
        "model__interactions": [0, 5, 10, 15], "model__learning_rate": loguniform(0.005, 0.05),
        "model__max_bins": [64, 128, 256, 512], "model__min_samples_leaf": randint(2, 50),
        "model__max_leaves": [2, 3, 4], "model__outer_bags": [8, 14]}),
    "Logistic Regression": (LogisticRegression(max_iter=4000), 8, {
        "model__C": loguniform(1e-3, 1e2), "model__class_weight": [None, "balanced"]}),
    "Decision Tree": (DecisionTreeClassifier(random_state=SEED), 20, {
        "model__max_depth": randint(3, 16), "model__min_samples_leaf": randint(5, 200), "model__criterion": ["gini", "entropy"]}),
    "KNN": (KNeighborsClassifier(), 8, {"model__n_neighbors": randint(5, 151), "model__weights": ["uniform", "distance"]}),
    "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1), 8, {
        "model__n_estimators": randint(200, 500), "model__max_depth": randint(6, 24),
        "model__min_samples_leaf": randint(2, 50), "model__max_features": ["sqrt", 0.3, 0.5]}),
    "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="mlogloss"), 12, {
        "model__n_estimators": randint(200, 800), "model__max_depth": randint(2, 8),
        "model__learning_rate": loguniform(0.01, 0.2), "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 30),
        "model__reg_lambda": loguniform(0.1, 20)}),
    "LightGBM": (LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1), 12, {
        "model__n_estimators": randint(200, 900), "model__num_leaves": randint(8, 64),
        "model__learning_rate": loguniform(0.01, 0.2), "model__min_child_samples": randint(10, 150),
        "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.5, 0.5), "model__reg_lambda": loguniform(0.1, 20)}),
    "CatBoost": (CatBoostClassifier(random_state=SEED, verbose=0, thread_count=-1, allow_writing_files=False), 6, {
        "model__iterations": randint(300, 900), "model__depth": randint(4, 8),
        "model__learning_rate": loguniform(0.02, 0.2), "model__l2_leaf_reg": loguniform(1, 20)}),
}

tuned, rows = {}, []
for name, (est, n_iter, space) in search_spaces.items():
    t0 = time.time()
    rs = RandomizedSearchCV(make_pipeline(name, est), space, n_iter=1 if QUICK else n_iter, cv=cv3, scoring="accuracy",
                            random_state=SEED, n_jobs=1, return_train_score=True)
    rs.fit(X_train, y_train, groups=groups_train)
    tuned[name] = rs.best_estimator_; i = rs.best_index_
    rows.append({"model": name, "cv_accuracy": rs.best_score_, "cv_sd": rs.cv_results_["std_test_score"][i],
                 "train_accuracy": rs.cv_results_["mean_train_score"][i], "fit_time_s": round(time.time() - t0),
                 "best_params": {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in rs.best_params_.items()}})
    print(f"{name:22s} CV accuracy {rs.best_score_:.4f} ± {rs.cv_results_['std_test_score'][i]:.4f}  ({rows[-1]['fit_time_s']}s)")
""")

code(r"""
# Effect of tuning: default vs tuned (3-fold GroupKFold accuracy on the training households)
default_rows = []
for name in [EBM_NAME, "Decision Tree", "Random Forest", "XGBoost", "LightGBM"]:
    est = sklearn.base.clone(search_spaces[name][0])
    s = cross_val_score(make_pipeline(name, est), X_train, y_train, groups=groups_train, cv=cv3, scoring="accuracy").mean()
    default_rows.append({"model": name, "default_cv_accuracy": s})
tuning_gain = pd.DataFrame(rows)[["model", "cv_accuracy"]].merge(pd.DataFrame(default_rows), on="model")
tuning_gain["gain"] = tuning_gain.cv_accuracy - tuning_gain.default_cv_accuracy
tuning_gain.round(4)
""")

code(r"""
# Stacking baseline: tuned RF + XGB + LGBM + CatBoost -> logistic meta-learner (cross-validated stacking)
base = [(n.replace(" ", "_").lower(), tuned[n].named_steps["model"]) for n in ["Random Forest", "XGBoost", "LightGBM", "CatBoost"]]
stack = Pipeline([("pre", make_preprocessor()),
                  ("model", StackingClassifier(estimators=base, final_estimator=LogisticRegression(max_iter=3000),
                                               cv=3, stack_method="predict_proba", n_jobs=1))])
t0 = time.time(); stack.fit(X_train, y_train); tuned["Stacking"] = stack
rows.append({"model": "Stacking", "cv_accuracy": np.nan, "cv_sd": np.nan, "train_accuracy": np.nan,
             "fit_time_s": round(time.time() - t0), "best_params": {"base": "RF+XGB+LGBM+CatBoost", "meta": "LogReg"}})
cv_table = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 200)
cv_table.round(4)
""")

# ------------------------------------------------------------------ 7. evaluation
md(r"""
## 7. Evaluation

**Metrics.** Accuracy (the share of correct band predictions), **balanced accuracy** and **macro-F1** (which
weight the three bands equally, because Medium dominates), **macro one-vs-rest ROC-AUC** (ranking quality)
and **log-loss** (probability quality). Every model is compared with the naive baseline.
""")

code(r"""
p_val = {n: m.predict_proba(X_val) for n, m in tuned.items()}
val_table = pd.DataFrame([naive_val] + [metrics(n, y_val, p) for n, p in p_val.items()]).sort_values("accuracy", ascending=False)
val_table.round(4)
""")

code(r"""
# Paired bootstrap over HOUSEHOLDS on the validation set: EBM vs every other model (accuracy difference)
rng = np.random.default_rng(SEED)
vh = val.household_ID.to_numpy(); uniq = np.unique(vh); idx_by_h = {h: np.where(vh == h)[0] for h in uniq}
yv = y_val.to_numpy(); B = 400
pred = {n: p.argmax(1) for n, p in p_val.items()}; pred["Naive"] = naive_proba(X_val).argmax(1)
boots = [np.concatenate([idx_by_h[h] for h in rng.choice(uniq, len(uniq))]) for _ in range(B)]
comp = []
for n in pred:
    if n == EBM_NAME: continue
    d = np.array([(pred[EBM_NAME][b] == yv[b]).mean() - (pred[n][b] == yv[b]).mean() for b in boots])
    lo, hi = np.percentile(d, [2.5, 97.5])
    comp.append({"compared_with": n, "EBM_minus_model": d.mean(), "CI_low": lo, "CI_high": hi,
                 "verdict": "EBM better" if lo > 0 else ("EBM worse" if hi < 0 else "tie")})
boot_table = pd.DataFrame(comp).sort_values("EBM_minus_model")
boot_table.round(4)
""")

md(r"""
<<INTERP:validation>>

## 8. Model selection and final test

**Selection rule (fixed before looking at the test set):** choose the model with the best validation
accuracy. If the EBM is **statistically tied** with it (the bootstrap CI includes 0), choose the EBM,
because its exact explanations are what make the app useful.
""")

code(r"""
top = val_table[val_table.model != naive_val["model"]].iloc[0].model
tie = top == EBM_NAME or boot_table.set_index("compared_with").loc[top, "verdict"] != "EBM worse"
best_name = EBM_NAME if tie else top
best = tuned[best_name]
print("Best validation accuracy:", top, "| EBM tied or better:", tie, "| SELECTED:", best_name)

test_table = pd.DataFrame([metrics("Naive: band of last-3-bill average", y_test, naive_proba(X_test))] +
                          [metrics(n, y_test, m.predict_proba(X_test)) for n, m in tuned.items()]).sort_values("accuracy", ascending=False)
test_table.round(4)
""")

code(r"""
p_test = best.predict_proba(X_test); yhat = p_test.argmax(1)
print(classification_report(y_test, yhat, target_names=BAND_SHORT, digits=3))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
cm = confusion_matrix(y_test, yhat)
sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", cbar=False, ax=axes[0], xticklabels=BAND_SHORT, yticklabels=BAND_SHORT)
axes[0].set(xlabel="predicted band", ylabel="actual band", title=f"{best_name} — test confusion matrix")
cmn = confusion_matrix(y_test, naive_proba(X_test).argmax(1))
sns.heatmap(cmn, annot=True, fmt=",", cmap="Greys", cbar=False, ax=axes[1], xticklabels=BAND_SHORT, yticklabels=BAND_SHORT)
axes[1].set(xlabel="predicted band", ylabel="actual band", title="Naive baseline — test confusion matrix")
fig.tight_layout(); fig.savefig(FIG / "confusion.png", dpi=120); plt.show()
""")

code(r"""
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
for k, ax in enumerate(axes):
    for n in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM", EBM_NAME]:
        p = tuned[n].predict_proba(X_test)[:, k]; fpr, tpr, _ = roc_curve(y_test == k, p)
        ax.plot(fpr, tpr, lw=3 if n == EBM_NAME else 1.2, label=f"{n} ({roc_auc_score(y_test == k, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set(title=f"ROC — {BAND_SHORT[k]} band vs rest (test)", xlabel="FPR", ylabel="TPR"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "roc_curves.png", dpi=120); plt.show()
""")

md("### 8.1 Where does the model add value over the naive rule?")

code(r"""
# Households whose recent usage is close to a band edge are the hard (and most useful) cases
d = np.minimum(abs(X_test.avg_last_3 - 60), abs(X_test.avg_last_3 - 180))
grp = pd.cut(d, [-1, 10, 25, 50, 1e9], labels=["within 10 units of an edge", "10–25", "25–50", "> 50 units away"])
nv = naive_proba(X_test).argmax(1)
near = pd.DataFrame({"group": grp, "y": y_test.to_numpy(), "ebm": yhat, "naive": nv}).groupby("group", observed=True).apply(
    lambda g: pd.Series({"rows": len(g), "EBM accuracy": (g.ebm == g.y).mean(), "naive accuracy": (g.naive == g.y).mean()}))
near["EBM gain"] = near["EBM accuracy"] - near["naive accuracy"]
near.round(3)
""")

code(r"""
from sklearn.base import clone
ebm_model = best.named_steps["model"]
print(f"Overfitting check — train {accuracy_score(y_train, best.predict(X_train)):.4f} | validation "
      f"{accuracy_score(y_val, best.predict(X_val)):.4f} | test {accuracy_score(y_test, yhat):.4f}")

# Temporal robustness: train on forecast dates Jan–Jun 2024 only, test on Jul–Sep 2024 (test households)
early, late = train[train.origin <= "2024-06-30"], test[test.origin >= "2024-07-31"]
m_t = clone(ebm_model).fit(early[elec.FEATURES], early.band)
pl = m_t.predict(late[elec.FEATURES])
print(f"Temporal check (train Jan–Jun, test Jul–Sep 2024, new households): accuracy {accuracy_score(late.band, pl):.4f} | "
      f"naive {accuracy_score(late.band, naive_proba(late).argmax(1)):.4f}")

# Value of the household survey: EBM with bill history only
hist_only = elec.HISTORY_FEATURES + elec.CALENDAR
m_h = clone(ebm_model).fit(X_train[hist_only], y_train)
print(f"EBM with bill history only: test accuracy {accuracy_score(y_test, m_h.predict(X_test[hist_only])):.4f} "
      f"(full model {accuracy_score(y_test, yhat):.4f})")
""")

code(r"""
# Fairness: does accuracy differ across socio-economic classes (not a model input) or house types?
ev = test.assign(pred=yhat, naive=naive_proba(X_test).argmax(1))
def grp_stats(g):
    return pd.Series({"rows": len(g), "accuracy": (g.pred == g.band).mean(), "naive": (g.naive == g.band).mean(),
                      "macro_f1": f1_score(g.band, g.pred, average="macro"), "share_High": (g.band == 2).mean()})
fair = pd.concat([ev.groupby("socio_economic_class").apply(grp_stats), ev.groupby("house_type").apply(grp_stats),
                  ev.groupby(ev.solar_panels.map({0: "no solar", 1: "solar"})).apply(grp_stats)],
                 keys=["socio-economic class", "house type", "solar"])
fair.round(3)
""")

md(r"""
<<INTERP:selection>>
""")

# ------------------------------------------------------------------ 9. XAI
md(r"""
## 9. Explainability and interpretation

| Method | Type | Question it answers |
|---|---|---|
| **EBM shape functions** | Glass box, exact, global and local | What did the model learn for each feature? |
| **Partial Dependence (PDP) + ICE** | Model-agnostic, global | How does P(High band) change as one feature changes? |
| **SHAP** | Model-agnostic, global and local | How much does each feature push each prediction? |
| **LIME** | Model-agnostic, local | Which simple rules explain this household's prediction? |

The explanations focus on the **High band**, the expensive outcome a household wants to avoid.
""")

code(r"""
imp = pd.Series(ebm_model.term_importances(), index=ebm_model.term_names_).sort_values()
top_imp = imp.tail(20)
fig, ax = plt.subplots(figsize=(8, 6.5))
top_imp.plot.barh(ax=ax, color=["#d1495b" if "&" in t else "#2a7ab9" for t in top_imp.index])
ax.set(xlabel="mean |score| (log-odds, averaged over classes)", title="EBM term importance — top 20 (red = interaction)")
fig.tight_layout(); fig.savefig(FIG / "ebm_importance.png", dpi=120); plt.show()
imp.sort_values(ascending=False).head(20).round(4).to_frame("importance")
""")

code(r"""
def shape(term):
    i = ebm_model.term_names_.index(term); return ebm_model.explain_global().data(i)

def plot_shape(ax, term, xlabel=None, xmax=None):
    d = shape(term); names = np.asarray(d["names"], dtype=float); sc = np.asarray(d["scores"])
    x = (names[:-1] + names[1:]) / 2 if len(names) == sc.shape[0] + 1 else names
    sc = sc if sc.ndim == 2 else sc[:, None]
    for k in range(sc.shape[1]):
        ax.step(x, sc[:, k], where="mid", color=BAND_COLORS[k] if sc.shape[1] == 3 else "#2a7ab9", label=BAND_SHORT[k] if sc.shape[1] == 3 else None)
    ax.axhline(0, c="k", lw=.8); ax.set(title=f"EBM shape: {term}", xlabel=xlabel or term, ylabel="score (log-odds)")
    if xmax: ax.set_xlim(right=xmax)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, (t, xm) in zip(axes.ravel(), [("avg_last_3", 400), ("last_1", 400), ("same_season_last_year", 400),
                                      ("n_ac", None), ("target_start_month", None), ("trend_3", None)]):
    if t in ebm_model.term_names_: plot_shape(ax, t, xmax=xm)
axes[0, 0].legend()
fig.tight_layout(); fig.savefig(FIG / "ebm_shapes.png", dpi=120); plt.show()
""")

code(r"""
appl_terms = [t for t in ["n_ac", "grid_water_heating", "n_water_pump", "n_washing_machine", "n_fridge", "electric_cooking",
                          "solar_panels", "n_geyser", "n_oven", "household_members", "floor_area_sqft", "n_bulbs"] if t in ebm_model.term_names_]
rows_ = []
for t in appl_terms:
    d = shape(t); sc = np.asarray(d["scores"]); sc = sc if sc.ndim == 2 else sc[:, None]
    rows_.append({"feature": t, "High-band score range": sc[:, -1].max() - sc[:, -1].min(),
                  "direction (High band)": "↑ raises" if sc[-1, -1] > sc[0, -1] else "↓ lowers"})
appl = pd.DataFrame(rows_).sort_values("High-band score range", ascending=False)
appl.round(3)
""")

md(r"""
<<INTERP:ebm_global>>

### 9.2 Partial Dependence and ICE
""")

code(r"""
from sklearn.inspection import partial_dependence
pdp_X = X_test.sample(1200, random_state=SEED).astype({c: float for c in elec.NUMERIC})
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
grid = np.arange(10, 400, 10).astype(float)
res = partial_dependence(best, pdp_X, ["avg_last_3"], kind="both", custom_values={"avg_last_3": grid}, response_method="predict_proba")
for k in range(3):
    axes[0].plot(grid, res["average"][k], color=BAND_COLORS[k], lw=3, label=BAND_SHORT[k])
for e in elec.BAND_EDGES: axes[0].axvline(e, c="k", ls="--", lw=.8)
axes[0].set(xlabel="average of last 3 bills (units)", ylabel="P(band)", title="PDP: recent usage"); axes[0].legend()
for ax, feat, vals in [(axes[1], "n_ac", np.arange(0, 5).astype(float)), (axes[2], "target_start_month", np.arange(1, 13).astype(float))]:
    r = partial_dependence(best, pdp_X, [feat], kind="both", custom_values={feat: vals}, response_method="predict_proba")
    for line in r["individual"][2][:60]: ax.plot(vals, line, color="#d1495b", alpha=.06)
    ax.plot(vals, r["average"][2], "o-", color="#d1495b", lw=3); ax.set(xlabel=feat, ylabel="P(High band)", title=f"PDP + ICE: {feat} → P(High)")
fig.tight_layout(); fig.savefig(FIG / "pdp.png", dpi=120); plt.show()
""")

md(r"""
<<INTERP:pdp>>

### 9.3 SHAP (for P(High band))
""")

code(r"""
cats = {c: sorted(X_train[c].dropna().unique()) for c in elec.CATEGORICAL}
def to_num(d):
    d = d.copy()
    for c in elec.CATEGORICAL: d[c] = d[c].map({v: i for i, v in enumerate(cats[c])})
    return d.to_numpy(dtype=float)
def from_num(a):
    d = pd.DataFrame(np.atleast_2d(a), columns=elec.FEATURES)
    for c in elec.CATEGORICAL: d[c] = d[c].round().clip(0, len(cats[c]) - 1).astype(int).map(dict(enumerate(cats[c])))
    return d
predict_fn = lambda a: best.predict_proba(from_num(a))
p_high = lambda a: predict_fn(a)[:, 2]

bg = X_train.sample(100, random_state=SEED); X_sh = X_test.sample(300 if not QUICK else 60, random_state=SEED)
explainer = shap.PermutationExplainer(p_high, shap.maskers.Independent(to_num(bg), max_samples=100), feature_names=elec.FEATURES, seed=SEED)
sv = explainer(to_num(X_sh), silent=True).values
shap_imp = pd.Series(np.abs(sv).mean(0), index=elec.FEATURES).sort_values(ascending=False)
shap.summary_plot(sv, pd.DataFrame(to_num(X_sh), columns=elec.FEATURES), max_display=15, show=False)
plt.title("SHAP — contribution to P(High band)"); plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm.png", dpi=120); plt.show()

ebm_feat = pd.Series(0.0, index=elec.FEATURES)
for t, v in zip(ebm_model.term_names_, ebm_model.term_importances()):
    parts = t.split(" & ")
    for p in parts: ebm_feat[p] += v / len(parts)
from scipy.stats import spearmanr
cmp = pd.DataFrame({"SHAP mean |value|": shap_imp, "EBM importance": ebm_feat}).sort_values("SHAP mean |value|", ascending=False)
print("Spearman rank correlation SHAP vs EBM importance: %.2f" % spearmanr(cmp.iloc[:, 0], cmp.iloc[:, 1]).correlation)
cmp.head(12).round(4)
""")

md(r"""
<<INTERP:shap>>

### 9.4 Local explanations (EBM exact vs LIME) for three households
""")

code(r"""
from lime.lime_tabular import LimeTabularExplainer
cat_idx = [elec.FEATURES.index(c) for c in elec.CATEGORICAL]
lime_bg = X_train.sample(2000, random_state=SEED)
lime_exp = LimeTabularExplainer(np.nan_to_num(to_num(lime_bg), nan=0.0), feature_names=elec.FEATURES, class_names=BAND_SHORT,
                                categorical_features=cat_idx, categorical_names={i: cats[elec.FEATURES[i]] for i in cat_idx},
                                discretize_continuous=True, random_state=SEED)
tp = X_test.assign(pH=p_test[:, 2], y=y_test.to_numpy(), pred=yhat)
cases = {"Likely High": tp.pH.idxmax(),
         "Near the 180 edge": (tp.avg_last_3 - 180).abs().idxmin(),
         "Near the 60 edge": (tp.avg_last_3 - 60).abs().idxmin()}
fig, axes = plt.subplots(2, 3, figsize=(20, 9))
for k, (label, idx) in enumerate(cases.items()):
    row = X_test.loc[[idx]]; r = tp.loc[idx]; cls = int(r.pred)
    title = (f"{label}: last-3 avg {r.avg_last_3:.0f} units, {int(r.n_ac)} AC, {int(r.household_members)} people\n"
             f"predicted {BAND_SHORT[cls]} (P={p_test[X_test.index.get_loc(idx), cls]:.2f}) | actual {BAND_SHORT[int(r.y)]}")
    contrib = pd.DataFrame(ebm_model.eval_terms(row)[0], index=ebm_model.term_names_)
    c = contrib.iloc[:, cls] if contrib.shape[1] == 3 else contrib.iloc[:, 0]
    c = c.loc[c.abs().sort_values().index[-8:]].sort_values()
    axes[0, k].barh(c.index, c.values, color=["#d1495b" if v > 0 else "#2a7ab9" for v in c.values])
    axes[0, k].axvline(0, c="k", lw=.8); axes[0, k].set_title("EBM — " + title, fontsize=9); axes[0, k].set_xlabel(f"score for '{BAND_SHORT[cls]}'")
    e = lime_exp.explain_instance(np.nan_to_num(to_num(row)[0], nan=0.0), predict_fn, labels=(cls,), num_features=8, num_samples=4000)
    w = pd.DataFrame(e.as_list(label=cls), columns=["rule", "weight"]).iloc[::-1]
    axes[1, k].barh(w.rule, w.weight, color=["#d1495b" if v > 0 else "#2a7ab9" for v in w.weight])
    axes[1, k].axvline(0, c="k", lw=.8); axes[1, k].set_title(f"LIME for '{BAND_SHORT[cls]}' (local R² = {e.score:.2f})", fontsize=9)
    print(label, "| top EBM terms:", c.round(3).to_dict())
fig.tight_layout(); fig.savefig(FIG / "local_explanations.png", dpi=120); plt.show()
""")

md(r"""
<<INTERP:local>>

### 9.5 From explanation to advice: usage scenarios
Section 8.1 showed that, once 12 months of bills are known, appliance details add almost no predictive
power: a home's appliances are already reflected in its bills. Advice is therefore based on **usage**. For a
household near the 180-unit edge, the scenarios below show how the predicted band changes if it reduces
consumption.
""")

code(r"""
cand = tp[(tp.avg_last_3.between(170, 200))].sort_values("pH")
base_row = X_test.loc[[cand.index[len(cand) // 2]]]
print(f"Household: last bill {base_row.last_1.iloc[0]:.0f}, last-3 average {base_row.avg_last_3.iloc[0]:.0f}, "
      f"12-month average {base_row.avg_last_12.iloc[0]:.0f}, 12-month max {base_row.max_last_12.iloc[0]:.0f} units")
out = []
for cut in [0, 0.05, 0.10, 0.20]:
    pr = best.predict_proba(elec.scale_usage(base_row, -cut))[0]
    out.append({"usage change": f"-{int(cut*100)}%", "P(Low)": pr[0], "P(Medium)": pr[1], "P(High)": pr[2], "predicted": BAND_SHORT[pr.argmax()]})
pd.DataFrame(out).round(3)
""")

md(r"""
### 9.6 New households with no bill history: what do appliances tell us?
New connections, recently moved families and planners have no 12-month history. A second EBM is trained on
the **home and appliance details only** (same households, same split, same tuned hyperparameters). It is
less accurate, but this is where the appliance effects can be estimated properly, which gives the
"which appliances drive consumption" insight.
""")

code(r"""
prof_params = {k: v for k, v in ebm_model.get_params().items()}
ebm_profile = ExplainableBoostingClassifier(**prof_params).fit(train[elec.PROFILE_FEATURES], y_train)
pp = ebm_profile.predict_proba(test[elec.PROFILE_FEATURES])
majority = np.bincount(y_train).argmax()
print(f"New-household model (no bills) — test accuracy {accuracy_score(y_test, pp.argmax(1)):.4f} | "
      f"balanced {balanced_accuracy_score(y_test, pp.argmax(1)):.4f} | macro AUC {roc_auc_score(y_test, pp, multi_class='ovr'):.4f} | "
      f"always-'Medium' baseline {accuracy_score(y_test, np.full(len(y_test), majority)):.4f}")
imp_p = pd.Series(ebm_profile.term_importances(), index=ebm_profile.term_names_).sort_values().tail(15)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
imp_p.plot.barh(ax=axes[0], color=["#d1495b" if "&" in t else "#2a7ab9" for t in imp_p.index])
axes[0].set(title="New-household EBM: term importance (top 15)", xlabel="mean |score|")
eff = []
for t in ["n_ac", "n_washing_machine", "n_fridge", "n_water_pump", "n_geyser", "grid_water_heating", "solar_panels",
          "electric_cooking", "n_oven", "n_microwave", "n_induction_cooker", "n_clothes_dryer"]:
    if t in ebm_profile.term_names_:
        d = ebm_profile.explain_global().data(ebm_profile.term_names_.index(t)); sc = np.asarray(d["scores"])
        sc = sc if sc.ndim == 2 else sc[:, None]
        eff.append({"feature": t, "effect on High score (0 → 1 unit)": sc[min(1, len(sc) - 1), -1] - sc[0, -1]})
eff = pd.DataFrame(eff).set_index("feature").iloc[:, 0].sort_values()
eff.plot.barh(ax=axes[1], color=["#2a7ab9" if v < 0 else "#d1495b" for v in eff])
axes[1].axvline(0, c="k", lw=.8); axes[1].set(title="Appliance effect on the High-band score (new-household model)", xlabel="change in log-odds")
fig.tight_layout(); fig.savefig(FIG / "new_household_model.png", dpi=120); plt.show()
eff.round(3).to_frame()
""")

# ------------------------------------------------------------------ 10. save
md(r"""
## 10. Saving the model and running inference

The full pipeline is saved with `joblib` (pickle), together with metadata. The Streamlit app and the
FastAPI service load exactly this file.
""")

code(r"""
import joblib
joblib.dump(best, MODELS / "tariff_band_model.pkl")
joblib.dump(ebm_profile, MODELS / "new_household_model.pkl")
row = test_table.set_index("model").loc[best_name]; nrow = test_table.set_index("model").loc["Naive: band of last-3-bill average"]
meta = {"model_name": best_name, "algorithm": "Explainable Boosting Machine (InterpretML GA2M), multiclass", "version": "1.0.0",
        "features": elec.FEATURES, "bands": elec.BANDS, "band_edges": list(elec.BAND_EDGES), "horizon_months": elec.HORIZON,
        "best_params": cv_table.set_index("model").loc[best_name, "best_params"],
        "test_metrics": {k: round(float(row[k]), 4) for k in ["accuracy", "balanced_acc", "macro_f1", "weighted_f1", "macro_auc", "log_loss"]},
        "naive_test_accuracy": round(float(nrow["accuracy"]), 4),
        "new_household_model": {"features": elec.PROFILE_FEATURES, "test_accuracy": round(float(accuracy_score(y_test, pp.argmax(1))), 4),
                                "test_macro_auc": round(float(roc_auc_score(y_test, pp, multi_class="ovr")), 4)},
        "n_households": {"train": int(train.household_ID.nunique()), "validation": int(val.household_ID.nunique()), "test": int(test.household_ID.nunique())},
        "n_rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "data_citation": "LIRNEasia & LECO (2026). Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL). IEEE DataPort. doi:10.21227/n1dk-q860"}
(MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
test_table.round(4).to_csv(MODELS / "model_comparison.csv", index=False)
cv_table.round(4).to_csv(MODELS / "tuning_results.csv", index=False)  # used by mlops/track_experiments.py
val_table.round(4).to_csv(MODELS / "validation_results.csv", index=False)
import monitoring; monitoring.save_reference(X_train, y_train, MODELS / "drift_reference.json")  # training baseline for drift checks
lime_bg.to_csv(MODELS / "lime_background.csv", index=False)
print(json.dumps(meta, indent=2, default=str))
""")

code(r"""
loaded = joblib.load(MODELS / "tariff_band_model.pkl")
demo = S[S.household_ID.isin(demo_hh)].sort_values(["household_ID", "origin"]).groupby("household_ID").tail(1)
out = demo[["household_ID", "origin", "avg_last_3", "n_ac", "household_members"]].copy()
pr = loaded.predict_proba(demo[elec.FEATURES])
out["predicted band"] = [BAND_SHORT[i] for i in pr.argmax(1)]; out["P(High)"] = pr[:, 2].round(3)
out["actual next-month units"] = demo.target_kwh.values; out["actual band"] = [BAND_SHORT[i] for i in demo.band]
out
""")

code(r"""
# A brand-new household typed in by a user (12 bills, oldest first, plus home details)
bills_12 = [150, 160, 175, 190, 185, 170, 160, 158, 165, 172, 178, 182]
home = {"household_members": 5, "floor_area_sqft": 1800, "n_ac": 1, "n_fridge": 1, "n_washing_machine": 1, "grid_water_heating": 1}
r = elec.make_input_row(bills_12, target_month=4, household=home)
pr = loaded.predict_proba(r)[0]
print({b: round(float(p), 3) for b, p in zip(BAND_SHORT, pr)}, "->", BAND_SHORT[pr.argmax()])
""")

md(r"""
<<INTERP:discussion>>
""")

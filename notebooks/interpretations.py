INTERP = {}

INTERP["validation"] = r"""
**Interpretation — validation results.**

* Every tuned model except KNN lands in a narrow 0.901–0.904 accuracy band on the validation households, and the
  paired household bootstrap shows **no statistically significant difference** between the EBM and Random Forest,
  XGBoost, LightGBM, CatBoost, Stacking, Decision Tree or Logistic Regression (all 95% CIs include 0).
* The EBM is **significantly better than the naive rule** (+1.1 points, CI +0.55 to +1.66) and than KNN (+14 points).
  KNN suffers because distance in a 39-dimensional space mixes kWh scales with appliance counts.
* The EBM has the **best log-loss (0.255) and macro AUC (0.975)** on validation, i.e. its probabilities are the
  best calibrated — this matters because the app shows probabilities, not just a band.
* Tuning helped every model (largest gain: Decision Tree +7.3 points, from depth limiting); the EBM gained least
  (+0.14) because its defaults are already strongly regularised.
"""

INTERP["selection"] = r"""
**Interpretation — test results and checks.**

* **Selection rule.** XGBoost had the best raw validation accuracy, but the EBM was statistically tied, so the
  pre-registered rule selects the EBM (equal accuracy + exact explanations). On the untouched test households the
  EBM scores **0.8956 accuracy, 0.880 balanced accuracy, 0.884 macro-F1 and 0.972 macro AUC** — second only to
  CatBoost (0.8967) by 0.1 point.
* **Against the naive rule** ("next month is in the band of my last-3 average") the EBM gains **+1.8 points
  accuracy** and **+0.08 macro AUC**. The gain is concentrated where it matters: for households **within 10 units of
  a band edge** the EBM is right 71.7% of the time vs 63.6% for the naive rule (+8.1 points); far from an edge both
  are ~96% correct. This is the honest headline: most households are easy, and the model earns its keep on the
  borderline ones — exactly those for whom a warning is useful.
* **Per-class.** Medium is easiest (F1 0.914), Low hardest (F1 0.858, recall 0.84): low bills fluctuate in
  relative terms (a 20-unit swing moves a 50-unit home across the 60 edge).
* **Overfitting.** Train 0.900 / validation 0.902 / test 0.896 — no meaningful gap.
* **Temporal check.** Trained on Jan–Jun 2024 forecasts, tested on Jul–Sep 2024 for unseen households: 0.882,
  equal to the naive rule (0.884). The model's advantage shrinks when the season it predicts is not in training,
  which is a genuine limitation (only one year of forecast origins exists).
* **History-only EBM** reaches 0.8943 vs 0.8956 for the full model: once 12 bills are known, the survey adds very
  little. The bills already "contain" the appliances.
* **Fairness.** Accuracy ranges from 0.876 (SEC A, wealthiest) to 0.928 (SEC E), and the EBM beats the naive rule
  in every group. The lower accuracy for SEC A and for flats reflects more variable, higher consumption near the
  180 edge, not a bias against poorer households — but SEC A users should be told the forecast is less certain.
  Solar homes (0.883) are slightly harder because their grid bills depend on sunshine.
"""

INTERP["ebm_global"] = r"""
**Interpretation — what the EBM learned (global).**

* **The last bill dominates** (importance 0.69), followed by the 12-month maximum and minimum and the 3/6/12-month
  averages. This is what an electricity engineer would expect: consumption is highly persistent.
* **Shape functions are sensible and readable.** In `last_1`, the Low score falls steeply between ~40 and ~70 units
  and the High score rises steeply between ~150 and ~190 — the model has rediscovered the 60 and 180 tariff edges
  from data alone. `avg_last_3` shows the same pattern more smoothly.
* **Seasonality matches Sri Lanka's climate.** `target_start_month` pushes towards High for **April** (+0.57, the
  hottest month, inter-monsoon; also Sinhala & Tamil New Year) and away from High for **June–July** (−0.38, the
  south-west monsoon, cooler and wetter, so fewer fans/AC). `trend_3` adds a momentum effect: a rising last quarter
  raises the High score.
* **Interactions** found by GA²M are mostly between history features (e.g. `last_1 × std_last_12`: a high last
  bill is less alarming for a household whose bills always swing).
* **Appliances** have small but correctly signed effects in this model (fridge, water pump, washing machine,
  geyser, household size raise the High score), because the bills already carry this information.
"""

INTERP["pdp"] = r"""
**Interpretation — partial dependence.**

* The PDP for `avg_last_3` is surprisingly **flat**. This is the known weakness of PDP with **correlated features**:
  it changes the 3-month average while holding `last_1`, `max_last_12` etc. fixed, creating unrealistic
  households (e.g. average 300 but last bill 40), and the model — correctly — trusts the other history features.
  This is why we rely on EBM shapes (one feature's own contribution) and the consistent what-if in §9.5 instead.
* The ICE lines for `n_ac` are almost horizontal: once bills are known, adding an AC on paper changes little.
* `target_start_month` shows the April peak and June–July dip for individual households (ICE), with a small
  average effect because most households are far from an edge.
"""

INTERP["shap"] = r"""
**Interpretation — SHAP (model-agnostic check).**

* SHAP (permutation explainer, 300 test households, explaining P(High)) ranks the features very similarly to the
  EBM's own importances (**Spearman ρ = 0.82**): `last_1` first by a wide margin, then `max_last_12`, the
  averages and `min_last_12`, then the month.
* The beeswarm shows the expected direction: high values of recent usage push P(High) up, low values push it down.
  AC count and grid water heating are the most influential household features but are an order of magnitude
  weaker than the bills.
* Agreement between two independent explanation methods (built-in glass-box vs post-hoc SHAP) gives confidence
  that the explanations reflect the model, not an artefact of one method.
"""

INTERP["local"] = r"""
**Interpretation — local explanations (EBM vs LIME).**

* **Clear High case** (last-3 average 316 units, 7 ACs): both methods agree — the high last bill, high 12-month
  max/min and the averages drive P(High) = 1.00. LIME's local surrogate fits well (R² 0.78).
* **Near the 180 edge** (average 180 units, 10 people): the EBM predicts Medium with only P = 0.53; the actual was
  High. The EBM shows the tension — the last bill pushes towards Medium, while a high 12-month max pushes away.
  This is an honest borderline case and the app reports it as uncertain.
* **Near the 60 edge**: correctly Medium (P = 0.77).
* LIME's surrogate is weak on the borderline cases (R² 0.20–0.24) and its explanations change with sampling;
  the **EBM's explanation is exact** (the contributions sum to the model's score), which is the main reason the
  EBM was chosen for the user-facing app.
"""

INTERP["discussion"] = r"""
## 11. Critical discussion

### Limitations
* **Accuracy ceiling.** ~89.6% test accuracy is close to the ceiling for one-month-ahead band forecasting from
  bills: a naive rule already gets 87.7%. The model's real value is on borderline households (+8 points within
  10 units of an edge) and in calibrated probabilities (AUC 0.97 vs 0.89).
* **One year of forecasts.** Forecast origins span only Jan–Sep 2024; the temporal check (train Jan–Jun, test
  Jul–Sep) shows no gain over the naive rule for unseen months. More years are needed to learn seasonality robustly.
* **Survey timing.** Household and appliance details come from survey wave 1 (Dec 2023 – Feb 2024) and are treated
  as constant; homes that bought an AC later are misdescribed.
* **Usage what-if is not causal.** §9.5 scales a household's whole history; it shows how the *model* responds to a
  household that uses X% less, not a measured effect of an intervention.
* **New-household model** is weak (60.6% vs 59.1% always-Medium; AUC 0.71). It is useful for *which appliances
  matter*, not as a reliable forecast.

### Data quality
* 57 consumption values were text with thousands separators ("1,159") and were converted; 590 readings were blank; values above 3,000
  units/month were treated as errors; single missing months were interpolated (limit 1), and samples need ≥10 of
  12 history bills. Meter-reading dates vary, so a "month" is a billing month, not a calendar month.

### Bias and fairness
* RECON-SL is a sample of Sri Lankan households, not a census: rural and estate-sector households may be
  under-represented, and results by socio-economic class show the model is **less accurate for wealthier (SEC A)
  households** (0.876) than for SEC E (0.928). Socio-economic class is deliberately **not** a model input — it is
  used only to audit fairness — so the model cannot treat households differently because of their class.

### Ethics and privacy
* Monthly consumption reveals occupancy and lifestyle. The app runs locally and stores nothing; the batch mode for
  utilities should only be used on data customers have consented to. The dataset's licence is not stated in its
  README, so the raw data are not redistributed in the public repository (`.gitignore`), only instructions to obtain it.
* The tool should **inform, not penalise**: it must never be used to deny connections, change tariffs or target
  customers.

### Impact
* Sri Lanka's domestic tariff is block-based, so crossing 60 or 180 units sharply raises the unit price. An early,
  explained warning ("you are likely to cross 180 units next month; you need ~X fewer units") lets households act
  before the bill arrives, and lets the CEB/LECO plan demand-side campaigns for borderline customers.

### Future work
* Add more years and weather data (temperature, rainfall); forecast the exact kWh with prediction intervals;
  update the tariff edges automatically when the PUCSL revises them; use smart-meter data for daily warnings.

### References
1. LIRNEasia & LECO (2026). *Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL).* IEEE DataPort. doi:10.21227/n1dk-q860
2. Lou, Y., Caruana, R., Gehrke, J., Hooker, G. (2013). *Accurate intelligible models with pairwise interactions.* KDD.
3. Nori, H., Jenkins, S., Koch, P., Caruana, R. (2019). *InterpretML: A unified framework for machine learning interpretability.* arXiv:1909.09223.
4. Lundberg, S. M., Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
5. Ribeiro, M. T., Singh, S., Guestrin, C. (2016). *"Why should I trust you?" Explaining the predictions of any classifier.* KDD.
6. Public Utilities Commission of Sri Lanka — domestic electricity tariff schedule.
"""

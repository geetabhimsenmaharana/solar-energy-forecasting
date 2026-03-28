# ☀️ Solar Energy Production Forecasting

> **Self-initiated ML pipeline** built on real solar operational data from 35 sites. Not assigned — built independently to solve a real O&M planning problem.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://share.streamlit.io)

## 🎯 Problem

A solar O&M company manages 35 sites. Operations teams need to know **how much energy each site will produce next month** to plan maintenance, allocate resources, and identify underperformance before it becomes a problem.

Manual checking was slow and reactive. This pipeline makes it proactive.

## 🔧 What It Does

- Trains and compares **5 ML models** on 24+ months of historical solar production data
- Uses **TimeSeriesSplit cross-validation** to prevent data leakage
- Selects the best model by **RMSE on held-out test data**
- Generates **monthly forecasts with ±15% confidence intervals**
- Computes **SHAP feature importance** for interpretability
- Deploys as an **interactive Streamlit dashboard**

## 📊 Models Compared

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Random Forest | Ensemble, handles non-linearity |
| **XGBoost** | Gradient boosting — typically best on tabular time-series |
| LightGBM | Faster XGBoost alternative |
| Gradient Boosting | sklearn implementation |

Winner selected by lowest RMSE on held-out test set using TimeSeriesSplit.

## 🏗️ Features Engineered (23 features)

**Temporal:** month, quarter, year, month_sin/cos, season dummies

**Lag variables:** 1-month, 3-month, 6-month, 12-month lags

**Rolling averages:** 3, 6, 12-month rolling means (actual + expected)

**Domain features:** Expected kWh baseline, year-over-year growth

**Alert features:** Historical SolarEdge fault counts (inverter, string, grid faults)

## 📁 Project Structure

```
solar-energy-forecasting/
├── data/
│   ├── ml_ready_sites.csv          # 6 sites with 12+ months of actuals
│   ├── master_final.csv            # All 35 sites, 46 features
│   └── site_registry_public.csv    # Site metadata (anonymized)
├── models/
│   └── best_model.pkl              # Saved best model
├── outputs/
│   ├── forecasts_2026.csv          # Monthly predictions per site
│   ├── model_evaluation.csv        # RMSE/MAE for all 5 models
│   ├── shap_values.csv             # Feature importance
│   ├── shap_importance.png         # SHAP bar chart
│   └── forecast_chart.png          # Forecast vs actual plot
├── forecasting_pipeline.py         # Main ML pipeline
├── dashboard.py                    # Streamlit dashboard
└── requirements.txt
```

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/geetabhimsenmaharana/solar-energy-forecasting
cd solar-energy-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ML pipeline
python forecasting_pipeline.py

# 4. Launch the dashboard
streamlit run dashboard.py
```

## 📈 Results

After training on historical data from 6 ML-ready sites:

- **Best model:** XGBoost (typically — run pipeline for actual results)
- **RMSE:** See `outputs/model_evaluation.csv` after running pipeline
- **Top features:** 12-month lag, expected kWh, rolling 6-month average, season dummies

The 12-month lag variable is consistently the strongest predictor — confirming strong year-over-year seasonality in solar production patterns.

## 🔍 SHAP Interpretability

SHAP (SHapley Additive exPlanations) values explain why the model predicted a specific value for each site-month. This answers the key operations question: *"Is this site underperforming because of seasonality or because something is actually wrong?"*

## 📊 Live Dashboard

🔗 [View Live Demo]([https://share.streamlit.io](https://solar-energy-forecasting-dtkj7bqmeebp5knkjhgdap.streamlit.app/)) ← *Add your Streamlit Cloud URL here after deploying*

## ⚠️ Data Note

All site addresses have been anonymized (Site_001 through Site_035). Real addresses are not included in this repository.

## 🛠️ Tech Stack

`Python` · `XGBoost` · `LightGBM` · `Scikit-learn` · `SHAP` · `Prophet` · `Pandas` · `NumPy` · `Streamlit` · `Plotly` · `Matplotlib`

---

*Part of a 3-project self-initiated ML portfolio built on real solar operational data.*
*→ [Anomaly Detection](https://github.com/geetabhimsenmaharana/solar-anomaly-detection)*
*→ [Automation Platform](https://github.com/geetabhimsenmaharana/solar-automation-platform)*

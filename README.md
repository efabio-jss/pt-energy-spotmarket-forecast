# ⚡ PT Energy Spot Market Forecast — Executive Energy Dashboard (Apple Night)

An end-to-end pipeline that downloads OMIE PT spot market data, engineers datasets, trains forecasting & risk models, and serves an executive-grade Streamlit dashboard.

## What you get

### ✅ Data Pipeline (omie.py)
- Downloads daily OMIE files (`marginalpdbcpt_YYYYMMDD.1`) from OMIE
- Builds clean datasets:
  - **Quarter-hour** (15-min)
  - **Daily metrics**
  - **Monthly metrics**
- Exports to `/processed` (CSV + Parquet)

### ✅ Data Science Layer (omie_ds.py)
- **Daily forecast**
  - Prophet baseline
  - XGBoost improved model
- **Intraday forecast (15-min)** with XGBoost
- **Seasonality**:
  - day-of-week
  - hourly intraday pattern
  - monthly averages
- **Peak-risk model** (classification) + **feature importance** for explainability
- Exports everything to `/processed/outputs`

### ✅ Executive Dashboard (app.py)
Streamlit dashboard with 3 pages:

- **Exec**
  - Annual/monthly KPIs
  - comparisons 2023/2024/2025
  - auto “Executive Narrative”
- **Spot Market**
  - day-of-week pattern
  - intraday profile
  - histograms + CSV download (filtered)
- **Forecast**
  - real vs predicted + error views
  - peak probabilities + alerts
  - explainability (feature importance)

---

## Setup (local)

### 1) Create environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

### 2) Install dependencies
pip install -r requirements.txt

### Run the pipeline

  A) Download + build datasets
    python omie.py
    
  B) Train models + generate outputs
    python omie_ds.py

### 4) Run the dashboard
  streamlit run app.py



## Requirements

pandas>=2.0
numpy>=1.24
requests>=2.31
pyarrow>=14.0

streamlit>=1.33
altair>=5.0

scikit-learn>=1.3
xgboost>=2.0

prophet>=1.1
cmdstanpy>=1.2


## Disclaimer

This project is for analytics and decision support. OMIE data terms apply.
No guarantee of forecasting accuracy (spot markets are volatile).

## Author:

Fabio Silva — Data & BI / Data Science







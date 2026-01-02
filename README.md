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

## yaml
> **Note:** The `raw/` and `processed/` folders should NOT be committed to GitHub (large + frequently changing). See `.gitignore` below.

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


Configuration

The scripts use local paths (Windows) that you can adjust:

  omie.py → BASE_DIR = Path(r"...\Omie")
  omie_ds.py → BASE_DIR = Path(r"...\Omie\processed")
  app.py → the dashboard asks for the outputs/ folder path in the sidebar (default is set).


Deploy (Free + Public)
Option A) Streamlit Community Cloud (recommended)

Push this repo to GitHub (public).

Go to Streamlit Community Cloud and create a new app:

Main file: app.py

Branch: main

In Streamlit Cloud “Advanced settings”, set Python version if needed.

For public/free deploy, yes — GitHub is the normal path.

Important: since OMIE data is generated locally, you have 2 common options:

(Best) Keep the pipeline local and commit only the code. The deployed app expects data to be present (not ideal for cloud).

(Better for cloud) Add a small “data refresh step” in the app or GitHub Actions to fetch/update data periodically (requires extra work).

(Simplest) Commit a small sample dataset to processed/outputs/ just for demo (e.g., last 30–90 days), and keep full datasets ignored.


.gitignore

# Python
__pycache__/
*.pyc
.venv/
.env

# Data (generated)
raw/
processed/
**/outputs/

# OS
.DS_Store
Thumbs.db


Disclaimer

This project is for analytics and decision support. OMIE data terms apply.
No guarantee of forecasting accuracy (spot markets are volatile).

Author:

Fabio Silva — Data & BI / Data Science


## 3) requirements.txt

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






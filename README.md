# Precursor - Market Signal Intelligence

Precursor is a financial machine learning pipeline built on **Databricks** that ingests S&P 500 market data, macroeconomic indicators, and SEC insider trading filings, engineers predictive features, trains two complementary models, and surfaces signals through a **Streamlit dashboard**.

---

## Overview

The pipeline follows a medallion architecture (Bronze → Silver → Gold) and is divided into five stages:

| Stage | Notebooks | Description |
|---|---|---|
| **Extraction** | `bootstrap_universe`, `00_bootstrap_fred`, `00_bootstrap_sec`, `01_bootstrap_alpaca` | One-time historical data ingestion from FRED, SEC EDGAR, and Alpaca Markets |
| **Transformation** | `02_bronze`, `03_silver`, `04_gold` | Cleaning, joining, and feature engineering |
| **Training** | `05_train` | XGBoost and Temporal Fusion Transformer (TFT) model training |
| **Evaluation** | `06_evaluate` | Checkpoint evaluation and optional training resumption |
| **Prediction** | `07_predict_v3` | Daily inference for all S&P 500 tickers |

A **Streamlit dashboard** (`Dashboard/app.py`) visualises the outputs across four pages: Insider Trades, Market Insights, Stock Explorer, and Today's Picks.

---

## Data Sources

- **Alpaca Markets** - 5 years of daily OHLCV price history for every S&P 500 constituent
- **FRED (Federal Reserve)** - Macroeconomic indicators (Fed Funds Rate, CPI, VIX, yield curves, etc.)
- **SEC EDGAR** - Form 4 insider trading filings for all tickers in the universe

---

## Pipeline Walkthrough

### 1. Extraction (run once manually, in order)

```
bootstrap_universe   →  builds precursor.bronze.universe + trading calendar
00_bootstrap_fred    →  writes precursor.bronze.fred_raw
00_bootstrap_sec     →  writes precursor.bronze.sec_raw
01_bootstrap_alpaca  →  writes precursor.bronze.alpaca_raw
```

> **Important:** Run `bootstrap_universe` first. All other bootstrap notebooks depend on `precursor.bronze.universe`.

After the initial bootstrap, daily updates are handled by separate ingestion jobs (not included in this repo).

### 2. Transformation

```
02_bronze  →  cleans raw tables (type fixes, null handling, deduplication)
03_silver  →  joins all three clean tables into precursor.silver.joined (one row per ticker/date)
04_gold    →  engineers ~40 features + target variables → precursor.gold.features
```

The gold notebook enforces strict **look-ahead bias prevention**: all feature windows use `rowsBetween(-n, -1)` and only the target variable uses `lead()`.

### 3. Training

`05_train` trains two models:

| Model | Target | AUC |
|---|---|---|
| XGBoost | `target_21d` (21-day price direction) | ~0.5316–0.5333 |
| Temporal Fusion Transformer (PyTorch) | `target_1d` (next-day direction) | - |

Models are tracked with **MLflow** and saved to `/Volumes/precursor/models/artifacts/`.

### 4. Evaluation

`06_evaluate` loads the TFT checkpoint, runs inference on the test set (2024-01-01 onward), logs metrics to MLflow, and optionally resumes training.

### 5. Prediction

`07_predict_v3` generates daily predictions for all tickers using both models and writes to:
- `precursor.gold.predictions` - per-model directional predictions with horizon (`1d` / `21d`)
- `precursor.gold.agreement` - combined signal (both UP / both DOWN / disagree)

---

## Dashboard

Start the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run Dashboard/app.py
```

The dashboard connects to Databricks SQL and provides four pages:

- **The Insider Trades** - SEC Form 4 activity across the S&P 500
- **Market Insights** - Macro signals, sector predictability rankings, and backtested strategy performance
- **Stock Explorer** - Deep-dive into any S&P 500 stock (price history, features, model predictions)
- **Today's Picks** - Top-ranked stocks by model confidence, filterable by model and horizon

The model selector in the sidebar switches between **XGBoost (21-day)** and **TFT (1-day)** views.

---

## Requirements

```
plotly==5.19.0
databricks-sql-connector==3.0.1
pandas==2.0.3
numpy==1.26.4
```

Additional notebook-level dependencies (installed via `%pip install` within each notebook):

- `fredapi` - FRED data extraction
- `alpaca-py` - Alpaca Markets API
- `xgboost`, `scikit-learn` - baseline model
- `pytorch-forecasting`, `pytorch-lightning`, `torch` - TFT model
- `mlflow` - experiment tracking
- `exchange-calendars`, `pandas-market-calendars` - trading calendar

> **Note:** All notebooks pin `numpy<2.0` to maintain ABI compatibility with pandas compiled C extensions. Do not upgrade numpy without testing.

---

## Databricks Catalog Structure

```
precursor/
├── bronze/
│   ├── universe          # S&P 500 constituents + trading calendar
│   ├── alpaca_raw        # Raw OHLCV data
│   ├── fred_raw          # Raw macro indicators
│   ├── sec_raw           # Raw Form 4 filings
│   ├── alpaca_clean      # Cleaned price data
│   ├── fred_clean        # Cleaned macro data
│   └── sec_clean         # Cleaned insider trades
├── silver/
│   └── joined            # Unified ticker/date table (~1.4M rows)
└── gold/
    ├── features          # Engineered features + targets
    ├── predictions       # Model predictions
    ├── backtest          # Backtest results
    ├── findings          # Analytical findings
    └── agreement         # Dual-model agreement signal
```

---

## Running the Full Pipeline

1. Configure your Databricks workspace and Unity Catalog.
2. Run extraction notebooks in order (see [Extraction](#1-extraction-run-once-manually-in-order)).
3. Run transformation notebooks: `02_bronze` → `03_silver` → `04_gold`.
4. Train models: `05_train`.
5. Evaluate: `06_evaluate`.
6. Generate predictions: `07_predict_v3`.
7. Launch the dashboard: `streamlit run Dashboard/app.py`.

After initial setup, steps 4–7 are designed to run on a daily schedule.

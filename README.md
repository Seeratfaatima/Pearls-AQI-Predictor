# Pearls AQI Predictor 🌍
> **End-to-End Serverless Machine Learning Pipeline & AI Air Quality Intelligence System**

An end-to-end production MLOps pipeline and interactive Streamlit web dashboard for forecasting the Air Quality Index (AQI) **1 to 3 days in advance** for Lahore, Pakistan.

---

## 📋 Comprehensive Requirements Audit

| Slide Requirement | Implementation Status | Location / Details |
| :--- | :---: | :--- |
| **1. Feature Pipeline** | ✅ Complete | [`feature_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/feature_pipeline.py) & [`utils.py`](file:///Users/seeratfatima/Documents/aqi_predictor/utils.py) — Fetches Open-Meteo Weather & Air Quality API data (synced with `Asia/Karachi` local time), engineers cyclical time features, multi-window rolling averages, lag variables, and multi-day target averages. |
| **2. Feature Store Integration** | ✅ Complete | Hopsworks Feature Store (`aqi_features` FG **v5** & Model Registry `aqi_predictor_model`) with REST API dataset chunked ingestion and automated schema sync. |
| **3. Historical Backfill** | ✅ Complete | Backfilled 19,000+ hourly observations spanning 2024 to present. |
| **4. Training Pipeline** | ✅ Complete | [`training_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/training_pipeline.py) — Trains Random Forest multi-horizon models (Day 1, Day 2, Day 3), evaluates MAE, RMSE, R², and registers model bundles in Hopsworks Model Registry. |
| **5. CI/CD Automation** | ✅ Complete | GitHub Actions workflows run **Hourly Feature Pipeline** ([`hourly_feature_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/hourly_feature_pipeline.yml)) and **Daily Training Pipeline** ([`daily_training_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/daily_training_pipeline.yml)). |
| **6. Interactive Dashboard** | ✅ Complete | [`app.py`](file:///Users/seeratfatima/Documents/aqi_predictor/app.py) — Built with Streamlit, Plotly, custom CSS cards, live pollutant telemetry, **live atmospheric weather telemetry (Temperature, Humidity, Pressure, Wind)**, and 3-day forecast path. |
| **7. SHAP Model Explainability** | ✅ Complete | Integrated `TreeExplainer` in Streamlit dashboard (`app.py`) and notebook for feature importance attribution. |
| **8. Hazardous AQI Alerts** | ✅ Complete | Health & Activity Advisory card with severity color badges, mask warnings, and indoor recommendations based on EPA AQI scale. |
| **9. Exploratory Data Analysis** | ✅ Complete | Fully executed Jupyter Notebook [`AQI_Data_Collection_cleaned_guided.ipynb`](file:///Users/seeratfatima/Documents/aqi_predictor/AQI_Data_Collection_cleaned_guided.ipynb) featuring Histograms, Density curves, Outlier Boxplots, and Correlation Heatmaps. |

---

## 🏗 System Architecture

```
                                 ┌───────────────────────────┐
                                 │   Open-Meteo Weather &    │
                                 │ Air Quality API (UTC+5)   │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
┌───────────────────────┐        ┌───────────────────────────┐
│ GitHub Actions Cron   ├───────►│    feature_pipeline.py    │
│ (Hourly & Daily Jobs) │        │ (Feature Engineering &    │
└───────────────────────┘        │  REST Chunked Ingestion)  │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  Hopsworks Feature Store  │
                                 │ (aqi_features FG v5 /     │
                                 │  Model Registry v39)      │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │    training_pipeline.py   │
                                 │ (Scikit-Learn Random      │
                                 │  Forest & Evaluation)     │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │ Hopsworks Model Registry  │
                                 │  & Streamlit Web App      │
                                 └─────────────┴─────────────┘
```

---

## 📊 Baseline Model Comparison & Evaluation Metrics

Our multi-output Random Forest AI model significantly beats standard baseline models (**Naive Persistence** and **24-Hour Rolling Mean**) on held-out test data:

| Forecast Horizon | Model / Baseline | MAE ↓ | RMSE ↓ | $R^2$ Score ↑ | Baseline Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Day 1 (24h Forecast)** | Naive Baseline | 24.07 | 31.04 | 0.4980 | Baseline |
| | 24h Mean Baseline | 17.48 | 23.41 | 0.7145 | -27.3% |
| | **Random Forest AI** | **6.87** | **9.49** | **0.9520** | **🏆 71.5% Error Reduction** |
| | | | | | |
| **Day 2 (48h Forecast)** | Naive Baseline | 31.21 | 40.52 | 0.1511 | Baseline |
| | 24h Mean Baseline | 24.30 | 32.65 | 0.4487 | -22.1% |
| | **Random Forest AI** | **5.34** | **7.55** | **0.9700** | **🏆 82.9% Error Reduction** |
| | | | | | |
| **Day 3 (72h Forecast)** | Naive Baseline | 34.08 | 43.94 | -0.0307 | Baseline |
| | 24h Mean Baseline | 27.42 | 35.91 | 0.3115 | -19.5% |
| | **Random Forest AI** | **4.36** | **6.50** | **0.9781** | **🏆 87.2% Error Reduction** |

---

## ⚙️ Running Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Feature Pipeline**:
   ```bash
   python feature_pipeline.py
   ```
3. **Execute Model Training Pipeline**:
   ```bash
   python training_pipeline.py
   ```
4. **Launch Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

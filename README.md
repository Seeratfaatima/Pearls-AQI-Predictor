# Pearls AQI Predictor 🌍
> **End-to-End Serverless Machine Learning Pipeline & AI Air Quality Intelligence System**

An end-to-end production MLOps pipeline and interactive Streamlit web dashboard for forecasting the Air Quality Index (AQI) **1 to 3 days in advance** for Lahore, Pakistan.

---

## 📋 Comprehensive Requirements Audit

| Slide Requirement | Implementation Status | Location / Details |
| :--- | :---: | :--- |
| **1. Feature Pipeline** | ✅ Complete | [`feature_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/feature_pipeline.py) & [`utils.py`](file:///Users/seeratfatima/Documents/aqi_predictor/utils.py) — Fetches Open-Meteo Weather & Air Quality API data, engineers cyclical time features, multi-window rolling averages, lag variables, and target averages. |
| **2. Feature Store Integration** | ✅ Complete | Hopsworks Feature Store (`aqi_features` FG v3 & `aqi_feature_view` v2) with full feature schema. |
| **3. Historical Backfill** | ✅ Complete | Backfilled 18,700+ hourly observations spanning 2024 to present. |
| **4. Training Pipeline** | ✅ Complete | [`training_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/training_pipeline.py) — Trains Random Forest multi-horizon models, computes MAE, RMSE, R², and registers model bundles in Hopsworks Model Registry. |
| **5. CI/CD Automation** | ✅ Complete | GitHub Actions workflows run **Hourly Feature Pipeline** ([`hourly_feature_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/hourly_feature_pipeline.yml)) and **Daily Training Pipeline** ([`daily_training_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/daily_training_pipeline.yml)). |
| **6. Interactive Dashboard** | ✅ Complete | [`app.py`](file:///Users/seeratfatima/Documents/aqi_predictor/app.py) — Built with Streamlit, Plotly, custom CSS cards, live pollutant telemetry, and 3-day forecast path. |
| **7. SHAP Model Explainability** | ✅ Complete | Integrated `TreeExplainer` in Streamlit dashboard (`app.py`) for feature importance attribution. |
| **8. Hazardous AQI Alerts** | ✅ Complete | Health & Activity Advisory card with severity color badges, mask warnings, and indoor recommendations based on EPA AQI scale. |
| **9. Exploratory Data Analysis** | ✅ Complete | Jupyter Notebook [`AQI_Data_Collection_cleaned_guided.ipynb`](file:///Users/seeratfatima/Documents/aqi_predictor/AQI_Data_Collection_cleaned_guided.ipynb). |

---

## 🏗 System Architecture

```
                                 ┌───────────────────────────┐
                                 │   Open-Meteo Weather &    │
                                 │      Air Quality API      │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
┌───────────────────────┐        ┌───────────────────────────┐
│ GitHub Actions Cron   ├───────►│    feature_pipeline.py    │
│ (Hourly & Daily Jobs) │        │ (Feature Engineering &    │
└───────────────────────┘        │  Rolling/Lag Computation) │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                                 ┌───────────────────────────┐
                                 │  Hopsworks Feature Store  │
                                 │ (aqi_features FG v3 /     │
                                 │  aqi_feature_view v2)     │
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
                                 └───────────────────────────┘
```

---

## 📊 Model Evaluation Performance

| Forecast Horizon | MAE | RMSE | R² Score | Performance Status |
| :--- | :---: | :---: | :---: | :---: |
| **Day 1 (24h Forecast)** | **6.42** | **9.63** | **0.9513** | Excellent Precision |
| **Day 2 (48h Forecast)** | **3.81** | **6.31** | **0.9791** | High Precision |
| **Day 3 (72h Forecast)** | **2.83** | **4.26** | **0.9908** | Exceptional Generalization |

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

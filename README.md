# Pearls AQI Predictor 🌍

An end-to-end serverless machine learning pipeline and interactive web dashboard for forecasting the Air Quality Index (AQI) 3 days in advance in Lahore, Pakistan.

## Architecture Overview

```
[Open-Meteo API] ---> (feature_pipeline.py) ---> [Hopsworks Feature Store]
                                                           |
                                                           v
[Streamlit Dashboard] <--- [Hopsworks Model Registry] <--- (training_pipeline.py)
```

1. **Feature Pipeline (`feature_pipeline.py`)**: Fetches hourly weather and air quality API data, engineers time-based cyclical features, multi-window rolling means (`rolling(2..72)`), and multi-step lag predictors, and updates Hopsworks Feature Store (`aqi_features` FG v3). Runs hourly via GitHub Actions.
2. **Training Pipeline (`training_pipeline.py`)**: Reads features from Hopsworks, trains `RandomForestRegressor` multi-output models, evaluates metrics (MAE, RMSE, R²), saves `.joblib` model artifacts, and registers models in Hopsworks Model Registry. Runs daily via GitHub Actions.
3. **Web Application Dashboard (`app.py`)**: A modern dark glassmorphic Streamlit web application rendering live 3-day AQI prediction cards, color-coded health severity badges, hazardous AQI level alerts, interactive Plotly timeline charts, and SHAP model explainability.

## Model Evaluation Metrics

| Forecast Horizon | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| **Day 1 Average AQI** | 6.49 | 9.29 | 0.9541 |
| **Day 2 Average AQI** | 4.17 | 6.54 | 0.9773 |
| **Day 3 Average AQI** | 3.34 | 5.03 | **0.9865** |

## Running Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run feature pipeline:
   ```bash
   python feature_pipeline.py
   ```
3. Run training pipeline:
   ```bash
   python training_pipeline.py
   ```
4. Launch Streamlit web dashboard:
   ```bash
   streamlit run app.py
   ```

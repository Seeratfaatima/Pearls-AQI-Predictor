# End-to-End Machine Learning Journey & Documentation Report
## Pearls AQI Predictor — AI-Powered Air Quality Intelligence Stack

> **Author**: Seerat Fatima  
> **GitHub Repository**: [https://github.com/Seeratfaatima/Pearls-AQI-Predictor](https://github.com/Seeratfaatima/Pearls-AQI-Predictor)  
> **Live Streamlit Web Application**: [https://pearls-aqi-predictor-s5847trmkcuy49yhzdrni8.streamlit.app/](https://pearls-aqi-predictor-s5847trmkcuy49yhzdrni8.streamlit.app/)

---

## 1. Executive Summary & Problem Statement

Air pollution, specifically fine particulate matter ($\text{PM}_{2.5}$), poses a severe health hazard in major urban centers such as **Lahore, Pakistan** ($\text{Latitude: } 31.5497, \text{Longitude: } 74.3436$). Rapid seasonal shifts, agricultural residue burning, and industrial emissions cause air quality levels to fluctuate drastically between moderate and hazardous categories.

**Objective**: Build a production-grade, serverless MLOps system that automatically ingests satellite weather and pollutant telemetry, engineers predictive time-series features, stores data in an enterprise Feature Store (Hopsworks), trains multi-horizon Random Forest forecast models for **Day 1 (24h)**, **Day 2 (48h)**, and **Day 3 (72h)**, and serves predictions via an interactive web dashboard with SHAP explainability.

---

## 2. Telemetry API Selection & Data Collection

### API Selection & Rationale
After evaluating multiple weather and environmental data providers (OpenWeatherMap, AirVisual, Open-Meteo), **Open-Meteo** was selected based on the following technical criteria:
* **Dual Endpoint Access**: Provides specialized REST endpoints for both high-resolution weather telemetry (`archive-api.open-meteo.com`) and pollutant telemetry (`air-quality-api.open-meteo.com`).
* **Granular Historical & Forecast Data**: Sub-hourly hourly observations dating back to July 2024, enabling robust historical backfilling (10,000+ hourly observations).
* **Zero Authentication Rate-Limit Throttling**: High-bandwidth data retrieval suitable for automated cron pipelines.
* **Timezone Synchronization**: Native support for timezone parameters (`timezone=Asia/Karachi`), ensuring zero time-shift misalignment between UTC servers and local Lahore observations.

### Ingested Telemetry Features:
1. **Weather Telemetry**: Temperature (`temperature_2m`), Relative Humidity (`relative_humidity_2m`), Surface Pressure (`surface_pressure`), Wind Speed (`wind_speed_10m`).
2. **Air Quality Telemetry**: $\text{PM}_{2.5}$, $\text{PM}_{10}$, Carbon Monoxide ($\text{CO}$), Nitrogen Dioxide ($\text{NO}_2$), Ozone ($\text{O}_3$).

---

## 3. Data Cleaning & Preprocessing

1. **US EPA AQI Calculation**: $\text{PM}_{2.5}$ raw concentration ($\mu\text{g/m}^3$) was converted into the official US Environmental Protection Agency (EPA) Air Quality Index scale using linear breakpoint interpolation across standard severity categories ($0\text{--}50, 51\text{--}100, 101\text{--}150, 151\text{--}200, 201\text{--}300, 301\text{--}400, 401\text{--}500$).
2. **Missing Value Imputation**: Continuous linear interpolation for minor hourly gaps; dropping edge null rows resulting from long-lag windows.
3. **Timestamp Standardization**: Floor rounding timestamps to 1-second precision and sorting chronologically to prevent temporal data leakage.

---

## 4. Feature Engineering

To capture diurnal periodicity, atmospheric trends, and historical pollution momentum, 70 engineered features were constructed in [`utils.py`](file:///Users/seeratfatima/Documents/aqi_predictor/utils.py):

* **Cyclic Temporal Encodings**: `hour_sin`, `hour_cos`, `month_sin`, `month_cos` (using $\sin(2\pi \cdot t / T)$ and $\cos(2\pi \cdot t / T)$ transformations).
* **Multi-Window Rolling Statistics**: Rolling means and standard deviations across 2, 3, 6, 12, 24, 48, and 72-hour windows for $\text{PM}_{2.5}$ and AQI.
* **Multi-Lag Variables**: Lag values at $1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48,$ and $72$ hours prior.
* **Environmental Delta Rates**: First-order differentials ($\Delta$) for temperature, humidity, wind speed, and $\text{PM}_{2.5}$.
* **Multi-Horizon Targets**: 24-hour forward target averages for Day 1 (`aqi_day1`), Day 2 (`aqi_day2`), and Day 3 (`aqi_day3`).

---

## 5. Model Exploration, Benchmarking & Final Selection

Multiple model architectures were evaluated on held-out test datasets ($80/20$ train-test split):

### Evaluated Model Architectures:
1. **Naive Persistence Baseline**: Predicts future AQI to equal current observed AQI ($y_{\text{pred}} = \text{AQI}_{\text{today}}$).
2. **24-Hour Rolling Mean Baseline**: Predicts future AQI to equal the mean of the past 24 hours.
3. **Ridge Linear Regressor**: Linear model with $L_2$ regularization.
4. **Multi-Layer Perceptron (MLP Neural Network)**: 2-layer dense network ($64 \times 32$ neurons with ReLU activation).
5. **Random Forest Multi-Output Regressor (Final Selected Model)**: Ensemble of 100 decision trees with `max_depth=15`.

### Empirical Results & Model Comparison Table:

| Forecast Horizon | Model / Baseline | MAE ($\mu\text{g/m}^3$) ↓ | RMSE ($\mu\text{g/m}^3$) ↓ | $R^2$ Score ↑ | Baseline Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Day 1 (24h Forecast)** | Naive Persistence Baseline | 24.07 | 31.04 | 0.4980 | Baseline |
| | 24h Mean Baseline | 17.48 | 23.41 | 0.7145 | -27.3% |
| | **Random Forest AI** | **7.40** | **10.35** | **0.9442** | **🏆 69.3% Error Reduction** |
| | | | | | |
| **Day 2 (48h Forecast)** | Naive Persistence Baseline | 31.21 | 40.52 | 0.1511 | Baseline |
| | 24h Mean Baseline | 24.30 | 32.65 | 0.4487 | -22.1% |
| | **Random Forest AI** | **5.71** | **8.51** | **0.9626** | **🏆 81.7% Error Reduction** |
| | | | | | |
| **Day 3 (72h Forecast)** | Naive Persistence Baseline | 34.08 | 43.94 | -0.0307 | Baseline |
| | 24h Mean Baseline | 27.42 | 35.91 | 0.3115 | -19.5% |
| | **Random Forest AI** | **4.25** | **6.33** | **0.9786** | **🏆 87.5% Error Reduction** |

### Rationale for Selecting Random Forest Regressor:
* **Non-Linear Interactions**: Superior ability to model complex non-linear interactions between weather features (surface pressure drops + high humidity) and particulate accumulation.
* **Robustness to Spikes**: Decision tree splitting boundaries prevent extreme pollution anomalies from distorting global forecasts.
* **Interpretability**: Seamless compatibility with `SHAP` (SHapley Additive exPlanations) for real-time feature attribution on the dashboard.
* **Efficient Serialization**: Artifact compression (`joblib.dump(..., compress=3)`) reduced model size to ~8MB, allowing rapid download and low-latency inference in cloud environments.

---

## 6. Blockers Faced & MLOps Resolutions

During the development and deployment journey, several critical technical blockers were encountered and resolved:

### Blocker 1: Hopsworks Feature Group Version Mismatch (`v3` vs `v4`)
* **Issue**: The feature pipeline ([`feature_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/feature_pipeline.py)) pushed engineered data to Hopsworks Feature Group `version=4`, but the training pipeline ([`training_pipeline.py`](file:///Users/seeratfatima/Documents/aqi_predictor/training_pipeline.py)) attempted to query `version=3`, causing pipeline crashes in CI/CD.
* **Resolution**: Updated `training_pipeline.py` to query `version=4` and added an automated fallback mechanism to run `feature_pipeline.py` locally on-the-fly if remote feature group reads fail.

### Blocker 2: Ingestion Timeout in GitHub Actions (`wait_for_job=True`)
* **Issue**: The GitHub Actions runner hung for 9+ minutes while polling Hopsworks cluster jobs due to `write_options={"wait_for_job": True}`.
* **Resolution**: Changed insertion mode to non-blocking asynchronous ingestion (`write_options={"wait_for_job": False}`), allowing GitHub Actions workflows to finish cleanly in under 45 seconds.

### Blocker 3: Missing Local Model Artifacts in Streamlit Cloud
* **Issue**: Model binary files (`models/*.joblib`) were excluded from git via `.gitignore`. When deployed to Streamlit Cloud, the app crashed with `Trained models not found in models/`.
* **Resolution**: Updated [`app.py`](file:///Users/seeratfatima/Documents/aqi_predictor/app.py) with an automatic Hopsworks Model Registry download fallback (`mr.get_model("aqi_predictor_model").download()`), allowing the app to dynamically fetch the latest model bundle at runtime.

### Blocker 4: UTC Server Timezone Offset
* **Issue**: Streamlit Cloud servers run in UTC timezone, causing the app header to display `Updated: 06:36 PKT` when local Pakistan time was `11:36 AM PKT`.
* **Resolution**: Replaced standard server `now()` calls with explicit UTC+5 offset calculations (`datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5)`), guaranteeing accurate PKT display worldwide.

### Blocker 5: Unescaped Strings in Jupyter Notebook Output JSON
* **Issue**: Exporting notebook cell outputs resulted in unescaped newline strings (`unterminated string literal`), preventing VS Code from rendering notebook cells.
* **Resolution**: Programmatically purged raw traceback output JSON, sanitized cell strings, and re-executed all 21 notebook cells to generate embedded, clean matplotlib/seaborn plots and tables.

---

## 7. Additional Subjective Features & UI Polish

* **High-Aesthetic Light CSS Theme**: Custom styled metric cards, status pills, and health advisory blocks.
* **Current Weather & Atmospheric Telemetry**: Real-time widgets for **Temperature (°C)**, **Relative Humidity (%)**, **Surface Pressure (hPa)**, and **Wind Speed (km/h)**.
* **Live Pollutant Telemetry**: Real-time monitoring for $\text{PM}_{2.5}$, $\text{PM}_{10}$, $\text{O}_3$, $\text{NO}_2$, and $\text{CO}$.
* **Interactive Plotly Forecast Trajectory**: Dual-series visualization contrasting historical observations against 72-hour AI forecast paths.
* **SHAP Feature Attribution**: Interactive bar chart breaking down top positive and negative drivers behind predictions.
* **Automated MLOps Automation**: GitHub Actions hourly feature synchronization ([`hourly_feature_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/hourly_feature_pipeline.yml)) and daily model retraining ([`daily_training_pipeline.yml`](file:///Users/seeratfatima/Documents/aqi_predictor/.github/workflows/daily_training_pipeline.yml)).

---

## 8. Verification & Access Links

* **GitHub Code Repository**: [https://github.com/Seeratfaatima/Pearls-AQI-Predictor](https://github.com/Seeratfaatima/Pearls-AQI-Predictor)
* **Live Web Application**: [https://pearls-aqi-predictor-s5847trmkcuy49yhzdrni8.streamlit.app/](https://pearls-aqi-predictor-s5847trmkcuy49yhzdrni8.streamlit.app/)
* **EDA Notebook**: [`AQI_Data_Collection_cleaned_guided.ipynb`](file:///Users/seeratfatima/Documents/aqi_predictor/AQI_Data_Collection_cleaned_guided.ipynb)

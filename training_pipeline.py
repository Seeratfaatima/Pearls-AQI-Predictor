#!/usr/bin/env python3
"""
Training Pipeline - Pearls AQI Predictor
Runs daily to read feature store data, train Random Forest models for Day 1, Day 2, and Day 3,
evaluate metrics (MAE, RMSE, R²), save artifacts locally, and register models in Hopsworks.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from utils import get_hopsworks_project

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "aqi_features.parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    """Reads feature data from Hopsworks or local fallback parquet."""
    project = get_hopsworks_project()
    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_features", version=3)
            df = fg.read()
            df = df.sort_values("time").reset_index(drop=True)
            print(f"[Training Pipeline] Successfully read {len(df)} rows from Hopsworks Feature Store.")
            return df
        except Exception as e:
            print(f"[Training Pipeline] Hopsworks read fallback: {e}")

    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        print(f"[Training Pipeline] Loaded {len(df)} rows from local snapshot {DATA_PATH}.")
        return df
    
    raise FileNotFoundError("No feature data found in Hopsworks or local data directory. Run feature_pipeline.py first!")

def run_training_pipeline():
    print("=" * 60)
    print("[Training Pipeline] Starting model training and evaluation...")
    print("=" * 60)

    df = load_data()

    # Ensure rolling features and multi-day targets are present
    if "aqi_day1" not in df.columns:
        from utils import engineer_features
        df = engineer_features(df, is_training=True)

    # Clean missing targets
    clean_df = df.dropna(subset=["aqi_day1", "aqi_day2", "aqi_day3"]).reset_index(drop=True)
    drop_cols = ["time", "aqi_day1", "aqi_day2", "aqi_day3"]
    feature_cols = [c for c in clean_df.columns if c not in drop_cols]
    
    X = clean_df[feature_cols]

    metrics_summary = {}
    trained_models = {}

    for target_col, day_name in [("aqi_day1", "Day 1"), ("aqi_day2", "Day 2"), ("aqi_day3", "Day 3")]:
        y = clean_df[target_col]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        mae = mean_absolute_error(y_te, preds)
        rmse = np.sqrt(mean_squared_error(y_te, preds))
        r2 = r2_score(y_te, preds)

        print(f"[{day_name} Forecast] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

        metrics_summary[f"mae_{day_name.lower().replace(' ', '')}"] = round(mae, 2)
        metrics_summary[f"r2_{day_name.lower().replace(' ', '')}"] = round(r2, 4)

        # Save model artifact
        model_path = os.path.join(MODELS_DIR, f"model_{day_name.lower().replace(' ', '')}.joblib")
        joblib.dump(model, model_path)
        trained_models[day_name] = model

    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    print(f"[Training Pipeline] Saved local model artifacts to {MODELS_DIR}/")

    # Hopsworks Model Registry upload
    project = get_hopsworks_project()
    if project:
        try:
            mr = project.get_model_registry()
            aqi_model = mr.python.create_model(
                name="aqi_predictor_model",
                metrics=metrics_summary,
                description="Random Forest 3-Day AQI Forecaster"
            )
            aqi_model.save(MODELS_DIR)
            print("[Training Pipeline] Successfully registered trained model bundle in Hopsworks Model Registry!")
        except Exception as e:
            print(f"[Training Pipeline] Warning: Hopsworks Model Registry upload skipped ({e})")

    print("=" * 60)
    print("[Training Pipeline] Training pipeline completed successfully!")
    print("=" * 60)
    return metrics_summary

if __name__ == "__main__":
    run_training_pipeline()

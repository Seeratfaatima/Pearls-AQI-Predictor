#!/usr/bin/env python3
"""
Feature Pipeline - Pearls AQI Predictor
Runs on schedule (or manually) to fetch raw weather/pollutant API data,
engineer lag & rolling features, and update the Hopsworks Feature Store.
"""

import os
import sys
import pandas as pd
from utils import fetch_open_meteo_data, engineer_features, get_hopsworks_project

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def run_feature_pipeline(start_date="2024-07-01", end_date=None):
    print("=" * 60)
    print(f"[Feature Pipeline] Starting data fetch from {start_date} to {end_date or 'today'}...")
    print("=" * 60)

    raw_df = fetch_open_meteo_data(start_date=start_date, end_date=end_date)
    print(f"[Feature Pipeline] Fetched {len(raw_df)} raw hourly rows.")

    processed_df = engineer_features(raw_df, is_training=True)
    clean_df = processed_df.dropna(subset=["aqi"]).reset_index(drop=True)
    print(f"[Feature Pipeline] Engineered {len(clean_df.columns)} features over {len(clean_df)} rows.")

    # Save local Parquet snapshot
    local_path = os.path.join(DATA_DIR, "aqi_features.parquet")
    clean_df.to_parquet(local_path, index=False)
    print(f"[Feature Pipeline] Saved local snapshot to {local_path}")

    # Hopsworks Feature Store Sync
    project = get_hopsworks_project()
    if project:
        try:
            fs = project.get_feature_store()
            fg = fs.get_or_create_feature_group(
                name="aqi_features",
                version=4,
                description="Historical AQI features with engineered rolling variables",
                primary_key=["time"],
                event_time="time",
                online_enabled=False
            )
            fg.insert(clean_df, write_options={"wait_for_job": True})
            print("[Feature Pipeline] Successfully inserted features into Hopsworks Feature Group version 4!")
        except Exception as e:
            print(f"[Feature Pipeline] Warning: Could not push to Hopsworks Feature Store: {e}")

    print("=" * 60)
    print("[Feature Pipeline] Feature pipeline execution completed successfully!")
    print("=" * 60)
    return clean_df

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-07-01"
    run_feature_pipeline(start_date=start)

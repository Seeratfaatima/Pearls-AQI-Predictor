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
                version=5,
                description="Historical AQI features with engineered rolling variables and pressure",
                primary_key=["time"],
                event_time="time",
                online_enabled=False
            )
            # Use REST ingestion engine to bypass external HDFS socket restrictions
            try:
                import math
                from hsfs import engine, util
                from hsfs.core import dataset_api
                eng = engine._get_instance()
                app_options = eng._get_app_options({"wait_for_job": False})
                ingestion_job = eng._feature_group_api._ingestion(fg, app_options)
                
                ds_api = dataset_api.DatasetApi()
                df_parquet = clean_df.to_parquet(index=False)
                parquet_length = len(df_parquet)
                num_chunks = math.ceil(parquet_length / ds_api.DEFAULT_FLOW_CHUNK_SIZE)
                fg_name = util._feature_group_name(fg)
                base_params = ds_api._get_flow_base_params(fg_name, num_chunks, parquet_length, ds_api.DEFAULT_FLOW_CHUNK_SIZE)
                
                chunk_number = 1
                for i in range(0, parquet_length, ds_api.DEFAULT_FLOW_CHUNK_SIZE):
                    query_params = base_params.copy()
                    query_params["flowCurrentChunkSize"] = len(df_parquet[i : i + ds_api.DEFAULT_FLOW_CHUNK_SIZE])
                    query_params["flowChunkNumber"] = chunk_number
                    ds_api._upload_request(
                        query_params,
                        ingestion_job.data_path,
                        fg_name,
                        df_parquet[i : i + ds_api.DEFAULT_FLOW_CHUNK_SIZE]
                    )
                    chunk_number += 1
                
                ingestion_job.job.run(await_termination=False)
                print("[Feature Pipeline] Successfully launched Hopsworks ingestion job for Feature Group v5!")
            except Exception as ing_err:
                print(f"[Feature Pipeline] Direct REST ingestion fallback attempt: {ing_err}")
                fg.insert(clean_df, write_options={"wait_for_job": False})
                print("[Feature Pipeline] Pushed features to Hopsworks Feature Group version 5!")
        except Exception as e:
            print(f"[Feature Pipeline] Warning: Could not push to Hopsworks Feature Store: {e}")

    print("=" * 60)
    print("[Feature Pipeline] Feature pipeline execution completed successfully!")
    print("=" * 60)
    return clean_df

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-07-01"
    run_feature_pipeline(start_date=start)

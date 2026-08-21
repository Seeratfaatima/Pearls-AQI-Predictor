import datetime
import numpy as np
import pandas as pd
import requests

LATITUDE = 31.5497
LONGITUDE = 74.3436
HOPSWORKS_PROJECT_NAME = "aqi_prediction_01"
HOPSWORKS_API_KEY = "RblzyVBAnPDoQuMd.QrdgNiJOf3hJJSPFfPSn6GiXS64GBuPzdQmm5kgVgrZnSsRGIzPfEMxMGDvnBYha"

def calculate_pm25_aqi(pm25):
    """Calculates US EPA Air Quality Index from PM2.5 concentration in ug/m3."""
    if pd.isna(pm25):
        return np.nan
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    if pm25 > 500.4:
        return 500
    return np.nan

def get_aqi_category(aqi_val):
    """Returns AQI safety status category and color hex."""
    if pd.isna(aqi_val):
        return "Unknown", "#888888"
    aqi_val = float(aqi_val)
    if aqi_val <= 50:
        return "Good", "#00e400"
    elif aqi_val <= 100:
        return "Moderate", "#ffff00"
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi_val <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi_val <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"

def fetch_open_meteo_data(start_date="2024-07-01", end_date=None, lat=LATITUDE, lon=LONGITUDE):
    """Fetches combined weather and air quality historical/forecast data from Open-Meteo."""
    if end_date is None:
        end_date = datetime.date.today().strftime("%Y-%m-%d")

    # Use archive API for past dates, forecast API for recent/future dates
    is_recent = pd.to_datetime(start_date) > (pd.to_datetime(datetime.date.today()) - pd.Timedelta(days=5))
    w_base = "https://api.open-meteo.com/v1/forecast" if is_recent else "https://archive-api.open-meteo.com/v1/archive"
    a_base = "https://air-quality-api.open-meteo.com/v1/air-quality"

    w_url = f"{w_base}?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    a_url = f"{a_base}?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone"

    w_res = requests.get(w_url).json()
    a_res = requests.get(a_url).json()

    if "hourly" not in w_res or "hourly" not in a_res:
        raise ValueError(f"Failed to fetch Open-Meteo data: {w_res.get('reason', a_res.get('reason', 'API error'))}")

    w_df = pd.DataFrame(w_res["hourly"])
    a_df = pd.DataFrame(a_res["hourly"])

    merged = pd.merge(w_df, a_df, on="time").rename(columns={
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2"
    })
    return merged

def engineer_features(df, is_training=True):
    """Computes time-based, rolling, and lag features for AQI model inputs."""
    aqi_df = df.copy()
    aqi_df["time"] = pd.to_datetime(aqi_df["time"]).dt.floor("s")
    
    # Sort chronologically
    aqi_df = aqi_df.sort_values("time").reset_index(drop=True)

    aqi_df["hour"] = aqi_df["time"].dt.hour
    aqi_df["day"] = aqi_df["time"].dt.day
    aqi_df["month"] = aqi_df["time"].dt.month
    aqi_df["day_of_week"] = aqi_df["time"].dt.dayofweek

    aqi_df["hour_sin"] = np.sin(2 * np.pi * aqi_df["hour"] / 24)
    aqi_df["hour_cos"] = np.cos(2 * np.pi * aqi_df["hour"] / 24)
    aqi_df["month_sin"] = np.sin(2 * np.pi * aqi_df["month"] / 12)
    aqi_df["month_cos"] = np.cos(2 * np.pi * aqi_df["month"] / 12)

    aqi_df["aqi"] = aqi_df["pm2_5"].apply(calculate_pm25_aqi)

    # Multi-window rolling statistics
    for window in [2, 3, 6, 12, 24, 48, 72]:
        aqi_df[f"aqi_roll_mean_{window}"] = aqi_df["aqi"].rolling(window).mean()
        aqi_df[f"aqi_roll_std_{window}"] = aqi_df["aqi"].rolling(window).std()
        aqi_df[f"pm25_roll_mean_{window}"] = aqi_df["pm2_5"].rolling(window).mean()

    # Multi-lag features
    for lag in [1, 2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 72]:
        aqi_df[f"aqi_lag_{lag}"] = aqi_df["aqi"].shift(lag)
        aqi_df[f"pm25_lag_{lag}"] = aqi_df["pm2_5"].shift(lag)

    aqi_df["pm25_change_rate"] = aqi_df["pm2_5"].diff()
    aqi_df["temp_change"] = aqi_df["temperature"].diff()
    aqi_df["humidity_change"] = aqi_df["humidity"].diff()
    aqi_df["wind_speed_change"] = aqi_df["wind_speed"].diff()

    if is_training:
        # Generate multi-day target averages
        aqi_df["aqi_day1"] = aqi_df["aqi"].shift(-1).rolling(24, min_periods=24).mean().shift(-23)
        aqi_df["aqi_day2"] = aqi_df["aqi"].shift(-25).rolling(24, min_periods=24).mean().shift(-23)
        aqi_df["aqi_day3"] = aqi_df["aqi"].shift(-49).rolling(24, min_periods=24).mean().shift(-23)

    return aqi_df

import os

def get_hopsworks_project():
    """Attempts Hopsworks login and returns project instance."""
    try:
        import hopsworks
        api_key = os.environ.get("HOPSWORKS_API_KEY", HOPSWORKS_API_KEY)
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT_NAME,
            api_key_value=api_key
        )
        return project
    except Exception as e:
        print(f"[Hopsworks] Warning: Connection skipped ({e})")
        return None

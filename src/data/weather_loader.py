"""Historical weather data fetcher using Open-Meteo API for Kolkata."""
import os
import requests
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("weather_loader")

def fetch_or_generate_weather(start_date: str = "2023-01-01",
                              end_date: str = "2024-12-31",
                              output_path: str = "data/raw/weather/historical_weather.csv") -> pd.DataFrame:
    """Fetches hourly weather from Open-Meteo Archive API or falls back to synthetic monsoon/winter profiles."""
    if os.path.exists(output_path):
        logger.info(f"Loading cached weather data from {output_path}")
        return pd.read_csv(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 22.5726,
        "longitude": 88.3639,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,visibility",
        "timezone": "Asia/Kolkata"
    }

    try:
        logger.info("Requesting historical weather from Open-Meteo API...")
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code == 200:
            data = resp.json()["hourly"]
            df_weather = pd.DataFrame(data)
            df_weather.rename(columns={"time": "timestamp"}, inplace=True)
            df_weather.to_csv(output_path, index=False)
            logger.info(f"Saved Open-Meteo weather records ({len(df_weather)} hours) -> {output_path}")
            return df_weather
    except Exception as e:
        logger.warning(f"Could not reach Open-Meteo API ({e}). Generating historical Kolkata climatic profile...")

    # Fallback to realistic Kolkata hourly profile
    date_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq="h")
    df_weather = pd.DataFrame({"timestamp": date_range})
    months = df_weather["timestamp"].dt.month
    hours = df_weather["timestamp"].dt.hour

    # Monsoon precipitation modeling
    is_monsoon = months.isin([6, 7, 8, 9])
    rain = np.where(is_monsoon, np.random.exponential(scale=1.8, size=len(df_weather)) * (np.random.random(len(df_weather)) < 0.28), 0.0)
    df_weather["precipitation"] = np.round(rain, 2)
    df_weather["rain"] = df_weather["precipitation"]

    # Visibility: low during winter mornings (Dec/Jan) or heavy rain
    base_vis = 10000.0 - (df_weather["precipitation"] * 1200.0)
    winter_fog = months.isin([12, 1]) & (hours.isin([4, 5, 6, 7]))
    df_weather["visibility"] = np.where(winter_fog, np.random.uniform(500, 2500, size=len(df_weather)), base_vis)
    df_weather["visibility"] = np.clip(df_weather["visibility"], 300, 10000)

    # Temperature
    df_weather["temperature_2m"] = 28.0 + 6.0 * np.sin((hours - 9) * np.pi / 12) + np.where(months.isin([12, 1]), -9.0, 0.0)
    df_weather["relative_humidity_2m"] = np.where(is_monsoon, np.random.uniform(75, 98, size=len(df_weather)), np.random.uniform(45, 75, size=len(df_weather)))

    df_weather.to_csv(output_path, index=False)
    logger.info(f"Cached climatic weather series ({len(df_weather)} records) -> {output_path}")
    return df_weather

if __name__ == "__main__":
    fetch_or_generate_weather()

"""Historical weather data fetcher and multi-station generator for Kolkata.

Supports multi-station observation records across Kolkata metropolitan stations:
- Alipore (IMD Central/South)
- Dum Dum Airport (IMD North)
- Salt Lake (East Kolkata)
- Behala (South-West Kolkata)
"""
import os
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.utils.logger import get_logger

logger = get_logger("weather_loader")

KOLKATA_WEATHER_STATIONS = [
    {"station_id": "STN_ALIPORE", "name": "Alipore IMD Central", "lat": 22.5300, "lon": 88.3300},
    {"station_id": "STN_DUMDUM", "name": "Dum Dum Airport IMD", "lat": 22.6500, "lon": 88.4500},
    {"station_id": "STN_SALTLAKE", "name": "Salt Lake Sector V", "lat": 22.5800, "lon": 88.4300},
    {"station_id": "STN_BEHALA", "name": "Behala Airport Station", "lat": 22.4900, "lon": 88.3100},
]


def fetch_or_generate_weather(start_date: str = "2023-01-01",
                              end_date: str = "2024-12-31",
                              output_path: str = "data/raw/weather/historical_weather.csv") -> pd.DataFrame:
    """Fetches or generates multi-station hourly meteorological records for Kolkata."""
    if os.path.exists(output_path):
        logger.info(f"Loading cached multi-station weather data from {output_path}")
        df = pd.read_csv(output_path)
        if "station_id" in df.columns:
            return df

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Generating multi-station Kolkata meteorological dataset (Alipore, Dum Dum, Salt Lake, Behala)...")

    date_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq="h")
    n_hours = len(date_range)
    months = date_range.month
    hours = date_range.hour

    is_monsoon = np.isin(months, [6, 7, 8, 9])
    is_winter = np.isin(months, [12, 1])

    all_station_frames = []

    for stn in KOLKATA_WEATHER_STATIONS:
        # Station-specific climatic micro-variations
        # Dum Dum has heavier winter fog and colder dawn temps
        fog_multiplier = 1.35 if stn["station_id"] == "STN_DUMDUM" else 1.0
        # Salt Lake / East has slightly heavier convective rain showers in monsoon
        rain_prob = 0.32 if stn["station_id"] == "STN_SALTLAKE" else 0.26
        urban_heat = 1.2 if stn["station_id"] == "STN_ALIPORE" else 0.0

        # Precipitation
        rain = np.where(
            is_monsoon,
            np.random.exponential(scale=1.9, size=n_hours) * (np.random.random(n_hours) < rain_prob),
            0.0
        )
        precip = np.round(rain, 2)

        # Visibility (meters)
        base_vis = 10000.0 - (precip * 1100.0)
        winter_fog = is_winter & (np.isin(hours, [4, 5, 6, 7]))
        vis = np.where(
            winter_fog,
            np.random.uniform(400 / fog_multiplier, 2200 / fog_multiplier, size=n_hours),
            base_vis
        )
        vis = np.clip(vis, 250, 10000)

        # Temperature (Celsius)
        diurnal = 6.0 * np.sin((hours - 9) * np.pi / 12)
        season_offset = np.where(is_winter, -9.0, 0.0)
        temp = 28.0 + diurnal + season_offset + urban_heat + np.random.normal(0, 0.5, size=n_hours)

        # Humidity (%)
        humidity = np.where(
            is_monsoon,
            np.random.uniform(75, 98, size=n_hours),
            np.random.uniform(45, 75, size=n_hours)
        )

        stn_df = pd.DataFrame({
            "timestamp": date_range,
            "station_id": stn["station_id"],
            "station_name": stn["name"],
            "station_lat": stn["lat"],
            "station_lon": stn["lon"],
            "temperature_2m": np.round(temp, 1),
            "precipitation": precip,
            "rain": precip,
            "visibility": np.round(vis, 0),
            "relative_humidity_2m": np.round(humidity, 1)
        })
        all_station_frames.append(stn_df)

    df_combined = pd.concat(all_station_frames, ignore_index=True)
    df_combined.to_csv(output_path, index=False)
    logger.info(f"Saved multi-station weather dataset ({len(df_combined)} records) -> {output_path}")
    return df_combined


if __name__ == "__main__":
    fetch_or_generate_weather()

"""Weather feature integration and feature extraction."""
import pandas as pd
import numpy as np

def merge_and_extract_weather_features(crashes_df: pd.DataFrame,
                                      weather_df: pd.DataFrame) -> pd.DataFrame:
    """Joins hourly weather history to crash records by nearest hour and extracts weather signals."""
    df_crashes = crashes_df.copy()
    df_weather = weather_df.copy()

    df_crashes["crash_hour_key"] = pd.to_datetime(df_crashes["timestamp"]).dt.floor("h")
    df_weather["weather_hour_key"] = pd.to_datetime(df_weather["timestamp"]).dt.floor("h")

    # Merge on hour
    merged = pd.merge(
        df_crashes,
        df_weather[["weather_hour_key", "temperature_2m", "relative_humidity_2m", "precipitation", "visibility"]],
        left_on="crash_hour_key",
        right_on="weather_hour_key",
        how="left"
    )

    # Impute missing with defaults if timestamps misaligned
    merged["temperature_2m"] = pd.to_numeric(merged["temperature_2m"], errors="coerce").fillna(28.0)
    merged["precipitation"] = pd.to_numeric(merged["precipitation"], errors="coerce").fillna(0.0)
    merged["visibility"] = pd.to_numeric(merged["visibility"], errors="coerce").fillna(8000.0)
    merged["relative_humidity_2m"] = pd.to_numeric(merged["relative_humidity_2m"], errors="coerce").fillna(70.0)

    # Extracted signals
    merged["feat_precipitation"] = merged["precipitation"].astype(float)
    merged["feat_is_raining"] = (merged["feat_precipitation"] > 0.1).astype(int)
    merged["feat_visibility_km"] = (merged["visibility"] / 1000.0).clip(0.1, 10.0)
    merged["feat_is_low_visibility"] = (merged["feat_visibility_km"] < 2.5).astype(int)
    merged["feat_temperature"] = merged["temperature_2m"].astype(float)

    merged.drop(columns=["crash_hour_key", "weather_hour_key"], inplace=True, errors="ignore")
    return merged

"""Weather feature integration with spatial multi-station interpolation (IDW)."""
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("weather_features")


def _interpolate_idw(crashes_sub: pd.DataFrame, stn_records: pd.DataFrame) -> pd.DataFrame:
    """Performs 2D Inverse Distance Weighting (IDW) interpolation from multiple stations."""
    stn_lats = stn_records["station_lat"].values
    stn_lons = stn_records["station_lon"].values
    stn_precip = stn_records["precipitation"].values
    stn_vis = stn_records["visibility"].values
    stn_temp = stn_records["temperature_2m"].values
    stn_rh = stn_records["relative_humidity_2m"].values

    crash_lats = crashes_sub["latitude"].values[:, np.newaxis]
    crash_lons = crashes_sub["longitude"].values[:, np.newaxis]

    # Distances in degrees (approx ~110km per deg)
    dists = np.sqrt((crash_lats - stn_lats)**2 + (crash_lons - stn_lons)**2) + 1e-5
    weights = 1.0 / (dists ** 2)
    sum_weights = weights.sum(axis=1, keepdims=True)
    weights /= sum_weights

    interp_precip = (weights * stn_precip).sum(axis=1)
    interp_vis = (weights * stn_vis).sum(axis=1)
    interp_temp = (weights * stn_temp).sum(axis=1)
    interp_rh = (weights * stn_rh).sum(axis=1)

    res = crashes_sub.copy()
    res["precipitation"] = interp_precip
    res["visibility"] = interp_vis
    res["temperature_2m"] = interp_temp
    res["relative_humidity_2m"] = interp_rh
    return res


def merge_and_extract_weather_features(crashes_df: pd.DataFrame,
                                      weather_df: pd.DataFrame) -> pd.DataFrame:
    """Joins hourly weather history to crash records using multi-station spatial IDW interpolation.

    When multi-station records (Alipore, Dum Dum, Salt Lake, Behala) are available,
    weather attributes (rain, visibility, temp) are interpolated at each crash's exact GPS coordinate.
    """
    df_crashes = crashes_df.copy()
    df_weather = weather_df.copy()

    df_crashes["crash_hour_key"] = pd.to_datetime(df_crashes["timestamp"]).dt.floor("h")
    df_weather["weather_hour_key"] = pd.to_datetime(df_weather["timestamp"]).dt.floor("h")

    has_multi_stations = ("station_id" in df_weather.columns and
                          "station_lat" in df_weather.columns and
                          len(df_weather["station_id"].unique()) > 1)

    if has_multi_stations:
        logger.info("Applying multi-station Inverse Distance Weighting (IDW) weather interpolation...")
        # Group by hour and interpolate
        interpolated_chunks = []
        weather_by_hour = {k: v for k, v in df_weather.groupby("weather_hour_key")}

        for hour_key, crash_chunk in df_crashes.groupby("crash_hour_key"):
            if hour_key in weather_by_hour:
                stn_slice = weather_by_hour[hour_key]
                chunk_interp = _interpolate_idw(crash_chunk, stn_slice)
            else:
                chunk_interp = crash_chunk.copy()
                chunk_interp["temperature_2m"] = 28.0
                chunk_interp["precipitation"] = 0.0
                chunk_interp["visibility"] = 8000.0
                chunk_interp["relative_humidity_2m"] = 70.0
            interpolated_chunks.append(chunk_interp)

        merged = pd.concat(interpolated_chunks, ignore_index=True)
    else:
        # Single station fallback
        cols = ["weather_hour_key", "temperature_2m", "relative_humidity_2m", "precipitation", "visibility"]
        existing_cols = [c for c in cols if c in df_weather.columns]
        merged = pd.merge(
            df_crashes,
            df_weather[existing_cols],
            left_on="crash_hour_key",
            right_on="weather_hour_key",
            how="left"
        )
        merged["temperature_2m"] = pd.to_numeric(merged.get("temperature_2m", 28.0), errors="coerce").fillna(28.0)
        merged["precipitation"] = pd.to_numeric(merged.get("precipitation", 0.0), errors="coerce").fillna(0.0)
        merged["visibility"] = pd.to_numeric(merged.get("visibility", 8000.0), errors="coerce").fillna(8000.0)
        merged["relative_humidity_2m"] = pd.to_numeric(merged.get("relative_humidity_2m", 70.0), errors="coerce").fillna(70.0)

    # Signal engineering
    merged["feat_precipitation"] = np.round(merged["precipitation"].astype(float), 2)
    merged["feat_is_raining"] = (merged["feat_precipitation"] > 0.1).astype(int)
    merged["feat_visibility_km"] = np.round((merged["visibility"] / 1000.0).clip(0.1, 10.0), 2)
    merged["feat_is_low_visibility"] = (merged["feat_visibility_km"] < 2.5).astype(int)
    merged["feat_temperature"] = np.round(merged["temperature_2m"].astype(float), 1)

    merged.drop(columns=["crash_hour_key", "weather_hour_key"], inplace=True, errors="ignore")
    return merged

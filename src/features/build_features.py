"""Orchestrates Stage 4: Compiles master feature matrix for Level 1 severity model."""
import os
import pandas as pd
from src.utils.logger import get_logger
from src.features.road_features import encode_road_features
from src.features.temporal_features import extract_temporal_features
from src.features.weather_features import merge_and_extract_weather_features

logger = get_logger("build_features")

FEATURE_COLUMNS = [
    # Road network features
    "feat_highway_hierarchy",
    "feat_speed_limit",
    "feat_is_high_speed",
    "feat_lanes",
    "feat_has_median",
    "feat_is_lit",
    "feat_is_junction",
    "feat_junction_density",
    # Temporal features
    "feat_hour",
    "feat_dayofweek",
    "feat_is_weekend",
    "feat_hour_sin",
    "feat_hour_cos",
    "feat_is_night",
    "feat_is_monsoon",
    "feat_is_festive_season",
    # Weather features
    "feat_precipitation",
    "feat_is_raining",
    "feat_visibility_km",
    "feat_is_low_visibility",
    "feat_temperature",
    # Vulnerable road users / participant flags
    "feat_is_two_wheeler",
    "feat_pedestrian_involved",
    "feat_is_heavy_vehicle",
    # Physics & infrastructure interaction features
    "feat_kinetic_energy",
    "feat_vulnerable_high_speed",
    "feat_unlit_night",
    "feat_wet_high_speed"
]

def build_feature_matrix(snapped_crashes_df: pd.DataFrame,
                         weather_df: pd.DataFrame,
                         output_parquet: str = "data/processed/crash_level_features.parquet") -> pd.DataFrame:
    """Builds full feature dataset and saves to processed parquet."""
    logger.info("Building full feature set for severity modeling...")

    df = encode_road_features(snapped_crashes_df)
    df = extract_temporal_features(df)
    df = merge_and_extract_weather_features(df, weather_df)

    # Participant features
    if "primary_vehicle" in df.columns:
        df["feat_is_two_wheeler"] = df["primary_vehicle"].str.lower().str.contains("two-wheeler").astype(int)
        df["feat_is_heavy_vehicle"] = df["primary_vehicle"].str.lower().str.contains("truck|bus").astype(int)
    else:
        df["feat_is_two_wheeler"] = 0
        df["feat_is_heavy_vehicle"] = 0

    if "pedestrian_involved" in df.columns:
        df["feat_pedestrian_involved"] = df["pedestrian_involved"].fillna(0).astype(int)
    else:
        df["feat_pedestrian_involved"] = 0

    # Domain interaction features
    df["feat_kinetic_energy"] = (df["feat_speed_limit"].astype(float) ** 2) / 1000.0
    df["feat_vulnerable_high_speed"] = (
        (df["feat_is_two_wheeler"] | df["feat_pedestrian_involved"]) & df["feat_is_high_speed"]
    ).astype(int)
    df["feat_unlit_night"] = ((1 - df["feat_is_lit"]) & df["feat_is_night"]).astype(int)
    df["feat_wet_high_speed"] = (df["feat_is_raining"] & df["feat_is_high_speed"]).astype(int)

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Feature compilation complete: {len(df)} records with {len(FEATURE_COLUMNS)} features -> {output_parquet}")
    return df

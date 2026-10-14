"""Stage 3: Crash data cleaning, standardization, and spatial bounding."""
import os
import pandas as pd
from src.utils.logger import get_logger
from src.utils.geo_utils import is_within_kolkata_bounds

logger = get_logger("crash_cleaner")

def clean_crash_data(input_csv: str = "data/raw/crashes/crashes.csv",
                     output_parquet: str = "data/interim/cleaned_crashes.parquet") -> pd.DataFrame:
    """Reads raw crash events, cleans timestamps, validates coordinates and standardizes severity."""
    logger.info(f"Loading crash records from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}. Run mock generator first if raw data is pending.")

    df = pd.read_csv(input_csv)
    initial_count = len(df)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Timestamp conversion
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])

    # Spatial bounding check (Kolkata study area)
    valid_coords = df.apply(
        lambda r: is_within_kolkata_bounds(r["latitude"], r["longitude"]),
        axis=1
    )
    df = df[valid_coords].copy()
    dropped_coords = initial_count - len(df)
    if dropped_coords > 0:
        logger.warning(f"Filtered out {dropped_coords} records outside Kolkata bounds.")

    # Standardize severity code (0: Minor, 1: Grievous, 2: Fatal)
    if "severity_code" not in df.columns and "severity_label" in df.columns:
        mapping = {
            "minor": 0, "minor / non-injury": 0, "non-injury": 0,
            "grievous": 1, "grievous injury": 1, "serious": 1,
            "fatal": 2, "death": 2
        }
        df["severity_code"] = df["severity_label"].str.lower().map(mapping).fillna(0).astype(int)

    # Ensure spatial zone grouping ID exists
    if "spatial_zone_id" not in df.columns:
        if "ward_number" in df.columns:
            df["spatial_zone_id"] = "Ward_" + df["ward_number"].astype(str)
        else:
            df["spatial_zone_id"] = "Zone_Default"

    # Fill missing participant info with 'Unknown' instead of silent dropping
    for col in ["primary_vehicle", "weather_condition"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Cleaned {len(df)} records saved to {output_parquet}")
    return df

if __name__ == "__main__":
    clean_crash_data()

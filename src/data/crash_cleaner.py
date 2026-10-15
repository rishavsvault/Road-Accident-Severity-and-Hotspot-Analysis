"""Stage 3: Crash data cleaning, standardization, and spatial bounding.

Uses precise point-in-polygon test against Kolkata metropolitan boundary polygon
instead of a simple rectangular bounding-box clip.
"""
import os
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("crash_cleaner")

# Kolkata Metropolitan Area / KMC administrative boundary polygon (WGS84 lat/lon)
# Tracing Cossipore -> Dum Dum -> Salt Lake -> EM Bypass -> Garia -> Behala -> Riverfront
KOLKATA_POLYGON_COORDS = [
    (22.6450, 88.3700),  # North Cossipore / Baranagar
    (22.6400, 88.4200),  # Dum Dum / Nagerbazar
    (22.6100, 88.4550),  # Lake Town / Ultadanga East
    (22.5850, 88.4450),  # Salt Lake Sector I-III
    (22.5700, 88.4400),  # Salt Lake Sector V / Chingrighata
    (22.5350, 88.4200),  # Science City / Topsia
    (22.5100, 88.4100),  # Ruby / Anandapur (EM Bypass)
    (22.4800, 88.4000),  # Mukundapur / Kalikapur
    (22.4550, 88.3850),  # Garia / Kavi Nazrul
    (22.4500, 88.3500),  # Tollygunge / Kudghat
    (22.4600, 88.3100),  # Thakurpukur / Diamond Harbour Rd
    (22.4900, 88.3000),  # Behala Chowrasta
    (22.5200, 88.3100),  # Taratala / Majerhat
    (22.5400, 88.3150),  # Kidderpore / Garden Reach
    (22.5650, 88.3350),  # Maidan / River Hooghly East Bank
    (22.5850, 88.3480),  # BBD Bagh / Burrabazar / Strand Rd
    (22.6150, 88.3650),  # Bagbazar / Shyambazar
    (22.6450, 88.3700),  # Closing vertex
]

try:
    from shapely.geometry import Point as _ShapelyPoint, Polygon as _ShapelyPolygon
    _KMC_SHAPELY_POLYGON = _ShapelyPolygon([(lon, lat) for lat, lon in KOLKATA_POLYGON_COORDS])
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False
    _KMC_SHAPELY_POLYGON = None

def _point_in_polygon(lat: float, lon: float) -> bool:
    """Multi-vertex polygon point-in-polygon test for Kolkata boundary.
    Uses Shapely if available, with robust ray-casting fallback.
    """
    if _HAS_SHAPELY and _KMC_SHAPELY_POLYGON is not None:
        return _KMC_SHAPELY_POLYGON.contains(_ShapelyPoint(lon, lat)) or _KMC_SHAPELY_POLYGON.touches(_ShapelyPoint(lon, lat))

    # Ray-casting fallback
    poly = KOLKATA_POLYGON_COORDS
    n = len(poly) - 1
    inside = False
    x, y = lon, lat
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][1], poly[i][0]
        xj, yj = poly[j][1], poly[j][0]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def clean_crash_data(input_csv: str = "data/raw/crashes/crashes.csv",
                     output_parquet: str = "data/interim/cleaned_crashes.parquet") -> pd.DataFrame:
    """Reads raw crash events, cleans timestamps, validates coordinates and standardizes severity."""
    logger.info(f"Loading crash records from {input_csv}...")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input file not found: {input_csv}. Run mock generator first if raw data is pending."
        )

    df = pd.read_csv(input_csv)
    initial_count = len(df)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Timestamp conversion
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "latitude", "longitude"])

    # Convert coordinate columns to numeric, drop non-parseable
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Point-in-polygon spatial validity check (replaces bounding-box crop)
    valid_mask = df.apply(
        lambda r: _point_in_polygon(r["latitude"], r["longitude"]),
        axis=1
    )
    df = df[valid_mask].copy()
    dropped_coords = initial_count - len(df)
    if dropped_coords > 0:
        logger.warning(
            f"Removed {dropped_coords} records outside Kolkata polygon boundary "
            f"({initial_count} -> {len(df)})."
        )

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

    # Remove duplicate crash records
    before_dedup = len(df)
    if "crash_id" in df.columns:
        df = df.drop_duplicates(subset=["crash_id"]).reset_index(drop=True)
    deduped = before_dedup - len(df)
    if deduped > 0:
        logger.info(f"Removed {deduped} duplicate crash records.")

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Cleaned {len(df)} records saved to {output_parquet}")
    return df

if __name__ == "__main__":
    clean_crash_data()

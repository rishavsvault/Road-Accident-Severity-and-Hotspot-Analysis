"""Snaps point crash records to nearest road network segment."""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from src.utils.logger import get_logger

logger = get_logger("spatial_snapper")

def snap_crashes_to_segments(crashes_df: pd.DataFrame,
                             segments_df: pd.DataFrame,
                             max_distance_meters: float = 300.0) -> pd.DataFrame:
    """Uses metric KDTree projection to snap each crash point to the nearest road segment ID."""
    logger.info(f"Snapping {len(crashes_df)} crash events to {len(segments_df)} road segments...")

    # Convert coordinates to approximate metric flat plane for fast, accurate matching
    # 1 deg lat ~ 111,000m, 1 deg lon at lat 22.5 deg ~ 102,500m
    lat_scale = 111000.0
    lon_scale = 102500.0

    segment_coords = np.column_stack([
        segments_df["mid_lat"].values * lat_scale,
        segments_df["mid_lon"].values * lon_scale
    ])
    tree = cKDTree(segment_coords)

    crash_coords = np.column_stack([
        crashes_df["latitude"].values * lat_scale,
        crashes_df["longitude"].values * lon_scale
    ])

    distances, indices = tree.query(crash_coords, k=1)

    matched_segments = segments_df.iloc[indices].copy().reset_index(drop=True)
    snapped_df = crashes_df.copy().reset_index(drop=True)

    # Attach segment attributes
    snapped_df["segment_id"] = matched_segments["segment_id"]
    snapped_df["segment_highway_type"] = matched_segments["highway_type"]
    snapped_df["segment_speed_limit"] = matched_segments["speed_limit"]
    snapped_df["segment_lanes"] = matched_segments["lanes"]
    snapped_df["segment_has_median"] = matched_segments["has_median"]
    snapped_df["segment_is_lit"] = matched_segments["is_lit"]
    snapped_df["segment_is_junction"] = matched_segments["is_junction"]
    snapped_df["segment_junction_density"] = matched_segments["junction_density_score"]
    snapped_df["snap_distance_meters"] = np.round(distances, 1)

    # Filter out crashes too far from any identified network segment if beyond threshold
    valid_snaps = snapped_df["snap_distance_meters"] <= max_distance_meters
    snapped_df = snapped_df[valid_snaps].copy()

    logger.info(f"Successfully snapped {len(snapped_df)} crashes. Median snap distance: {snapped_df['snap_distance_meters'].median():.1f}m")
    return snapped_df

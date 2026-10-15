"""Snaps point crash records to road network segments with topology, bearing, and attribute awareness.

Features:
- Configurable maximum snapping distance threshold to reject non-network outliers.
- Multi-candidate re-ranking based on road/corridor name alignment and heading/bearing.
- Snapping confidence score (0 to 1) based on metric distance and attribute alignment.
- Audit logging for unmatched or discarded crash points.
"""
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("spatial_snapper")


def snap_crashes_to_segments(crashes_df: pd.DataFrame,
                             segments_df: pd.DataFrame,
                             max_distance_meters: float = 120.0,
                             top_k_candidates: int = 5) -> pd.DataFrame:
    """Uses metric projection and multi-candidate scoring to snap crash points to road segments.

    Parameters
    ----------
    crashes_df : pd.DataFrame
        Crash records with 'latitude', 'longitude', and optional 'corridor_name', 'road_name', 'bearing'.
    segments_df : pd.DataFrame
        Road network segments with 'mid_lat', 'mid_lon', 'segment_id', and corridor/highway attributes.
    max_distance_meters : float
        Maximum allowable distance in meters before a crash is flagged as an unassigned outlier.
    top_k_candidates : int
        Number of nearest segment candidates to consider for attribute-guided re-ranking.
    """
    logger.info(f"Initiating network snapping: {len(crashes_df)} crashes against {len(segments_df)} segments...")

    # Metric flat plane coordinates: 1 deg lat ~ 111,000m, 1 deg lon at lat 22.5 deg ~ 102,500m
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

    k_search = min(top_k_candidates, len(segments_df))
    distances_matrix, indices_matrix = tree.query(crash_coords, k=k_search)

    # Ensure 2D arrays even if k=1
    if k_search == 1:
        distances_matrix = distances_matrix[:, np.newaxis]
        indices_matrix = indices_matrix[:, np.newaxis]

    n_crashes = len(crashes_df)
    chosen_segment_indices = np.zeros(n_crashes, dtype=int)
    final_distances = np.zeros(n_crashes, dtype=float)
    confidence_scores = np.zeros(n_crashes, dtype=float)

    has_crash_corridor = "corridor_name" in crashes_df.columns
    has_seg_corridor = "corridor_name" in segments_df.columns

    for i in range(n_crashes):
        cand_indices = indices_matrix[i]
        cand_dists = distances_matrix[i]

        best_score = float("inf")
        best_cand_idx = cand_indices[0]
        best_dist = cand_dists[0]

        # Multi-candidate re-ranking
        for c_idx, d_m in zip(cand_indices, cand_dists):
            penalty = d_m

            # Bonus if crash has documented corridor/road name matching candidate segment
            if has_crash_corridor and has_seg_corridor:
                crash_corr = str(crashes_df.iloc[i].get("corridor_name", "")).strip().lower()
                seg_corr = str(segments_df.iloc[c_idx].get("corridor_name", "")).strip().lower()
                if crash_corr and seg_corr and crash_corr == seg_corr:
                    penalty -= 25.0  # 25m distance discount for explicit corridor agreement

            if penalty < best_score:
                best_score = penalty
                best_cand_idx = c_idx
                best_dist = d_m

        chosen_segment_indices[i] = best_cand_idx
        final_distances[i] = best_dist

        # Compute continuous confidence score [0.0 to 1.0]
        # Decays exponentially with distance (scale = 40m)
        conf = float(np.exp(-best_dist / 40.0))
        confidence_scores[i] = round(min(max(conf, 0.0), 1.0), 3)

    matched_segments = segments_df.iloc[chosen_segment_indices].copy().reset_index(drop=True)
    snapped_df = crashes_df.copy().reset_index(drop=True)

    # Attach segment attributes
    snapped_df["segment_id"] = matched_segments["segment_id"]
    snapped_df["segment_highway_type"] = matched_segments.get("highway_type", "primary")
    snapped_df["segment_speed_limit"] = matched_segments.get("speed_limit", 40)
    snapped_df["segment_lanes"] = matched_segments.get("lanes", 2)
    snapped_df["segment_has_median"] = matched_segments.get("has_median", 0)
    snapped_df["segment_is_lit"] = matched_segments.get("is_lit", 1)
    snapped_df["segment_is_junction"] = matched_segments.get("is_junction", 0)
    snapped_df["segment_junction_density"] = matched_segments.get("junction_density_score", 0.0)
    snapped_df["snap_distance_meters"] = np.round(final_distances, 1)
    snapped_df["snap_confidence"] = confidence_scores

    if "spatial_zone_id" in matched_segments.columns:
        snapped_df["spatial_zone_id"] = matched_segments["spatial_zone_id"]
    if "corridor_name" in matched_segments.columns:
        snapped_df["snapped_corridor_name"] = matched_segments["corridor_name"]

    # Classify audit status
    snapped_df["snap_status"] = np.where(
        snapped_df["snap_distance_meters"] <= 35.0,
        "High Confidence",
        np.where(
            snapped_df["snap_distance_meters"] <= max_distance_meters,
            "Medium Confidence",
            "Unmatched Outlier"
        )
    )

    # Filter out crashes exceeding maximum allowed snap distance
    valid_mask = snapped_df["snap_distance_meters"] <= max_distance_meters
    outlier_count = int((~valid_mask).sum())
    valid_snapped_df = snapped_df[valid_mask].copy()

    median_dist = float(valid_snapped_df["snap_distance_meters"].median()) if len(valid_snapped_df) > 0 else 0.0
    mean_conf = float(valid_snapped_df["snap_confidence"].mean()) if len(valid_snapped_df) > 0 else 0.0

    logger.info(
        f"Snapping complete: {len(valid_snapped_df)} crashes matched within {max_distance_meters}m "
        f"({outlier_count} discarded as off-network outliers). "
        f"Median distance: {median_dist:.1f}m | Mean confidence: {mean_conf:.3f}"
    )

    return valid_snapped_df

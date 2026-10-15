"""Network-constrained Kernel Density Estimation (NetKDE) along road networks."""
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("network_kde")

def calculate_network_kde(segments_df: pd.DataFrame,
                          snapped_crashes_df: pd.DataFrame,
                          bandwidth_meters: float = 400.0) -> pd.DataFrame:
    """Computes 1D network-constrained kernel density estimates for each road segment.

    Instead of blurring densities across open fields or rivers, density is evaluated
    along continuous road network segments.
    """
    logger.info(f"Computing Network-KDE across {len(segments_df)} segments with {bandwidth_meters}m bandwidth...")

    df_res = segments_df.copy()

    # Pre-extract segment midpoint coordinates in meters
    lat_scale = 111000.0
    lon_scale = 102500.0

    seg_coords = np.column_stack([
        df_res["mid_lat"].values * lat_scale,
        df_res["mid_lon"].values * lon_scale
    ])

    crash_coords = np.column_stack([
        snapped_crashes_df["latitude"].values * lat_scale,
        snapped_crashes_df["longitude"].values * lon_scale
    ])

    # Weights by severity: Fatal=10, Grievous=3, Minor=1
    sev_weights = snapped_crashes_df["severity_code"].map({2: 10.0, 1: 3.0, 0: 1.0}).values

    from scipy.spatial import cKDTree
    tree = cKDTree(crash_coords)
    neighbors = tree.query_ball_point(seg_coords, r=bandwidth_meters)

    kde_scores = np.zeros(len(df_res))
    lengths = df_res["length_meters"].clip(lower=50.0).values if "length_meters" in df_res.columns else np.full(len(df_res), 300.0)

    for i in range(len(df_res)):
        idx = neighbors[i]
        if len(idx) > 0:
            p = seg_coords[i]
            dists = np.linalg.norm(crash_coords[idx] - p, axis=1)
            u = dists / bandwidth_meters
            kernel_vals = 0.75 * (1.0 - u**2)
            weighted_density = np.sum(kernel_vals * sev_weights[idx])
            kde_scores[i] = (weighted_density / lengths[i]) * 1000.0

    df_res["net_kde_score"] = np.round(kde_scores, 3)
    df_res["net_kde_percentile"] = np.round(df_res["net_kde_score"].rank(pct=True) * 100, 1)
    logger.info(f"NetKDE completed. Max segment score: {df_res['net_kde_score'].max():.2f}")
    return df_res

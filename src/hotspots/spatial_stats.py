"""Spatial statistics: Getis-Ord Gi* hotspot detection, DBSCAN clustering, and Moran's I autocorrelation."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.cluster import DBSCAN
from typing import Dict, Any, Tuple
from src.utils.logger import get_logger

logger = get_logger("spatial_stats")


def compute_getis_ord_gi(segments_df: pd.DataFrame,
                         metric_col: str = "net_kde_score",
                         distance_threshold_m: float = 800.0) -> pd.DataFrame:
    """Calculates local Getis-Ord Gi* z-scores to identify statistically significant hot spots."""
    df = segments_df.copy()
    n = len(df)
    x = df[metric_col].values

    x_mean = np.mean(x)
    s = np.std(x)
    if s == 0:
        df["gi_z_score"] = 0.0
        df["is_statistically_significant_hotspot"] = 0
        return df

    lat_scale = 111000.0
    lon_scale = 102500.0
    coords = np.column_stack([df["mid_lat"].values * lat_scale, df["mid_lon"].values * lon_scale])

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=distance_threshold_m)

    gi_z = np.zeros(n)
    for i in range(n):
        idx = neighbors[i]
        w_sum = len(idx)
        if w_sum > 0 and s > 0:
            sum_x = np.sum(x[idx])
            numerator = sum_x - (x_mean * w_sum)
            denom_sq = (n * w_sum - (w_sum**2)) / (n - 1.0)
            if denom_sq > 0:
                gi_z[i] = numerator / (s * np.sqrt(denom_sq))

    df["gi_z_score"] = np.round(gi_z, 3)
    # Significant at 95% confidence if z >= 1.96
    df["is_statistically_significant_hotspot"] = (df["gi_z_score"] >= 1.96).astype(int)
    return df


def cluster_crash_points_dbscan(crashes_df: pd.DataFrame,
                                eps_meters: float = 150.0,
                                min_samples: int = 5) -> pd.DataFrame:
    """Clusters point crash events using spatial metric DBSCAN."""
    df = crashes_df.copy()
    coords_m = np.column_stack([
        df["latitude"].values * 111000.0,
        df["longitude"].values * 102500.0
    ])

    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords_m)
    df["spatial_cluster_id"] = clustering.labels_
    return df


def compute_global_morans_i(segments_df: pd.DataFrame,
                           metric_col: str = "total_crashes",
                           distance_threshold_m: float = 1000.0) -> Dict[str, Any]:
    """Calculates Global Moran's I to test for spatial autocorrelation across the road network.

    Returns observed Moran's I, expected I, variance under randomization, z-score, and two-tailed p-value.
    """
    df = segments_df.copy()
    n = len(df)
    if n < 4:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0, "interpretation": "Insufficient samples"}

    x = df[metric_col].values.astype(float)
    x_mean = np.mean(x)
    z = x - x_mean
    ss_total = np.sum(z**2)

    if ss_total == 0:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0, "interpretation": "Zero variance in metric"}

    lat_scale = 111000.0
    lon_scale = 102500.0
    coords = np.column_stack([df["mid_lat"].values * lat_scale, df["mid_lon"].values * lon_scale])

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=distance_threshold_m)

    # Build binary spatial weight matrix sum
    s0 = 0.0
    num = 0.0
    row_sums = np.zeros(n)

    for i in range(n):
        nb = [j for j in neighbors[i] if j != i]
        row_sums[i] = len(nb)
        s0 += len(nb)
        for j in nb:
            num += z[i] * z[j]

    if s0 == 0:
        return {"morans_i": 0.0, "p_value": 1.0, "z_score": 0.0, "interpretation": "No connected neighbors"}

    morans_i = (n / s0) * (num / ss_total)
    e_i = -1.0 / (n - 1.0)

    # Variance under randomization assumption
    s1 = 2.0 * s0  # for symmetric binary weights
    s2 = np.sum((2 * row_sums)**2)
    k = (np.sum(z**4) / n) / ((ss_total / n)**2)

    var_num = (n * ((n**2 - 3*n + 3) * s1 - n * s2 + 3 * (s0**2))) - (k * ((n**2 - n) * s1 - 2 * n * s2 + 6 * (s0**2)))
    var_denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * (s0**2)
    var_i = (var_num / var_denom) - (e_i**2) if var_denom > 0 else 0.0

    z_score = (morans_i - e_i) / np.sqrt(max(var_i, 1e-9))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    interpretation = (
        "Statistically significant spatial clustering (Hotspots clustered)"
        if (p_value < 0.05 and morans_i > 0) else
        "Statistically significant spatial dispersion" if (p_value < 0.05 and morans_i < 0) else
        "Random spatial distribution (No significant spatial autocorrelation)"
    )

    result = {
        "morans_i": round(float(morans_i), 4),
        "expected_i": round(float(e_i), 4),
        "z_score": round(float(z_score), 3),
        "p_value": round(float(p_value), 5),
        "interpretation": interpretation
    }
    logger.info(f"Global Moran's I: {result['morans_i']} | z: {result['z_score']} | p: {result['p_value']} ({interpretation})")
    return result

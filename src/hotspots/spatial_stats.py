"""Spatial statistics: Getis-Ord Gi* hotspot detection and DBSCAN point clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
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

    gi_z = np.zeros(n)
    for i in range(n):
        dists = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
        wij = (dists <= distance_threshold_m).astype(float)
        w_sum = np.sum(wij)
        w_sq_sum = np.sum(wij**2)

        numerator = np.sum(wij * x) - (x_mean * w_sum)
        denom = s * np.sqrt((n * w_sq_sum - (w_sum**2)) / (n - 1.0))
        gi_z[i] = numerator / denom if denom > 0 else 0.0

    df["gi_z_score"] = np.round(gi_z, 3)
    # Significant at 95% confidence if z > 1.96
    df["is_statistically_significant_hotspot"] = (df["gi_z_score"] >= 1.96).astype(int)
    return df

def cluster_crash_points_dbscan(crashes_df: pd.DataFrame,
                                eps_meters: float = 150.0,
                                min_samples: int = 5) -> pd.DataFrame:
    """Clusters crash points using spatial DBSCAN."""
    df = crashes_df.copy()
    coords_m = np.column_stack([
        df["latitude"].values * 111000.0,
        df["longitude"].values * 102500.0
    ])

    clustering = DBSCAN(eps=eps_meters, min_samples=min_samples).fit(coords_m)
    df["spatial_cluster_id"] = clustering.labels_
    return df

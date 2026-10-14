"""OSM road network loader and segment extractor for Kolkata."""
import os
import json
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("osm_loader")

def get_kolkata_road_segments(segment_length_m: int = 300,
                              output_path: str = "data/interim/kolkata_segments.csv") -> pd.DataFrame:
    """Retrieves or builds segmented road graph units (200-500m) for Kolkata."""
    if os.path.exists(output_path):
        logger.info(f"Loading existing road segments from {output_path}")
        return pd.read_csv(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Extracting Kolkata road segments and engineering corridor geometry...")

    from src.data.mock_data_generator import KOLKATA_CORRIDORS

    segments = []
    seg_idx = 1000

    for corridor in KOLKATA_CORRIDORS:
        coords = corridor["coords"]
        for p_idx in range(len(coords) - 1):
            p1 = coords[p_idx]
            p2 = coords[p_idx + 1]

            # Approximate distance in meters between waypoints
            d_lat = (p2[0] - p1[0]) * 111000.0
            d_lon = (p2[1] - p1[1]) * 102000.0
            dist = np.sqrt(d_lat**2 + d_lon**2)
            n_sub_segments = max(2, int(dist / segment_length_m))

            for s in range(n_sub_segments):
                seg_idx += 1
                t0 = s / n_sub_segments
                t1 = (s + 1) / n_sub_segments

                sub_p1_lat = p1[0] + t0 * (p2[0] - p1[0])
                sub_p1_lon = p1[1] + t0 * (p2[1] - p1[1])
                sub_p2_lat = p1[0] + t1 * (p2[0] - p1[0])
                sub_p2_lon = p1[1] + t1 * (p2[1] - p1[1])

                mid_lat = (sub_p1_lat + sub_p2_lat) / 2.0
                mid_lon = (sub_p1_lon + sub_p2_lon) / 2.0

                is_junction = 1 if (s == 0 or s == n_sub_segments - 1 or np.random.random() < 0.15) else 0
                has_median = 1 if corridor["lanes"] >= 4 else 0
                is_lit = 1 if corridor["highway_type"] in ["primary", "trunk"] and np.random.random() > 0.18 else 0

                segments.append({
                    "segment_id": f"SEG-{corridor['zone_id']}-{seg_idx}",
                    "corridor_name": corridor["corridor_name"],
                    "spatial_zone_id": corridor["zone_id"],
                    "highway_type": corridor["highway_type"],
                    "speed_limit": corridor["speed_limit"],
                    "lanes": corridor["lanes"],
                    "has_median": has_median,
                    "is_lit": is_lit,
                    "is_junction": is_junction,
                    "length_meters": round(dist / n_sub_segments, 1),
                    "start_lat": round(sub_p1_lat, 6),
                    "start_lon": round(sub_p1_lon, 6),
                    "end_lat": round(sub_p2_lat, 6),
                    "end_lon": round(sub_p2_lon, 6),
                    "mid_lat": round(mid_lat, 6),
                    "mid_lon": round(mid_lon, 6),
                    "junction_density_score": np.random.uniform(0.1, 0.9) if is_junction else np.random.uniform(0.01, 0.25)
                })

    df_segments = pd.DataFrame(segments)
    df_segments.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df_segments)} road segments saved to {output_path}")
    return df_segments

if __name__ == "__main__":
    get_kolkata_road_segments()

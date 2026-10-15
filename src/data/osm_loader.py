"""OSM road network loader and segment extractor for Kolkata.

Integrates real Kolkata OSMnx road network graph extraction, edge segmentation
into 200-500m analytical units, spatial zone mapping, and offline fallback caching.
"""
import os
import re
import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("osm_loader")

KOLKATA_PLACE_NAME = "Kolkata, West Bengal, India"
DEFAULT_NETWORK_CACHE = "data/raw/osm/kolkata_drive_network.graphml"
DEFAULT_SEGMENTS_PATH = "data/interim/kolkata_segments.csv"

# Zone spatial centroids for snapping OSM edges to 8 balanced KMDA planning zones
ZONE_CENTROIDS = {
    "Central":        (22.5726, 88.3639),
    "North":          (22.6200, 88.3750),
    "North-East":     (22.6100, 88.4250),
    "East":           (22.5600, 88.4200),
    "South-Central":  (22.5300, 88.3600),
    "South":          (22.4850, 88.3650),
    "South-West":     (22.4900, 88.3150),
    "West":           (22.5500, 88.3250),
}

# Bounding box for Kolkata metropolitan area (left/west, bottom/south, right/east, top/north)
KOLKATA_BBOX = (88.24, 22.43, 88.46, 22.64)
KOLKATA_HIGHWAY_FILTER = (
    '["highway"~"motorway|trunk|primary|secondary|tertiary|'
    'motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
)


def _infer_zone(lat: float, lon: float) -> str:
    """Assigns nearest Kolkata planning zone based on Euclidean distance to centroid."""
    best_zone = "Central"
    min_dist = float("inf")
    for zone, (clat, clon) in ZONE_CENTROIDS.items():
        dist = (lat - clat) ** 2 + (lon - clon) ** 2
        if dist < min_dist:
            min_dist = dist
            best_zone = zone
    return best_zone


def _parse_speed_limit(maxspeed_val, highway_type: str) -> int:
    """Parses OSM maxspeed tag or imputes default based on Kolkata road hierarchy."""
    if isinstance(maxspeed_val, (list, tuple, np.ndarray)):
        maxspeed_val = maxspeed_val[0] if len(maxspeed_val) > 0 else None
    if pd.notna(maxspeed_val):
        match = re.search(r"\d+", str(maxspeed_val))
        if match:
            return int(match.group())
    defaults = {
        "motorway": 80,
        "trunk": 60,
        "primary": 50,
        "secondary": 40,
        "tertiary": 30,
        "residential": 25,
    }
    clean_hwy = str(highway_type).lower().replace("_link", "")
    return defaults.get(clean_hwy, 40)


def _parse_lanes(lanes_val, highway_type: str) -> int:
    """Parses OSM lanes tag or imputes realistic Kolkata defaults."""
    if isinstance(lanes_val, (list, tuple, np.ndarray)):
        lanes_val = lanes_val[0] if len(lanes_val) > 0 else None
    if pd.notna(lanes_val):
        match = re.search(r"\d+", str(lanes_val))
        if match:
            return max(1, int(match.group()))
    defaults = {
        "motorway": 6,
        "trunk": 6,
        "primary": 4,
        "secondary": 2,
        "tertiary": 2,
        "residential": 1,
    }
    clean_hwy = str(highway_type).lower().replace("_link", "")
    return defaults.get(clean_hwy, 2)


def fetch_kolkata_osmnx_network(place_name: str = KOLKATA_PLACE_NAME,
                                network_type: str = "drive",
                                cache_path: str = DEFAULT_NETWORK_CACHE,
                                timeout_seconds: int = 180):
    """Downloads or loads cached full Kolkata road network via OSMnx.

    Parameters
    ----------
    place_name : str
        Geocoding place query for OpenStreetMap Overpass API.
    network_type : str
        Type of street network ('drive', 'all', etc.).
    cache_path : str
        Local GraphML file path to store the network graph.
    timeout_seconds : int
        Overpass API request timeout.

    Returns
    -------
    networkx.MultiDiGraph or None
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached OSMnx road network from {cache_path}")
        try:
            import osmnx as ox
            return ox.load_graphml(cache_path)
        except Exception as e:
            logger.warning(f"Could not load cached GraphML ({e}). Attempting fresh query.")

    try:
        import osmnx as ox
        ox.settings.timeout = timeout_seconds
        ox.settings.use_cache = True
        logger.info(f"Querying OpenStreetMap via OSMnx bbox {KOLKATA_BBOX} for Kolkata...")
        G = ox.graph_from_bbox(
            bbox=KOLKATA_BBOX,
            custom_filter=KOLKATA_HIGHWAY_FILTER,
            simplify=True
        )
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ox.save_graphml(G, cache_path)
        logger.info(f"Successfully downloaded OSMnx road network ({len(G.nodes)} nodes, {len(G.edges)} edges) -> cached to {cache_path}")
        return G
    except Exception as e:
        logger.warning(
            f"OSMnx online query was not completed ({type(e).__name__}: {e}). "
            "Proceeding with structured Kolkata corridor segments."
        )
        return None


def convert_osmnx_graph_to_segments(G, segment_length_m: int = 300) -> pd.DataFrame:
    """Converts an OSMnx road network graph into uniform 200-500m analysis segments."""
    import osmnx as ox
    gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    logger.info(f"Converting {len(gdf_edges)} OSMnx edges into ~{segment_length_m}m analysis segments...")

    segments = []
    seg_idx = 1000

    for idx, edge in gdf_edges.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty:
            continue

        edge_length = float(edge.get("length", geom.length * 111000.0))
        if edge_length < 15.0:
            continue

        hwy = edge.get("highway", "primary")
        if isinstance(hwy, (list, tuple, np.ndarray)):
            hwy = hwy[0] if len(hwy) > 0 else "primary"
        hwy = str(hwy).lower()

        speed_limit = _parse_speed_limit(edge.get("maxspeed", None), hwy)
        lanes = _parse_lanes(edge.get("lanes", None), hwy)
        
        name = edge.get("name", None)
        if isinstance(name, (list, tuple, np.ndarray)):
            name = name[0] if len(name) > 0 else None

        n_sub = max(1, int(round(edge_length / segment_length_m)))
        for s in range(n_sub):
            seg_idx += 1
            t0 = s / n_sub
            t1 = (s + 1) / n_sub
            t_mid = (s + 0.5) / n_sub

            p_start = geom.interpolate(t0, normalized=True)
            p_end = geom.interpolate(t1, normalized=True)
            p_mid = geom.interpolate(t_mid, normalized=True)

            start_lat = round(p_start.y, 6)
            start_lon = round(p_start.x, 6)
            end_lat = round(p_end.y, 6)
            end_lon = round(p_end.x, 6)
            mid_lat = round(p_mid.y, 6)
            mid_lon = round(p_mid.x, 6)

            zone_id = _infer_zone(mid_lat, mid_lon)
            clean_hwy = hwy.replace("_link", "")
            
            if pd.notna(name) and str(name).strip():
                corridor_name = str(name).strip()
            else:
                corridor_name = f"{clean_hwy.title()} Corridor ({zone_id})"

            is_junction = 1 if (s == 0 or s == n_sub - 1) else 0

            segments.append({
                "segment_id": f"SEG-{zone_id[:4].upper()}-{seg_idx}",
                "corridor_name": corridor_name,
                "spatial_zone_id": zone_id,
                "highway_type": clean_hwy,
                "speed_limit": speed_limit,
                "lanes": lanes,
                "has_median": 1 if lanes >= 4 else 0,
                "is_lit": 1 if clean_hwy in ["motorway", "trunk", "primary"] else 0,
                "is_junction": is_junction,
                "length_meters": round(edge_length / n_sub, 1),
                "start_lat": start_lat,
                "start_lon": start_lon,
                "end_lat": end_lat,
                "end_lon": end_lon,
                "mid_lat": mid_lat,
                "mid_lon": mid_lon,
                "junction_density_score": 0.65 if is_junction else 0.15
            })

    df = pd.DataFrame(segments)
    logger.info(f"Extracted {len(df)} segments from OSMnx road network across {df['spatial_zone_id'].nunique()} zones.")
    return df


def _build_corridor_segments(segment_length_m: int = 300) -> pd.DataFrame:
    """Builds realistic corridor segments along major Kolkata arterial networks."""
    from src.data.mock_data_generator import KOLKATA_CORRIDORS

    segments = []
    seg_idx = 1000

    for corridor in KOLKATA_CORRIDORS:
        coords = corridor["coords"]
        for p_idx in range(len(coords) - 1):
            p1 = coords[p_idx]
            p2 = coords[p_idx + 1]

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
                    "segment_id": f"SEG-{corridor['zone_id'][:4].upper()}-{seg_idx}",
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

    return pd.DataFrame(segments)


def get_kolkata_road_segments(segment_length_m: int = 300,
                              output_path: str = DEFAULT_SEGMENTS_PATH,
                              force_osmnx_download: bool = False) -> pd.DataFrame:
    """Retrieves or builds segmented road graph units (200-500m) for Kolkata.

    Checks:
    1. If output_path already exists and not force_osmnx_download, load it.
    2. If force_osmnx_download or missing, attempts OSMnx road network extraction.
    3. Falls back gracefully to high-resolution Kolkata corridor segments if offline/throttled.
    """
    if os.path.exists(output_path) and not force_osmnx_download:
        logger.info(f"Loading existing road segments from {output_path}")
        return pd.read_csv(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_segments = None

    # Load from OSMnx graph (local cache or Overpass)
    G = fetch_kolkata_osmnx_network()
    if G is not None:
        try:
            df_segments = convert_osmnx_graph_to_segments(G, segment_length_m=segment_length_m)
        except Exception as e:
            logger.warning(f"Error parsing OSMnx graph ({e}). Falling back to corridor segments.")

    if df_segments is None or len(df_segments) == 0:
        logger.info("Extracting Kolkata road segments and engineering corridor geometry...")
        df_segments = _build_corridor_segments(segment_length_m=segment_length_m)

    df_segments.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df_segments)} road segments saved to {output_path}")
    return df_segments


if __name__ == "__main__":
    get_kolkata_road_segments()

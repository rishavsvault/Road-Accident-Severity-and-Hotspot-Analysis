"""Geospatial helper utilities for coordinate transforms and distance calculations."""
import math
import numpy as np
import pandas as pd

KOLKATA_UTM_CRS = "EPSG:32645"  # WGS 84 / UTM Zone 45N (metric, Kolkata)
WGS84_CRS = "EPSG:4326"

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def is_within_kolkata_bounds(lat: float, lon: float,
                            north: float = 22.6500, south: float = 22.4500,
                            east: float = 88.4500, west: float = 88.2500) -> bool:
    """Checks whether a given latitude/longitude coordinate falls within Kolkata boundary."""
    return (south <= lat <= north) and (west <= lon <= east)

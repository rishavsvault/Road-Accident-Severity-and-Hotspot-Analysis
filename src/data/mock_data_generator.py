"""Generates realistic crash event records across Kolkata's road network.

Reflects real Kolkata road geometries, typical MoRTH severity distributions
(approx 60% Minor, 28% Grievous, 12% Fatal), monsoon peaks, and time-of-day dynamics.
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger("mock_data_generator")

# Key Kolkata Corridors & Typical Waypoints
KOLKATA_CORRIDORS = [
    {
        "corridor_name": "EM Bypass",
        "zone_id": "Zone_East",
        "ward_range": (56, 108),
        "highway_type": "primary",
        "speed_limit": 60,
        "lanes": 6,
        "coords": [(22.5800, 88.4050), (22.5400, 88.3980), (22.4950, 88.3900)]
    },
    {
        "corridor_name": "VIP Road",
        "zone_id": "Zone_NorthEast",
        "ward_range": (20, 35),
        "highway_type": "primary",
        "speed_limit": 60,
        "lanes": 6,
        "coords": [(22.5950, 88.4100), (22.6200, 88.4280), (22.6450, 88.4400)]
    },
    {
        "corridor_name": "AJC Bose Road & Flyover",
        "zone_id": "Zone_Central",
        "ward_range": (60, 75),
        "highway_type": "primary",
        "speed_limit": 50,
        "lanes": 4,
        "coords": [(22.5430, 88.3550), (22.5410, 88.3680), (22.5390, 88.3750)]
    },
    {
        "corridor_name": "Strand Road",
        "zone_id": "Zone_West",
        "ward_range": (40, 55),
        "highway_type": "secondary",
        "speed_limit": 40,
        "lanes": 4,
        "coords": [(22.5800, 88.3450), (22.5650, 88.3420), (22.5500, 88.3400)]
    },
    {
        "corridor_name": "Diamond Harbour Road",
        "zone_id": "Zone_SouthWest",
        "ward_range": (115, 132),
        "highway_type": "primary",
        "speed_limit": 50,
        "lanes": 4,
        "coords": [(22.5150, 88.3180), (22.4800, 88.3100), (22.4550, 88.3050)]
    },
    {
        "corridor_name": "Park Street / Camac Street",
        "zone_id": "Zone_Central",
        "ward_range": (61, 65),
        "highway_type": "tertiary",
        "speed_limit": 30,
        "lanes": 2,
        "coords": [(22.5530, 88.3520), (22.5510, 88.3590), (22.5490, 88.3660)]
    },
    {
        "corridor_name": "Belghoria Expressway",
        "zone_id": "Zone_North",
        "ward_range": (1, 15),
        "highway_type": "trunk",
        "speed_limit": 70,
        "lanes": 6,
        "coords": [(22.6400, 88.3700), (22.6500, 88.4000), (22.6550, 88.4200)]
    },
    {
        "corridor_name": "Raja SC Mullick Road",
        "zone_id": "Zone_South",
        "ward_range": (90, 105),
        "highway_type": "secondary",
        "speed_limit": 40,
        "lanes": 2,
        "coords": [(22.5050, 88.3680), (22.4850, 88.3710), (22.4650, 88.3730)]
    }
]

def generate_kolkata_crashes(n_samples: int = 3500,
                             start_date: str = "2023-01-01",
                             end_date: str = "2024-12-31",
                             seed: int = 42,
                             output_path: str = "data/raw/crashes/crashes.csv") -> pd.DataFrame:
    """Generates synthetic crash records distributed across the real Kolkata road network."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Generating {n_samples} synthetic crash events across Kolkata road network...")

    # Load real road segments if available
    segments_path = "data/interim/kolkata_segments.csv"
    if not os.path.exists(segments_path):
        from src.data.osm_loader import get_kolkata_road_segments
        df_segments = get_kolkata_road_segments()
    else:
        df_segments = pd.read_csv(segments_path)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_seconds = int((end_dt - start_dt).total_seconds())

    records = []
    vehicle_types = ["Two-Wheeler", "Four-Wheeler / Car", "Bus", "Auto-Rickshaw / Toto", "Truck / Heavy", "Pedestrian-Direct"]

    if df_segments is not None and len(df_segments) > 0:
        # Filter to segments within Kolkata administrative polygon
        try:
            from src.data.crash_cleaner import _point_in_polygon
            valid_mask = df_segments.apply(lambda r: _point_in_polygon(r["mid_lat"], r["mid_lon"]), axis=1)
            if valid_mask.sum() > 100:
                df_segments = df_segments[valid_mask].reset_index(drop=True)
        except Exception:
            pass

        # Weight segments by road hierarchy traffic volume
        hwy_weights = {
            "motorway": 4.5,
            "trunk": 3.8,
            "primary": 2.5,
            "secondary": 1.4,
            "tertiary": 1.0,
            "residential": 0.6
        }
        weights = df_segments["highway_type"].map(hwy_weights).fillna(1.0).values
        weights = weights / weights.sum()

        chosen_indices = np.random.choice(len(df_segments), size=n_samples, p=weights)
        sampled_segments = df_segments.iloc[chosen_indices].reset_index(drop=True)

        for i in range(n_samples):
            crash_id = f"KOL-CRASH-{2023 + (i % 2)}-{i+1:05d}"
            random_sec = random.randint(0, total_seconds)
            dt = start_dt + timedelta(seconds=random_sec)
            hour = dt.hour
            month = dt.month

            seg = sampled_segments.iloc[i]
            # Interpolate along segment line with realistic GPS noise (~12-15m)
            alpha = random.random()
            s_lat, s_lon = float(seg.get("start_lat", seg["mid_lat"])), float(seg.get("start_lon", seg["mid_lon"]))
            e_lat, e_lon = float(seg.get("end_lat", seg["mid_lat"])), float(seg.get("end_lon", seg["mid_lon"]))
            base_lat = s_lat + alpha * (e_lat - s_lat)
            base_lon = s_lon + alpha * (e_lon - s_lon)
            lat = base_lat + np.random.normal(0, 0.00012)
            lon = base_lon + np.random.normal(0, 0.00012)

            ward = random.randint(1, 144)
            primary_vehicle = random.choices(
                vehicle_types,
                weights=[0.38, 0.22, 0.14, 0.12, 0.08, 0.06],
                k=1
            )[0]
            pedestrian_involved = 1 if (primary_vehicle == "Pedestrian-Direct" or random.random() < 0.25) else 0

            is_night = 1 if (hour >= 18 or hour < 6) else 0
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0

            speed_val = int(seg.get("speed_limit", 40))
            is_lit = int(seg.get("is_lit", 1 if (random.random() < 0.45) else 0))
            has_median = int(seg.get("has_median", 1 if (random.random() < 0.35) else 0))
            is_junction = int(seg.get("is_junction", 1 if (random.random() < 0.75) else 0))

            # Physics-grounded crash severity assignment (WHO & IRC-SP:88 empirical risk curves)
            score = -2.1
            score += ((speed_val / 40.0) ** 2.2) * 1.6 - 1.6

            is_vru = (primary_vehicle in ["Two-Wheeler", "Pedestrian-Direct"]) or (pedestrian_involved == 1)
            is_heavy = primary_vehicle in ["Truck / Heavy", "Bus"]

            if is_vru and is_heavy:
                score += 2.5
            elif is_vru:
                score += 1.4
            elif is_heavy:
                score += 0.7

            if is_night and not is_lit:
                score += 1.3
            elif is_night:
                score += 0.4

            if not has_median and speed_val >= 50:
                score += 0.85
            if is_junction and speed_val >= 40:
                score += 0.55
            if is_monsoon:
                score += 0.45

            # Unobserved real-world variance (helmet use, impact angle, braking reaction)
            score += np.random.normal(0, 0.40)

            cut1 = 0.0
            cut2 = 2.1
            p_grievous_plus = 1.0 / (1.0 + np.exp(-(score - cut1)))
            p_fatal = 1.0 / (1.0 + np.exp(-(score - cut2)))
            p_minor = 1.0 - p_grievous_plus
            p_grievous = p_grievous_plus - p_fatal

            probs = np.clip([p_minor, p_grievous, p_fatal], 0, 1)
            probs /= probs.sum()

            severity = int(np.random.choice([0, 1, 2], p=probs))

            records.append({
                "crash_id": crash_id,
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "reported_corridor": str(seg.get("corridor_name", "Kolkata Arterial")),
                "spatial_zone_id": str(seg.get("spatial_zone_id", "Central")),
                "ward_number": ward,
                "severity_code": severity,
                "severity_label": ["Minor / Non-Injury", "Grievous Injury", "Fatal"][severity],
                "primary_vehicle": primary_vehicle,
                "pedestrian_involved": pedestrian_involved,
                "reported_speed_limit": speed_val,
                "weather_condition": "Rainy" if is_monsoon and random.random() < 0.55 else ("Foggy" if month in [12, 1] and hour <= 7 else "Clear")
            })
    else:
        # Fallback to corridor sampling if segments cannot be loaded
        for i in range(n_samples):
            crash_id = f"KOL-CRASH-{2023 + (i % 2)}-{i+1:05d}"
            random_sec = random.randint(0, total_seconds)
            dt = start_dt + timedelta(seconds=random_sec)
            hour = dt.hour
            month = dt.month

            corridor = random.choices(
                KOLKATA_CORRIDORS,
                weights=[0.24, 0.20, 0.16, 0.12, 0.10, 0.06, 0.07, 0.05],
                k=1
            )[0]

            p1, p2 = random.choice(list(zip(corridor["coords"][:-1], corridor["coords"][1:])))
            alpha = random.random()
            base_lat = p1[0] + alpha * (p2[0] - p1[0])
            base_lon = p1[1] + alpha * (p2[1] - p1[1])
            lat = base_lat + np.random.normal(0, 0.0008)
            lon = base_lon + np.random.normal(0, 0.0008)

            ward = random.randint(corridor["ward_range"][0], corridor["ward_range"][1])
            primary_vehicle = random.choices(
                vehicle_types,
                weights=[0.38, 0.22, 0.14, 0.12, 0.08, 0.06],
                k=1
            )[0]
            pedestrian_involved = 1 if (primary_vehicle == "Pedestrian-Direct" or random.random() < 0.25) else 0

            is_night = 1 if (hour >= 18 or hour < 6) else 0
            is_monsoon = 1 if month in [6, 7, 8, 9] else 0

            speed_val = corridor["speed_limit"]
            is_lit = 1 if (random.random() < 0.45) else 0
            has_median = 1 if (random.random() < 0.35) else 0
            is_junction = 1 if (random.random() < 0.70) else 0

            score = -2.1
            score += ((speed_val / 40.0) ** 2.2) * 1.6 - 1.6
            is_vru = (primary_vehicle in ["Two-Wheeler", "Pedestrian-Direct"]) or (pedestrian_involved == 1)
            is_heavy = primary_vehicle in ["Truck / Heavy", "Bus"]
            if is_vru and is_heavy:
                score += 2.5
            elif is_vru:
                score += 1.4
            elif is_heavy:
                score += 0.7
            if is_night and not is_lit:
                score += 1.3
            elif is_night:
                score += 0.4
            if not has_median and speed_val >= 50:
                score += 0.85
            if is_junction and speed_val >= 40:
                score += 0.55
            if is_monsoon:
                score += 0.45
            score += np.random.normal(0, 0.40)

            cut1 = 0.0
            cut2 = 2.1
            p_grievous_plus = 1.0 / (1.0 + np.exp(-(score - cut1)))
            p_fatal = 1.0 / (1.0 + np.exp(-(score - cut2)))
            p_minor = 1.0 - p_grievous_plus
            p_grievous = p_grievous_plus - p_fatal
            probs = np.clip([p_minor, p_grievous, p_fatal], 0, 1)
            probs /= probs.sum()
            severity = int(np.random.choice([0, 1, 2], p=probs))

            records.append({
                "crash_id": crash_id,
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "reported_corridor": corridor["corridor_name"],
                "spatial_zone_id": corridor["zone_id"],
                "ward_number": ward,
                "severity_code": severity,
                "severity_label": ["Minor / Non-Injury", "Grievous Injury", "Fatal"][severity],
                "primary_vehicle": primary_vehicle,
                "pedestrian_involved": pedestrian_involved,
                "reported_speed_limit": corridor["speed_limit"],
                "weather_condition": "Rainy" if is_monsoon and random.random() < 0.55 else ("Foggy" if month in [12, 1] and hour <= 7 else "Clear")
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully generated {len(df)} records -> {output_path}")
    logger.info(f"Class distribution: {df['severity_label'].value_counts().to_dict()}")
    return df

if __name__ == "__main__":
    generate_kolkata_crashes()

"""Feature naming utilities for converting between model-internal prefixes (feat_*) and simple, human-friendly names."""
from typing import Dict

# Mapping from raw model feature column names to clean, presentation-ready names
FEATURE_DISPLAY_MAP: Dict[str, str] = {
    "feat_highway_hierarchy": "Highway Hierarchy",
    "feat_speed_limit": "Speed Limit (km/h)",
    "feat_is_high_speed": "High-Speed Corridor",
    "feat_lanes": "Number of Lanes",
    "feat_has_median": "Median Barrier",
    "feat_is_lit": "Street Lighting",
    "feat_is_junction": "Junction Intersection",
    "feat_junction_density": "Junction Density",
    "feat_hour": "Hour of Day",
    "feat_dayofweek": "Day of Week",
    "feat_is_weekend": "Weekend",
    "feat_hour_sin": "Hour of Day (Sin)",
    "feat_hour_cos": "Hour of Day (Cos)",
    "feat_is_night": "Nighttime",
    "feat_is_monsoon": "Monsoon Season",
    "feat_is_festive_season": "Festive Season",
    "feat_precipitation": "Precipitation (mm/h)",
    "feat_is_raining": "Active Rainfall",
    "feat_visibility_km": "Visibility (km)",
    "feat_is_low_visibility": "Low Visibility",
    "feat_temperature": "Temperature (°C)",
    "feat_is_two_wheeler": "Two-Wheeler Involved",
    "feat_pedestrian_involved": "Pedestrian Involved",
}

# Mapping from raw model feature column names to simple identifier names
FEATURE_SIMPLE_MAP: Dict[str, str] = {
    "feat_highway_hierarchy": "highway_hierarchy",
    "feat_speed_limit": "speed_limit",
    "feat_is_high_speed": "is_high_speed",
    "feat_lanes": "lanes",
    "feat_has_median": "has_median",
    "feat_is_lit": "is_lit",
    "feat_is_junction": "is_junction",
    "feat_junction_density": "junction_density",
    "feat_hour": "hour",
    "feat_dayofweek": "dayofweek",
    "feat_is_weekend": "is_weekend",
    "feat_hour_sin": "hour_sin",
    "feat_hour_cos": "hour_cos",
    "feat_is_night": "is_night",
    "feat_is_monsoon": "is_monsoon",
    "feat_is_festive_season": "is_festive_season",
    "feat_precipitation": "precipitation",
    "feat_is_raining": "is_raining",
    "feat_visibility_km": "visibility_km",
    "feat_is_low_visibility": "is_low_visibility",
    "feat_temperature": "temperature",
    "feat_is_two_wheeler": "is_two_wheeler",
    "feat_pedestrian_involved": "pedestrian_involved",
}

# Reverse mapping from simple names to feat_* model column names
REVERSE_SIMPLE_MAP: Dict[str, str] = {v: k for k, v in FEATURE_SIMPLE_MAP.items()}


def to_display_name(feature_name: str) -> str:
    """Returns human-friendly display name for a feature."""
    if feature_name in FEATURE_DISPLAY_MAP:
        return FEATURE_DISPLAY_MAP[feature_name]
    # Fallback: remove feat_ prefix and title case
    clean = feature_name.replace("feat_", "").replace("_", " ").title()
    return clean


def to_simple_name(feature_name: str) -> str:
    """Returns simplified identifier name for a feature (strips feat_ prefix)."""
    if feature_name in FEATURE_SIMPLE_MAP:
        return FEATURE_SIMPLE_MAP[feature_name]
    return feature_name.replace("feat_", "")


def to_model_feature_name(simple_name: str) -> str:
    """Returns internal model feature name (e.g. speed_limit -> feat_speed_limit)."""
    if simple_name in REVERSE_SIMPLE_MAP:
        return REVERSE_SIMPLE_MAP[simple_name]
    if simple_name.startswith("feat_"):
        return simple_name
    return f"feat_{simple_name}"


# Actionable Physical and Civil Engineering Drivers (causally relevant for road safety audits)
ACTIONABLE_INFRASTRUCTURE_FEATURES = {
    "feat_speed_limit", "speed_limit",
    "feat_is_high_speed", "is_high_speed",
    "feat_lanes", "lanes",
    "feat_has_median", "has_median",
    "feat_is_lit", "is_lit",
    "feat_is_junction", "is_junction",
    "feat_junction_density", "junction_density",
    "feat_highway_hierarchy", "highway_hierarchy",
    "feat_is_two_wheeler", "is_two_wheeler", "two_wheeler",
    "feat_pedestrian_involved", "pedestrian_involved",
    "feat_is_low_visibility", "is_low_visibility", "visibility_km", "feat_visibility_km",
    "feat_is_raining", "is_raining", "precipitation", "feat_precipitation",
    "feat_is_night", "is_night"
}

# Non-Causal Contextual / Observational Covariates (nuisance proxies, e.g. ambient temp or hour trigonometric transforms)
NON_CAUSAL_CONTEXT_FEATURES = {
    "feat_temperature", "temperature",
    "feat_hour_cos", "hour_cos",
    "feat_hour_sin", "hour_sin",
    "feat_hour", "hour",
    "feat_dayofweek", "dayofweek",
    "feat_is_weekend", "is_weekend",
    "feat_is_monsoon", "is_monsoon",
    "feat_is_festive_season", "is_festive_season"
}


def is_causal_actionable(feature_name: str) -> bool:
    """Determines whether a feature represents a physical/civil engineering driver rather than a non-causal observational proxy."""
    clean = feature_name.strip()
    return clean in ACTIONABLE_INFRASTRUCTURE_FEATURES or clean.replace("feat_", "") in ACTIONABLE_INFRASTRUCTURE_FEATURES


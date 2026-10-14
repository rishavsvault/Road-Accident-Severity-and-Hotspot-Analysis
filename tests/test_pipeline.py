"""Unit tests for geospatial and feature transformations."""
import pytest
import pandas as pd
import numpy as np
from src.utils.geo_utils import haversine_distance, is_within_kolkata_bounds
from src.features.road_features import encode_road_features
from src.features.temporal_features import extract_temporal_features
from src.explainability.intervention_engine import KMDAInterventionEngine

def test_haversine_distance():
    # Distance between two nearby Kolkata coordinates
    d = haversine_distance(22.5726, 88.3639, 22.5800, 88.3639)
    assert 800 < d < 850

def test_kolkata_bounding_box():
    assert is_within_kolkata_bounds(22.5726, 88.3639) is True
    # Coordinate outside Kolkata (e.g. Mumbai)
    assert is_within_kolkata_bounds(19.0760, 72.8777) is False

def test_road_feature_encoding():
    df = pd.DataFrame([{
        "segment_highway_type": "primary",
        "segment_speed_limit": 60,
        "segment_lanes": 4,
        "segment_has_median": 1,
        "segment_is_lit": 1,
        "segment_is_junction": 0,
        "segment_junction_density": 0.2
    }])
    encoded = encode_road_features(df)
    assert encoded["feat_highway_hierarchy"].iloc[0] == 3
    assert encoded["feat_is_high_speed"].iloc[0] == 1

def test_temporal_features():
    df = pd.DataFrame([{"timestamp": "2024-07-15 23:30:00"}])
    extracted = extract_temporal_features(df)
    assert extracted["feat_hour"].iloc[0] == 23
    assert extracted["feat_is_night"].iloc[0] == 1
    assert extracted["feat_is_monsoon"].iloc[0] == 1

def test_intervention_engine():
    engine = KMDAInterventionEngine()
    seg = {
        "is_lit": 0,
        "speed_limit": 60,
        "is_junction": 1,
        "night_crashes": 3,
        "pedestrian_crashes": 2
    }
    actions = engine.recommend_for_segment(seg)
    assert len(actions) >= 2
    ids = [a["intervention_id"] for a in actions]
    assert "INT_LIGHTING" in ids
    assert "INT_SPEED_CALMING" in ids

"""Unit tests for geospatial, feature transformations, models, and calibration."""
import os
import tempfile
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


def test_causal_countermeasure_impact():
    engine = KMDAInterventionEngine()
    b_dict = {"is_lit": 0, "speed_limit": 60, "has_median": 0, "is_night": 1}
    s_dict = {"is_lit": 1, "speed_limit": 40, "has_median": 1, "is_night": 1}
    p_base = np.array([0.20, 0.30, 0.50])
    p_sim = np.array([0.40, 0.40, 0.20])

    res = engine.estimate_causal_countermeasure_impact(b_dict, s_dict, p_base, p_sim)
    assert "cmf_compound_multiplier" in res
    assert res["cmf_compound_multiplier"] < 1.0  # Compound reduction
    assert res["causal_fatal_delta_pct"] < 0
    assert len(res["cmf_citations"]) >= 3


def test_spatial_grouped_cv():
    from src.models.spatial_cv import SpatialGroupedCV
    df = pd.DataFrame({
        "spatial_zone_id": ["Zone_A", "Zone_A", "Zone_B", "Zone_B", "Zone_C", "Zone_C"],
        "feature": [1, 2, 3, 4, 5, 6]
    })
    cv = SpatialGroupedCV(n_splits=3, group_col="spatial_zone_id")
    splits = list(cv.split(df))
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_zones = set(df.iloc[train_idx]["spatial_zone_id"])
        val_zones = set(df.iloc[val_idx]["spatial_zone_id"])
        assert len(train_zones.intersection(val_zones)) == 0


def test_ordinal_severity_model():
    from src.models.xgboost_trainer import OrdinalSeverityModel
    X = np.random.RandomState(42).randn(30, 4)
    y = np.random.RandomState(42).choice([0, 1, 2], size=30)
    feat_names = ["f1", "f2", "f3", "f4"]

    model = OrdinalSeverityModel(random_state=42)
    model.fit(X, y, feature_names=feat_names)

    probs = model.predict_proba(X)
    assert probs.shape == (30, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    preds = model.predict(X)
    assert len(preds) == 30
    assert set(preds).issubset({0, 1, 2})


def test_segment_risk_score_normalization():
    from src.hotspots.segment_ranker import rank_and_tier_segments
    segments = pd.DataFrame([
        {"segment_id": "SEG_001", "corridor_name": "EM Bypass", "length_meters": 1200},
        {"segment_id": "SEG_002", "corridor_name": "AJC Bose", "length_meters": 900},
        {"segment_id": "SEG_003", "corridor_name": "VIP Road", "length_meters": 1500},
    ])
    crashes = pd.DataFrame([
        # SEG_001 has 3 fatal crashes
        {"crash_id": "C1", "segment_id": "SEG_001", "severity_code": 2, "feat_pedestrian_involved": 1, "feat_is_night": 1, "feat_is_monsoon": 0},
        {"crash_id": "C2", "segment_id": "SEG_001", "severity_code": 2, "feat_pedestrian_involved": 0, "feat_is_night": 0, "feat_is_monsoon": 1},
        {"crash_id": "C3", "segment_id": "SEG_001", "severity_code": 1, "feat_pedestrian_involved": 0, "feat_is_night": 1, "feat_is_monsoon": 0},
        # SEG_002 has 1 minor crash
        {"crash_id": "C4", "segment_id": "SEG_002", "severity_code": 0, "feat_pedestrian_involved": 0, "feat_is_night": 0, "feat_is_monsoon": 0},
    ])
    ranked = rank_and_tier_segments(segments, crashes, output_csv=None, top_blackspots_csv=None)
    assert "segment_risk_score" in ranked.columns
    assert (ranked["segment_risk_score"] >= 0).all()
    assert (ranked["segment_risk_score"] <= 100).all()
    assert ranked.iloc[0]["segment_id"] == "SEG_001"
    assert ranked.iloc[0]["segment_risk_score"] == 100.0


def test_polygon_point_in_polygon():
    from src.data.crash_cleaner import _point_in_polygon
    # Inside Kolkata KMC perimeter
    assert _point_in_polygon(22.5650, 88.3500) is True
    # Outside Kolkata
    assert _point_in_polygon(21.0000, 88.3500) is False


def test_multi_station_weather_idw():
    from src.data.weather_loader import KOLKATA_WEATHER_STATIONS
    from src.features.weather_features import merge_and_extract_weather_features
    assert len(KOLKATA_WEATHER_STATIONS) == 4

    stn_df = pd.DataFrame([
        {"timestamp": "2023-06-15 14:00:00", "station_id": "STN_ALIPORE", "station_lat": 22.53, "station_lon": 88.33, "temperature_2m": 30.0, "precipitation": 0.0, "visibility": 9000.0, "relative_humidity_2m": 70.0},
        {"timestamp": "2023-06-15 14:00:00", "station_id": "STN_DUMDUM", "station_lat": 22.65, "station_lon": 88.45, "temperature_2m": 34.0, "precipitation": 5.0, "visibility": 4000.0, "relative_humidity_2m": 85.0},
    ])
    crashes = pd.DataFrame([
        {"crash_id": "C_ALIPORE", "timestamp": "2023-06-15 14:15:00", "latitude": 22.531, "longitude": 88.331},
        {"crash_id": "C_DUMDUM", "timestamp": "2023-06-15 14:15:00", "latitude": 22.649, "longitude": 88.449},
    ])
    merged = merge_and_extract_weather_features(crashes, stn_df)
    assert len(merged) == 2
    row_alipore = merged[merged["crash_id"] == "C_ALIPORE"].iloc[0]
    assert abs(row_alipore["feat_temperature"] - 30.0) < 0.5
    row_dumdum = merged[merged["crash_id"] == "C_DUMDUM"].iloc[0]
    assert row_dumdum["feat_precipitation"] > 3.0


def test_calibration_and_pr_auc():
    from src.models.evaluate import evaluate_predictions, export_calibration_curves
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    y_proba = np.array([
        [0.85, 0.10, 0.05],
        [0.80, 0.15, 0.05],
        [0.10, 0.80, 0.10],
        [0.15, 0.75, 0.10],
        [0.05, 0.15, 0.80],
        [0.05, 0.10, 0.85],
    ])
    metrics = evaluate_predictions(y_true, y_pred, y_proba)
    assert "pr_auc_fatal" in metrics
    assert "calibration_curves" in metrics
    assert "fatal" in metrics["calibration_curves"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_json = os.path.join(tmpdir, "calib.json")
        export_calibration_curves(metrics, tmp_json)
        assert os.path.exists(tmp_json)

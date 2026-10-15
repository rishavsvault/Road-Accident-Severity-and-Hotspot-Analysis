"""FastAPI REST Service for Road Safety Severity Prediction, Hotspots, and Causal Interventions."""
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from src.api.schemas import (
    IncidentFeatures,
    PredictionResponse,
    InterventionRequest,
    InterventionResponse,
    HotspotSummary,
    SpatialDiagnosticsResponse
)
from src.explainability.intervention_engine import KMDAInterventionEngine
from src.hotspots.spatial_stats import compute_global_morans_i

app = FastAPI(
    title="Kolkata Urban Road Safety API",
    description="Production REST API for police dispatch, road safety audits, and civil intervention planning.",
    version="2.0.0"
)

# Global in-memory cache for models and datasets
_MODEL_CACHE = {}


def get_model():
    if "severity_model" not in _MODEL_CACHE:
        model_path = "data/processed/xgboost_severity_model.joblib"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=503, detail="Trained model artifact not found. Please run the training pipeline.")
        data = joblib.load(model_path)
        _MODEL_CACHE["severity_model"] = data["model"]
        _MODEL_CACHE["calibrated_model"] = data.get("calibrated_model")
        _MODEL_CACHE["features"] = data["features"]
    return _MODEL_CACHE


def get_segments_data() -> pd.DataFrame:
    path = "data/processed/segment_risk_tiers.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="Segment risk tiers dataset not found.")
    return pd.read_csv(path)


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Road Accident Severity & Hotspot Analysis Engine",
        "version": "2.0.0",
        "city": "Kolkata Metropolitan Area (KMDA)"
    }


@app.post("/api/v1/predict-severity", response_model=PredictionResponse)
def predict_severity(payload: IncidentFeatures):
    """Predicts crash severity probabilities (Minor, Grievous, Fatal) using calibrated XGBoost."""
    cache = get_model()
    model = cache.get("calibrated_model") or cache["severity_model"]
    feature_names = cache["features"]

    # Construct feature mapping
    row_dict = {
        "feat_speed_limit": payload.speed_limit,
        "feat_is_high_speed": 1 if payload.speed_limit >= 50 else 0,
        "feat_lanes": payload.lanes,
        "feat_has_median": payload.has_median,
        "feat_is_lit": payload.is_lit,
        "feat_is_junction": payload.is_junction,
        "feat_junction_density": payload.junction_density,
        "feat_is_night": payload.is_night,
        "feat_is_weekend": payload.is_weekend,
        "feat_is_monsoon": payload.is_monsoon,
        "feat_precipitation": payload.precipitation,
        "feat_is_raining": 1 if payload.precipitation > 0.1 else 0,
        "feat_visibility_km": payload.visibility_km,
        "feat_is_low_visibility": 1 if payload.visibility_km < 2.5 else 0,
        "feat_temperature": payload.temperature,
        "feat_pedestrian_involved": payload.pedestrian_involved,
        "feat_commercial_vehicle": payload.commercial_vehicle,
        "feat_two_wheeler": payload.two_wheeler,
        "feat_is_two_wheeler": payload.two_wheeler,
        "feat_is_heavy_vehicle": payload.commercial_vehicle,
        "feat_kinetic_energy": (payload.speed_limit ** 2) / 1000.0,
        "feat_vulnerable_high_speed": 1 if ((payload.two_wheeler or payload.pedestrian_involved) and payload.speed_limit >= 50) else 0,
        "feat_unlit_night": 1 if (payload.is_night and not payload.is_lit) else 0,
        "feat_wet_high_speed": 1 if (payload.precipitation > 0.1 and payload.speed_limit >= 50) else 0,
        "feat_peak_traffic_window": 0
    }

    # Vectorize strictly matching trained feature names
    vec = [row_dict.get(col, 0.0) for col in feature_names]
    X = np.array([vec], dtype=float)

    probs = model.predict_proba(X)[0]
    pred_code = int(np.argmax(probs))
    labels = ["Minor Injury", "Grievous Injury", "Fatal"]

    risk_tier = "High Alert" if probs[2] >= 0.35 else ("Medium Alert" if probs[1] + probs[2] >= 0.50 else "Low Alert")

    return PredictionResponse(
        predicted_severity_code=pred_code,
        predicted_severity_label=labels[pred_code],
        probabilities={
            "minor_injury": round(float(probs[0]), 4),
            "grievous_injury": round(float(probs[1]), 4),
            "fatal": round(float(probs[2]), 4)
        },
        risk_tier=risk_tier,
        is_calibrated=(cache.get("calibrated_model") is not None)
    )


@app.get("/api/v1/hotspots", response_model=List[HotspotSummary])
def get_top_hotspots(limit: int = Query(20, ge=1, le=100)):
    """Returns top prioritized high-risk road segments with exposure-adjusted metrics."""
    df = get_segments_data()
    # Filter to segments with crashes and sort by risk score
    ranked = df[df["total_crashes"] > 0].sort_values(by="segment_risk_score", ascending=False).head(limit)

    results = []
    for _, r in ranked.iterrows():
        results.append(HotspotSummary(
            segment_id=str(r["segment_id"]),
            corridor_name=str(r.get("corridor_name", "Urban Corridor")),
            total_crashes=int(r["total_crashes"]),
            fatal_crashes=int(r.get("fatal_crashes", 0)),
            segment_risk_score=float(r["segment_risk_score"]),
            priority_tier=str(r.get("priority_tier", "High Priority")),
            crash_rate_per_mvkt=float(r.get("crash_rate_per_mvkt", 0.0)),
            eb_expected_crashes=float(r.get("eb_expected_crashes")) if pd.notnull(r.get("eb_expected_crashes")) else None,
            potential_for_safety_improvement=float(r.get("potential_for_safety_improvement")) if pd.notnull(r.get("potential_for_safety_improvement")) else None,
            mid_lat=float(r["mid_lat"]),
            mid_lon=float(r["mid_lon"])
        ))
    return results


@app.post("/api/v1/simulate-intervention", response_model=InterventionResponse)
def simulate_intervention(payload: InterventionRequest):
    """Simulates civil engineering interventions using both ML and empirical CMF causal models."""
    cache = get_model()
    model = cache.get("calibrated_model") or cache["severity_model"]
    feature_names = cache["features"]

    b_dict = payload.baseline.model_dump()
    p_dict = payload.proposed.model_dump()

    b_feats = {f"feat_{k}": v for k, v in b_dict.items()}
    p_feats = {f"feat_{k}": v for k, v in p_dict.items()}

    b_vec = np.array([[b_feats.get(c, 0.0) for c in feature_names]], dtype=float)
    p_vec = np.array([[p_feats.get(c, 0.0) for c in feature_names]], dtype=float)

    p_base = model.predict_proba(b_vec)[0]
    p_sim = model.predict_proba(p_vec)[0]

    engine = KMDAInterventionEngine()
    causal_res = engine.estimate_causal_countermeasure_impact(b_dict, p_dict, p_base, p_sim)

    # Recommendations
    dummy_seg = {**b_dict, "night_crashes": 3 if b_dict.get("is_night") else 0}
    actions = engine.recommend_for_segment(dummy_seg)

    return InterventionResponse(
        ml_predicted_fatal_pct=causal_res["ml_predicted_fatal_pct"],
        ml_fatal_delta_pct=causal_res["ml_fatal_delta_pct"],
        causal_fatal_delta_pct=causal_res["causal_fatal_delta_pct"],
        consensus_fatal_delta_pct=causal_res["consensus_fatal_delta_pct"],
        cmf_compound_multiplier=causal_res["cmf_compound_multiplier"],
        cmf_citations=causal_res["cmf_citations"],
        recommended_civil_actions=actions
    )


@app.get("/api/v1/spatial-diagnostics", response_model=SpatialDiagnosticsResponse)
def get_spatial_diagnostics():
    """Computes Global Moran's I spatial autocorrelation for network crash patterns."""
    df = get_segments_data()
    res = compute_global_morans_i(df, metric_col="total_crashes")
    return SpatialDiagnosticsResponse(**res)

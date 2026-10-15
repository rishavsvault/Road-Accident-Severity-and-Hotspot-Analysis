"""Pydantic request and response schemas for Kolkata Road Safety Production API."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class IncidentFeatures(BaseModel):
    speed_limit: float = Field(default=50.0, description="Road speed limit in km/h", ge=20.0, le=100.0)
    lanes: float = Field(default=4.0, description="Number of carriage lanes", ge=1.0, le=10.0)
    has_median: int = Field(default=1, description="Physical median barrier present (1=Yes, 0=No)", ge=0, le=1)
    is_lit: int = Field(default=1, description="Street lighting present (1=Yes, 0=No)", ge=0, le=1)
    is_junction: int = Field(default=0, description="Intersection / junction (1=Yes, 0=No)", ge=0, le=1)
    junction_density: float = Field(default=1.5, description="Junctions per km", ge=0.0)
    is_night: int = Field(default=0, description="Night hours 18:00 to 06:00 (1=Yes, 0=No)", ge=0, le=1)
    is_weekend: int = Field(default=0, description="Saturday or Sunday (1=Yes, 0=No)", ge=0, le=1)
    is_monsoon: int = Field(default=0, description="June to September (1=Yes, 0=No)", ge=0, le=1)
    precipitation: float = Field(default=0.0, description="Precipitation rate in mm/h", ge=0.0)
    visibility_km: float = Field(default=8.0, description="Visibility in km", ge=0.1, le=10.0)
    temperature: float = Field(default=28.0, description="Ambient temperature in Celsius")
    pedestrian_involved: int = Field(default=0, description="Pedestrian involved (1=Yes, 0=No)", ge=0, le=1)
    commercial_vehicle: int = Field(default=0, description="Truck/Bus involved (1=Yes, 0=No)", ge=0, le=1)
    two_wheeler: int = Field(default=1, description="Two wheeler involved (1=Yes, 0=No)", ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_severity_code: int
    predicted_severity_label: str
    probabilities: Dict[str, float]
    risk_tier: str
    is_calibrated: bool


class InterventionRequest(BaseModel):
    segment_id: Optional[str] = "SEG_001"
    baseline: IncidentFeatures
    proposed: IncidentFeatures


class InterventionResponse(BaseModel):
    ml_predicted_fatal_pct: float
    ml_fatal_delta_pct: float
    causal_fatal_delta_pct: float
    consensus_fatal_delta_pct: float
    cmf_compound_multiplier: float
    cmf_citations: List[str]
    recommended_civil_actions: List[Dict[str, Any]]


class HotspotSummary(BaseModel):
    segment_id: str
    corridor_name: Optional[str]
    total_crashes: int
    fatal_crashes: int
    segment_risk_score: float
    priority_tier: str
    crash_rate_per_mvkt: float
    eb_expected_crashes: Optional[float]
    potential_for_safety_improvement: Optional[float]
    mid_lat: float
    mid_lon: float


class SpatialDiagnosticsResponse(BaseModel):
    morans_i: float
    expected_i: float
    z_score: float
    p_value: float
    interpretation: str

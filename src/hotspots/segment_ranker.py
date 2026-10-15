"""Aggregates crash severity into segment-level risk tiers with traffic and population exposure normalization."""
import os
import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("segment_ranker")

# Estimated Annual Average Daily Traffic (AADT) baseline by Kolkata road hierarchy
AADT_BY_HIGHWAY_TYPE = {
    "motorway": 75000,
    "trunk": 60000,
    "primary": 45000,
    "secondary": 25000,
    "tertiary": 14000,
    "residential": 6000,
}

# Population density exposure multiplier (normalized relative to metropolitan mean)
ZONE_POPULATION_EXPOSURE = {
    "Central":        1.45,   # High mixed pedestrian & market density (Burrabazar, BBD Bagh, Park St)
    "North":          1.35,   # Dense historic residential & transit corridors (Shyambazar, Cossipore)
    "North-East":     1.10,   # Airport corridor & planned township access (Dum Dum, VIP Rd)
    "East":           1.05,   # Planned IT corridor & bypass (Salt Lake, EM Bypass)
    "South-Central":  1.25,   # High-density institutional & residential fabric (Ballygunge, Bhowanipore)
    "South":          1.15,   # Dense southern sprawl (Jadavpur, Garia, Tollygunge)
    "South-West":     1.20,   # High freight & arterial congestion (Behala, Diamond Harbour Rd)
    "West":           0.95,   # Port & industrial freight transit (Khidirpur, Strand Rd, Hastings)
}


def rank_and_tier_segments(segments_df: pd.DataFrame,
                           snapped_crashes_df: pd.DataFrame,
                           output_csv: Optional[str] = "data/processed/segment_risk_tiers.csv",
                           top_blackspots_csv: Optional[str] = "data/processed/kmda_top20_blackspots.csv") -> pd.DataFrame:
    """Computes composite severity-weighted risk index per segment with traffic & population exposure normalization.

    Zero-crash segments are labeled 'Unmonitored (No Data)' — distinct from modelled
    low-risk segments — to prevent misleading 'safe corridor' signals.
    """
    logger.info("Computing segment-level risk aggregation with traffic and population exposure normalization...")

    # Count by severity per segment
    crash_agg = snapped_crashes_df.groupby("segment_id").agg(
        total_crashes=("crash_id", "count"),
        fatal_crashes=("severity_code", lambda s: (s == 2).sum()),
        grievous_crashes=("severity_code", lambda s: (s == 1).sum()),
        minor_crashes=("severity_code", lambda s: (s == 0).sum()),
        pedestrian_crashes=("feat_pedestrian_involved", "sum"),
        night_crashes=("feat_is_night", "sum"),
        monsoon_crashes=("feat_is_monsoon", "sum")
    ).reset_index()

    merged = pd.merge(segments_df, crash_agg, on="segment_id", how="left").fillna(0)

    # 1. Traffic Exposure Modeling (AADT & Million Vehicle Kilometers Traveled - MVKT)
    def compute_aadt(row):
        hwy = str(row.get("highway_type", "primary")).lower()
        base_aadt = AADT_BY_HIGHWAY_TYPE.get(hwy, 35000)
        lanes = max(1, int(row.get("lanes", 2)))
        # Lane adjustment factor (roads with more lanes carry proportionally higher volume)
        lane_factor = 0.70 + 0.15 * lanes
        return int(base_aadt * lane_factor)

    merged["estimated_aadt"] = merged.apply(compute_aadt, axis=1)

    # 2-Year Study Vehicle-Kilometers Traveled (MVKT in millions)
    # MVKT = (AADT * length_km * 365 days * 2 years) / 1,000,000
    length_km = merged["length_meters"].clip(lower=20.0) / 1000.0
    merged["two_year_mvkt"] = np.round((merged["estimated_aadt"] * length_km * 365.0 * 2.0) / 1e6, 3)

    # 2. Population Exposure Modeling
    if "spatial_zone_id" in merged.columns:
        merged["population_exposure_factor"] = merged["spatial_zone_id"].map(ZONE_POPULATION_EXPOSURE).fillna(1.0)
    else:
        merged["population_exposure_factor"] = 1.0

    # 3. Crash rate per Million Vehicle Kilometers Traveled (standard HSM exposure metric)
    merged["crash_rate_per_mvkt"] = np.round(
        merged["total_crashes"] / merged["two_year_mvkt"].clip(lower=0.01), 2
    )

    # Crash rate per km (simple linear exposure metric)
    merged["crash_rate_per_km"] = np.round(
        merged["total_crashes"] / length_km.clip(lower=0.05), 2
    )

    # 4. Highway Safety Manual (HSM) Empirical Bayes (EB) Expected Crash Estimation
    # SPF Baseline: expected crashes as a function of exposure (AADT and length)
    base_spf_rate = 0.00012 * (merged["estimated_aadt"] ** 0.82) * length_km
    merged["spf_predicted_crashes"] = np.round(base_spf_rate, 2)
    # Overdispersion parameter k=0.25 (typical for urban arterials)
    k_dispersion = 0.25
    eb_weight = 1.0 / (1.0 + k_dispersion * merged["spf_predicted_crashes"])
    merged["eb_expected_crashes"] = np.round(
        eb_weight * merged["spf_predicted_crashes"] + (1.0 - eb_weight) * merged["total_crashes"], 2
    )
    # Potential for Safety Improvement (PSI = Excess crashes over baseline expectation)
    merged["potential_for_safety_improvement"] = np.round(
        np.maximum(0.0, merged["eb_expected_crashes"] - merged["spf_predicted_crashes"]), 2
    )

    # 5. Severity-Weighted Exposure Adjusted Composite Score
    # Severity weighting: Fatal*10 + Grievous*3 + Minor*1
    weighted_sum = (
        merged["fatal_crashes"] * 10.0 +
        merged["grievous_crashes"] * 3.0 +
        merged["minor_crashes"] * 1.0
    )

    # Composite Exposure: combined traffic (MVKT) and pedestrian/population exposure
    exposure_denom = np.sqrt(merged["two_year_mvkt"].clip(lower=0.05)) * merged["population_exposure_factor"]
    length_adj = merged["length_meters"].clip(lower=50.0) / 100.0

    # Hybrid risk: 65% severity density per 100m + 35% exposure-normalized rate
    density_component = weighted_sum / length_adj
    exposure_component = (weighted_sum / exposure_denom) * 0.40
    raw_score = 0.65 * density_component + 0.35 * exposure_component
    merged["raw_risk_score"] = np.round(raw_score, 2)

    # Standardize to 0 - 100 scale (0 = zero risk, 100 = peak corridor risk in Kolkata)
    max_raw = raw_score.max()
    if max_raw > 0:
        merged["segment_risk_score"] = np.round((raw_score / max_raw) * 100.0, 1)
    else:
        merged["segment_risk_score"] = 0.0

    # Risk Tiers: compute thresholds only on segments that have crash records
    has_crashes = merged["total_crashes"] > 0
    if has_crashes.sum() > 0:
        high_thresh = merged.loc[has_crashes, "segment_risk_score"].quantile(0.85)
        med_thresh  = merged.loc[has_crashes, "segment_risk_score"].quantile(0.50)
    else:
        high_thresh, med_thresh = 70.0, 35.0

    def assign_tier(row):
        if row["total_crashes"] == 0:
            return "Unmonitored (No Data)"
        elif row["segment_risk_score"] >= high_thresh:
            return "High Priority"
        elif row["segment_risk_score"] >= med_thresh:
            return "Medium Priority"
        else:
            return "Low Priority / Baseline"

    merged["priority_tier"] = merged.apply(assign_tier, axis=1)

    # Rank descending (zero-crash segments at the bottom of the list)
    merged = merged.sort_values(by="segment_risk_score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        merged.to_csv(output_csv, index=False)
        logger.info(f"Segment risk tiers saved -> {output_csv}")

    unmonitored = (merged["priority_tier"] == "Unmonitored (No Data)").sum()
    logger.info(
        f"Tier breakdown -> High: {(merged['priority_tier']=='High Priority').sum()} | "
        f"Medium: {(merged['priority_tier']=='Medium Priority').sum()} | "
        f"Low: {(merged['priority_tier']=='Low Priority / Baseline').sum()} | "
        f"Unmonitored: {unmonitored}"
    )

    # Top-20 KMDA Blackspots (only segments with crash data)
    top_20 = merged[merged["total_crashes"] > 0].head(20).copy()
    if top_blackspots_csv:
        os.makedirs(os.path.dirname(top_blackspots_csv), exist_ok=True)
        top_20.to_csv(top_blackspots_csv, index=False)
        logger.info(f"Top-20 KMDA Action Segments saved -> {top_blackspots_csv}")

    return merged


if __name__ == "__main__":
    segments = pd.read_csv("data/interim/kolkata_segments.csv")
    crashes = pd.read_parquet("data/processed/crash_level_features.parquet")
    rank_and_tier_segments(segments, crashes)

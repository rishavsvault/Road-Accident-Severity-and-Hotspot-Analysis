"""Aggregates crash severity into segment-level risk tiers (Level 2 model) and outputs KMDA Top-20 list."""
import os
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("segment_ranker")

def rank_and_tier_segments(segments_df: pd.DataFrame,
                           snapped_crashes_df: pd.DataFrame,
                           output_csv: str = "data/processed/segment_risk_tiers.csv",
                           top_blackspots_csv: str = "data/processed/kmda_top20_blackspots.csv") -> pd.DataFrame:
    """Computes composite severity-weighted risk index per segment and bins into Priority Tiers."""
    logger.info("Computing segment-level risk aggregation and priority tiers...")

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

    # Severity weighted composite score (Fatal*10 + Grievous*3 + Minor*1)
    # Normalized per 100 meters
    length_adj = merged["length_meters"].clip(lower=50.0) / 100.0
    weighted_sum = (
        merged["fatal_crashes"] * 10.0 +
        merged["grievous_crashes"] * 3.0 +
        merged["minor_crashes"] * 1.0
    )
    raw_score = weighted_sum / length_adj
    merged["raw_risk_score"] = np.round(raw_score, 2)

    # Standardize to 0 - 100 scale (0 = zero risk, 100 = peak corridor risk in Kolkata)
    max_raw = raw_score.max()
    if max_raw > 0:
        merged["segment_risk_score"] = np.round((raw_score / max_raw) * 100.0, 1)
    else:
        merged["segment_risk_score"] = 0.0

    # Risk Tiers: High (top 15%), Medium (next 35%), Low (remaining 50%)
    high_thresh = merged["segment_risk_score"].quantile(0.85)
    med_thresh = merged["segment_risk_score"].quantile(0.50)

    def assign_tier(score):
        if score >= high_thresh and score > 0:
            return "High Priority"
        elif score >= med_thresh and score > 0:
            return "Medium Priority"
        else:
            return "Low Priority / Baseline"

    merged["priority_tier"] = merged["segment_risk_score"].apply(assign_tier)

    # Rank descending
    merged = merged.sort_values(by="segment_risk_score", ascending=False).reset_index(drop=True)
    merged["rank"] = merged.index + 1

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    merged.to_csv(output_csv, index=False)
    logger.info(f"Segment risk tiers saved -> {output_csv}")

    # Top-20 KMDA Blackspots
    top_20 = merged.head(20).copy()
    top_20.to_csv(top_blackspots_csv, index=False)
    logger.info(f"Top-20 KMDA Action Segments saved -> {top_blackspots_csv}")

    return merged

"""Master End-to-End Pipeline Execution Script.

Runs Stages 2 through 7:
1. Data collection/simulation (crashes, weather, OSM segments)
2. Spatial snapping & road network alignment
3. Feature engineering (road, temporal, weather)
4. Network KDE & Spatial Hotspot statistical ranking
5. Spatial Grouped Cross-Validation & XGBoost Training
6. SHAP feature importance extraction & KMDA intervention export
"""
import os
import time
from src.utils.logger import get_logger
from src.data.mock_data_generator import generate_kolkata_crashes
from src.data.crash_cleaner import clean_crash_data
from src.data.weather_loader import fetch_or_generate_weather
from src.data.osm_loader import get_kolkata_road_segments
from src.data.spatial_snapper import snap_crashes_to_segments
from src.features.build_features import build_feature_matrix, FEATURE_COLUMNS
from src.hotspots.network_kde import calculate_network_kde
from src.hotspots.spatial_stats import compute_getis_ord_gi
from src.hotspots.segment_ranker import rank_and_tier_segments
from src.models.xgboost_trainer import SeverityXGBoost
from src.explainability.shap_explainer import SeveritySHAPExplainer
from src.explainability.intervention_engine import KMDAInterventionEngine

logger = get_logger("run_pipeline")

def run():
    start_time = time.time()
    logger.info("================================================================")
    logger.info("Starting Kolkata Road Accident Severity & Hotspot Pipeline")
    logger.info("Target Authority: Kolkata Metropolitan Development Authority (KMDA)")
    logger.info("================================================================")

    # 1. Data Ingestion & Cleaning
    raw_crash_path = "data/raw/crashes/crashes.csv"
    if not os.path.exists(raw_crash_path):
        generate_kolkata_crashes(n_samples=3200, output_path=raw_crash_path)

    cleaned_crashes = clean_crash_data(
        input_csv=raw_crash_path,
        output_parquet="data/interim/cleaned_crashes.parquet"
    )

    weather_df = fetch_or_generate_weather()
    segments_df = get_kolkata_road_segments()

    # 2. Spatial Snapping
    snapped_crashes = snap_crashes_to_segments(cleaned_crashes, segments_df)

    # 3. Feature Matrix
    features_df = build_feature_matrix(
        snapped_crashes_df=snapped_crashes,
        weather_df=weather_df,
        output_parquet="data/processed/crash_level_features.parquet"
    )

    # 4. Hotspot Analysis (Network KDE + Getis-Ord Gi* + Segment Tiering)
    kde_segments = calculate_network_kde(segments_df, features_df, bandwidth_meters=400.0)
    gi_segments = compute_getis_ord_gi(kde_segments, metric_col="net_kde_score")
    ranked_segments = rank_and_tier_segments(
        segments_df=gi_segments,
        snapped_crashes_df=features_df,
        output_csv="data/processed/segment_risk_tiers.csv",
        top_blackspots_csv="data/processed/kmda_top20_blackspots.csv"
    )

    # 5. XGBoost Modeling with Spatial Grouped Cross-Validation
    logger.info("Training Level 1 Crash Severity XGBoost model...")
    trainer = SeverityXGBoost(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42
    )
    cv_metrics, _, _ = trainer.fit_with_spatial_cv(
        df=features_df,
        feature_cols=FEATURE_COLUMNS,
        target_col="severity_code",
        group_col="spatial_zone_id",
        n_splits=5
    )
    trainer.save_model("data/processed/xgboost_severity_model.joblib")

    # 6. SHAP Explainability & Global Diagnostics
    logger.info("Computing SHAP explanations on test sample...")
    shap_tool = SeveritySHAPExplainer(trainer.model, FEATURE_COLUMNS)
    sample_feat = features_df.sample(min(400, len(features_df)), random_state=42)
    importance_df = shap_tool.get_global_feature_importance(sample_feat)
    importance_df.to_csv("data/processed/shap_global_feature_importance.csv", index=False)
    logger.info(f"Top 5 SHAP Risk Drivers for Fatal Crashes:\n{importance_df.head(5).to_string()}")

    # 7. KMDA Engineering Interventions for Top 20 Blackspots
    engine = KMDAInterventionEngine()
    top_20 = ranked_segments.head(20).to_dict(orient="records")
    action_records = []
    for seg in top_20:
        fixes = engine.recommend_for_segment(seg)
        for fix in fixes:
            action_records.append({
                "segment_id": seg["segment_id"],
                "corridor_name": seg["corridor_name"],
                "priority_tier": seg["priority_tier"],
                "risk_score": seg["segment_risk_score"],
                "intervention_id": fix["intervention_id"],
                "authority": fix["authority"],
                "action": fix["recommended_action"],
                "estimated_cost_tier": fix["estimated_cost_tier"]
            })
    import pandas as pd
    pd.DataFrame(action_records).to_csv("data/processed/kmda_prescribed_actions.csv", index=False)
    logger.info("KMDA Prescribed Interventions exported -> data/processed/kmda_prescribed_actions.csv")

    elapsed = round(time.time() - start_time, 2)
    logger.info("================================================================")
    logger.info(f"PIPELINE RUN COMPLETE in {elapsed} seconds!")
    logger.info(f"Spatial CV Macro-F1: {cv_metrics['macro_f1']} | Fatal Recall: {cv_metrics['fatal_recall']}")
    logger.info("Launch dashboard with: streamlit run dashboard/app.py")
    logger.info("================================================================")

if __name__ == "__main__":
    run()

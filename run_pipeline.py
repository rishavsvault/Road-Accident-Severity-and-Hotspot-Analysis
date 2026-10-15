"""Master End-to-End Pipeline Execution Script.

Stages executed:
1. Data collection/simulation (crashes, multi-station weather, OSM segments)
2. Spatial snapping & road network alignment
3. Feature engineering (road, temporal, multi-station interpolated weather)
4. Network KDE, Getis-Ord Gi*, & Exposure-Normalized Spatial Hotspot Ranking
5. Multiclass XGBoost + Ordinal model with Spatial Grouped CV & Optuna Best Params
6. Probability calibration (Isotonic / Platt scaling)
7. PR-AUC, calibration curve export, SHAP feature importance, & KMDA intervention export
"""
import os
import sys
import yaml
import time
import argparse
import pandas as pd
from src.utils.logger import get_logger
from src.data.mock_data_generator import generate_kolkata_crashes
from src.data.crash_cleaner import clean_crash_data
from src.data.weather_loader import fetch_or_generate_weather
from src.data.osm_loader import get_kolkata_road_segments
from src.data.spatial_snapper import snap_crashes_to_segments
from src.features.build_features import build_feature_matrix, FEATURE_COLUMNS
from src.hotspots.network_kde import calculate_network_kde
from src.hotspots.spatial_stats import compute_getis_ord_gi, compute_global_morans_i
from src.hotspots.segment_ranker import rank_and_tier_segments
from src.models.xgboost_trainer import SeverityXGBoost, OrdinalSeverityModel
from src.models.evaluate import evaluate_predictions, export_calibration_curves, evaluate_temporal_walk_forward
from src.models.tune import run_optuna_tuning
from src.explainability.shap_explainer import SeveritySHAPExplainer
from src.explainability.intervention_engine import KMDAInterventionEngine

logger = get_logger("run_pipeline")

DEFAULT_TUNED_PARAMS_PATH = "configs/best_tuned_params.yaml"


def _load_best_params(path: str = DEFAULT_TUNED_PARAMS_PATH) -> dict:
    """Loads Optuna best-tuned hyperparameters if present, otherwise defaults."""
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                params = yaml.safe_load(f)
                if isinstance(params, dict):
                    logger.info(f"Loaded Optuna best-tuned hyperparameters from {path}")
                    return params
        except Exception as e:
            logger.warning(f"Could not read {path} ({e}). Using default hyperparameters.")
    return {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.024,
        "subsample": 0.886,
        "colsample_bytree": 0.891,
        "calibration_method": "isotonic"
    }


def run(run_optuna: bool = False, force_osmnx: bool = False):
    start_time = time.time()
    logger.info("================================================================")
    logger.info("Starting Kolkata Road Accident Severity & Hotspot Pipeline")
    logger.info("OFFICIAL DATA DISCLAIMER:")
    logger.info("This system currently runs on SYNTHETIC crash distributions calibrated")
    logger.info("to MoRTH national benchmarks. Real Kolkata Police / KMDA FIR records")
    logger.info("have not yet been ingested. Outputs are for methodology validation only.")
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
    segments_df = get_kolkata_road_segments(force_osmnx_download=force_osmnx)

    # 2. Spatial Snapping
    snapped_crashes = snap_crashes_to_segments(cleaned_crashes, segments_df)

    # 3. Feature Matrix
    features_df = build_feature_matrix(
        snapped_crashes_df=snapped_crashes,
        weather_df=weather_df,
        output_parquet="data/processed/crash_level_features.parquet"
    )

    # 4. Hotspot Analysis (Network KDE + Getis-Ord Gi* + Exposure-Normalized Segment Tiering)
    kde_segments = calculate_network_kde(segments_df, features_df, bandwidth_meters=400.0)
    gi_segments = compute_getis_ord_gi(kde_segments, metric_col="net_kde_score")
    ranked_segments = rank_and_tier_segments(
        segments_df=gi_segments,
        snapped_crashes_df=features_df,
        output_csv="data/processed/segment_risk_tiers.csv",
        top_blackspots_csv="data/processed/kmda_top20_blackspots.csv"
    )

    # 4b. Spatial Diagnostics (Global Moran's I Autocorrelation)
    moran_stats = compute_global_morans_i(ranked_segments, metric_col="total_crashes")
    import json
    with open("data/processed/spatial_autocorrelation_moran.json", "w") as f:
        json.dump(moran_stats, f, indent=2)

    # 5. Hyperparameter Loading / Live Optuna Tuning
    if run_optuna:
        logger.info("Running live Optuna hyperparameter tuning (n_trials=20)...")
        best_params = run_optuna_tuning(features_df, n_trials=20)
        os.makedirs(os.path.dirname(DEFAULT_TUNED_PARAMS_PATH), exist_ok=True)
        with open(DEFAULT_TUNED_PARAMS_PATH, "w") as f:
            yaml.dump(best_params, f)
        logger.info(f"Saved freshly tuned Optuna parameters -> {DEFAULT_TUNED_PARAMS_PATH}")
    else:
        best_params = _load_best_params()
        logger.info(f"Using Optuna tuned parameters: {best_params}")

    # 6. Multiclass XGBoost + Probability Calibration (Isotonic / Platt)
    logger.info("Training Level 1 Crash Severity XGBoost model (multiclass)...")
    calib_method = best_params.get("calibration_method", "isotonic")
    trainer = SeverityXGBoost(
        n_estimators=int(best_params.get("n_estimators", 200)),
        max_depth=int(best_params.get("max_depth", 5)),
        learning_rate=float(best_params.get("learning_rate", 0.024)),
        subsample=float(best_params.get("subsample", 0.886)),
        colsample_bytree=float(best_params.get("colsample_bytree", 0.891)),
        calibration_method=calib_method,
        random_state=42
    )
    cv_metrics, oof_preds, oof_probs = trainer.fit_with_spatial_cv(
        df=features_df,
        feature_cols=FEATURE_COLUMNS,
        target_col="severity_code",
        group_col="spatial_zone_id",
        n_splits=5
    )
    trainer.save_model("data/processed/xgboost_severity_model.joblib")

    # Export Calibration Curve data points
    export_calibration_curves(cv_metrics, "data/processed/calibration_curves.json")

    # 6b. Temporal Walk-Forward Out-of-Time Backtesting
    oot_metrics = evaluate_temporal_walk_forward(
        df=features_df,
        feature_cols=FEATURE_COLUMNS,
        target_col="severity_code",
        time_col="timestamp",
        split_date="2024-01-01"
    )

    # 7. Ordinal Model Comparison
    logger.info("Training Ordinal (cumulative binary) comparison model...")
    X_all = features_df[FEATURE_COLUMNS].values.astype(float)
    y_all = features_df["severity_code"].values
    ordinal_model = OrdinalSeverityModel(random_state=42)
    ordinal_model.fit(X_all, y_all, feature_names=FEATURE_COLUMNS)
    ordinal_preds = ordinal_model.predict(X_all)
    ordinal_probs = ordinal_model.predict_proba(X_all)
    ordinal_metrics = evaluate_predictions(y_all, ordinal_preds, y_proba=ordinal_probs)
    ordinal_model.save("data/processed/ordinal_severity_model.joblib")
    logger.info(f"Ordinal Model -> Macro-F1: {ordinal_metrics['macro_f1']} | Fatal Recall: {ordinal_metrics['fatal_recall']}")

    # Save model performance summary (Spatial Grouped CV)
    comparison = pd.DataFrame([
        {"model": "XGBoost Multiclass (Calibrated)", **cv_metrics}
    ])
    comparison.to_csv("data/processed/model_comparison.csv", index=False)
    logger.info("Production model performance saved -> data/processed/model_comparison.csv")

    # 8. SHAP Explainability
    logger.info("Computing SHAP explanations on sample...")
    shap_tool = SeveritySHAPExplainer(trainer.model, FEATURE_COLUMNS)
    sample_feat = features_df.sample(min(400, len(features_df)), random_state=42)
    importance_df = shap_tool.get_global_feature_importance(sample_feat)
    importance_df.to_csv("data/processed/shap_global_feature_importance.csv", index=False)
    logger.info(f"Top 5 SHAP Risk Drivers for Fatal Crashes:\n{importance_df.head(5).to_string()}")

    # 9. KMDA Engineering Interventions for Top 20 Blackspots
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
    pd.DataFrame(action_records).to_csv("data/processed/kmda_prescribed_actions.csv", index=False)
    logger.info("KMDA Prescribed Interventions exported -> data/processed/kmda_prescribed_actions.csv")

    elapsed = round(time.time() - start_time, 2)
    logger.info("================================================================")
    logger.info(f"PIPELINE COMPLETE in {elapsed}s")
    logger.info(f"XGBoost Macro-F1: {cv_metrics['macro_f1']} | Fatal Recall: {cv_metrics['fatal_recall']} | "
                f"Fatal PR-AUC: {cv_metrics.get('pr_auc_fatal', 'N/A')}")
    logger.info(f"Ordinal Macro-F1: {ordinal_metrics['macro_f1']} | Fatal Recall: {ordinal_metrics['fatal_recall']}")
    logger.info("[DISCLAIMER] All outputs generated from synthetic data distributions.")
    logger.info("Launch dashboard with: streamlit run dashboard/app.py")
    logger.info("================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kolkata Road Safety Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter optimization")
    parser.add_argument("--force-osmnx", action="store_true", help="Attempt live OSMnx download of Kolkata network")
    args = parser.parse_args()
    run(run_optuna=args.tune, force_osmnx=args.force_osmnx)

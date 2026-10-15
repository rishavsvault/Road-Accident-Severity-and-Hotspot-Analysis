"""Evaluation metrics tailored to imbalanced accident severity and ordinal error penalties.

Includes PR-AUC per class, full calibration curves, Brier scores, and calibration exports.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    confusion_matrix,
    average_precision_score,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from src.utils.logger import get_logger

logger = get_logger("evaluate")

SEVERITY_LABELS = ["Minor", "Grievous", "Fatal"]


def evaluate_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         target_names: list = None) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics including macro-F1, fatal recall,
    quadratic kappa, PR-AUC per class, Brier scores, and calibration curve coordinates."""
    if target_names is None:
        target_names = SEVERITY_LABELS

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)

    # Quadratic weighted kappa (strongly penalizes predicting Minor when actual was Fatal)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred)

    fatal_recall = float(recalls[2]) if len(recalls) > 2 else 0.0
    grievous_recall = float(recalls[1]) if len(recalls) > 1 else 0.0
    minor_recall = float(recalls[0]) if len(recalls) > 0 else 0.0

    metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "fatal_recall": round(fatal_recall, 4),
        "grievous_recall": round(grievous_recall, 4),
        "minor_recall": round(minor_recall, 4),
        "fatal_precision": round(float(precisions[2]), 4) if len(precisions) > 2 else 0.0,
        "confusion_matrix": cm.tolist()
    }

    # PR-AUC & Calibration per class
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 3:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        pr_aucs = {}
        brier_scores = {}
        calibration_data = {}

        for i, name in enumerate(target_names):
            key = name.lower()
            if y_bin[:, i].sum() > 0:
                pr_aucs[f"pr_auc_{key}"] = round(
                    float(average_precision_score(y_bin[:, i], y_proba[:, i])), 4
                )
            brier_scores[f"brier_{key}"] = round(
                float(brier_score_loss(y_bin[:, i], y_proba[:, i])), 4
            )

            # Extract 5-bin calibration curve points
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_bin[:, i], y_proba[:, i], n_bins=5, strategy="uniform"
                )
                calibration_data[key] = {
                    "mean_predicted": [round(float(x), 4) for x in mean_pred],
                    "fraction_positives": [round(float(x), 4) for x in frac_pos],
                }
            except Exception:
                pass

        metrics.update(pr_aucs)
        metrics.update(brier_scores)
        metrics["calibration_curves"] = calibration_data

        # Calibration slope for fatal class
        if "fatal" in calibration_data and len(calibration_data["fatal"]["mean_predicted"]) >= 2:
            mp = calibration_data["fatal"]["mean_predicted"]
            fp = calibration_data["fatal"]["fraction_positives"]
            try:
                metrics["calibration_fatal_slope"] = round(float(np.polyfit(mp, fp, 1)[0]), 3)
            except Exception:
                pass

    logger.info(
        f"Evaluation Summary -> Macro-F1: {macro_f1:.4f} | "
        f"Fatal Recall: {fatal_recall:.4f} | Fatal PR-AUC: {metrics.get('pr_auc_fatal', 'N/A')} | "
        f"QWK: {qwk:.4f}"
    )
    return metrics


def export_calibration_curves(metrics: Dict[str, Any],
                              output_path: str = "data/processed/calibration_curves.json") -> None:
    """Exports calibration curve coordinates to JSON for dashboard visualization and audits."""
    if "calibration_curves" not in metrics:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics["calibration_curves"], f, indent=2)
    logger.info(f"Calibration curve coordinates exported -> {output_path}")


def evaluate_temporal_walk_forward(df,
                                   feature_cols: list,
                                   target_col: str = "severity_code",
                                   time_col: str = "timestamp",
                                   split_date: str = "2024-01-01",
                                   model_params: dict = None) -> Dict[str, Any]:
    """Evaluates chronological out-of-time generalization (train on historical, test on future).

    Prevents temporal data leakage and tests for concept drift in road safety conditions.
    """
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_sample_weight

    logger.info(f"Initiating Temporal Out-Of-Time Evaluation (Split Date: {split_date})...")
    df_eval = df.copy()
    df_eval[time_col] = pd.to_datetime(df_eval[time_col])

    train_mask = df_eval[time_col] < pd.to_datetime(split_date)
    test_mask = df_eval[time_col] >= pd.to_datetime(split_date)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        # Fallback to 70/30 chronological split if date doesn't cleanly partition
        df_sorted = df_eval.sort_values(by=time_col)
        split_idx = int(len(df_sorted) * 0.70)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
    else:
        train_df = df_eval[train_mask]
        test_df = df_eval[test_mask]

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df[target_col].values

    params = model_params or {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
        "objective": "multi:softprob", "num_class": 3, "random_state": 42, "n_jobs": -1
    }

    w_train = compute_sample_weight("balanced", y_train)
    w_train[y_train == 2] *= 1.8

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, sample_weight=w_train, verbose=False)

    probs = clf.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)

    oot_metrics = evaluate_predictions(y_test, preds, y_proba=probs)
    oot_metrics["temporal_train_samples"] = int(len(train_df))
    oot_metrics["temporal_test_samples"] = int(len(test_df))
    oot_metrics["split_date"] = str(split_date)

    out_file = "data/processed/temporal_evaluation_metrics.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    # Filter calibration curves out of json export to keep it clean
    export_dict = {k: v for k, v in oot_metrics.items() if k != "calibration_curves"}
    with open(out_file, "w") as f:
        json.dump(export_dict, f, indent=2)
    logger.info(f"Temporal Walk-Forward metrics saved -> {out_file} (F1: {oot_metrics['macro_f1']}, Fatal Recall: {oot_metrics['fatal_recall']})")

    return oot_metrics


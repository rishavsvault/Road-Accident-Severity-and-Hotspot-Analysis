"""Evaluation metrics tailored to imbalanced accident severity and ordinal error penalties."""
import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    classification_report,
    f1_score,
    recall_score,
    precision_score,
    cohen_kappa_score,
    confusion_matrix
)
from src.utils.logger import get_logger

logger = get_logger("evaluate")

def evaluate_predictions(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         target_names: list = None) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics including macro-F1, fatal recall, and quadratic kappa."""
    if target_names is None:
        target_names = ["Minor", "Grievous", "Fatal"]

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)

    # Quadratic weighted kappa (strongly penalizes predicting Minor when actual was Fatal)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred)

    fatal_recall = recalls[2] if len(recalls) > 2 else 0.0
    grievous_recall = recalls[1] if len(recalls) > 1 else 0.0
    minor_recall = recalls[0] if len(recalls) > 0 else 0.0

    metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "quadratic_weighted_kappa": round(float(qwk), 4),
        "fatal_recall": round(float(fatal_recall), 4),
        "grievous_recall": round(float(grievous_recall), 4),
        "minor_recall": round(float(minor_recall), 4),
        "fatal_precision": round(float(precisions[2]), 4) if len(precisions) > 2 else 0.0,
        "confusion_matrix": cm.tolist()
    }

    logger.info(f"Evaluation Summary -> Macro-F1: {macro_f1:.4f} | Fatal Recall: {fatal_recall:.4f} | QWK: {qwk:.4f}")
    return metrics

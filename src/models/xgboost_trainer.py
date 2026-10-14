"""XGBoost trainer supporting Multiclass and Ordinal classification with spatial CV."""
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
from sklearn.utils.class_weight import compute_sample_weight
from src.utils.logger import get_logger
from src.models.spatial_cv import SpatialGroupedCV
from src.models.evaluate import evaluate_predictions

logger = get_logger("xgboost_trainer")

class SeverityXGBoost:
    """XGBoost model suite for 3-class crash severity prediction with spatial CV."""

    def __init__(self,
                 n_estimators: int = 250,
                 max_depth: int = 5,
                 learning_rate: float = 0.05,
                 subsample: float = 0.85,
                 colsample_bytree: float = 0.80,
                 random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "random_state": random_state,
            "n_jobs": -1
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names = []

    def fit_with_spatial_cv(self,
                            df: pd.DataFrame,
                            feature_cols: list,
                            target_col: str = "severity_code",
                            group_col: str = "spatial_zone_id",
                            n_splits: int = 5) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Runs spatial cross-validation and records out-of-fold metrics."""
        self.feature_names = feature_cols
        X = df[feature_cols].values
        y = df[target_col].values

        cv = SpatialGroupedCV(n_splits=n_splits, group_col=group_col)
        oof_preds = np.zeros(len(df))
        oof_probs = np.zeros((len(df), 3))

        logger.info(f"Initiating Spatial Grouped Cross-Validation (K={n_splits})...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Heavy sample weighting towards Fatal (class 2) and Grievous (class 1)
            sample_weights = compute_sample_weight("balanced", y_train)
            sample_weights[y_train == 2] *= 1.8  # Extra penalty for fatal crash miss

            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            probs = fold_model.predict_proba(X_val)
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = np.argmax(probs, axis=1)

        metrics = evaluate_predictions(y, oof_preds)

        # Train final model on full dataset
        logger.info("Fitting final production model on complete dataset...")
        full_weights = compute_sample_weight("balanced", y)
        full_weights[y == 2] *= 1.8
        self.model.fit(X, y, sample_weight=full_weights, verbose=False)

        return metrics, oof_preds, oof_probs

    def save_model(self, output_path: str = "data/processed/xgboost_severity_model.joblib"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"model": self.model, "features": self.feature_names}, output_path)
        logger.info(f"Model saved to {output_path}")

    @staticmethod
    def load_model(model_path: str = "data/processed/xgboost_severity_model.joblib"):
        data = joblib.load(model_path)
        clf = SeverityXGBoost()
        clf.model = data["model"]
        clf.feature_names = data["features"]
        return clf

"""XGBoost Multiclass and Ordinal trainer with probability calibration and spatial CV."""
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from src.utils.logger import get_logger
from src.models.spatial_cv import SpatialGroupedCV
from src.models.evaluate import evaluate_predictions

logger = get_logger("xgboost_trainer")


class SeverityXGBoost:
    """XGBoost model suite for 3-class crash severity prediction with spatial CV
    and probability calibration."""

    def __init__(self,
                 n_estimators: int = 250,
                 max_depth: int = 5,
                 learning_rate: float = 0.05,
                 subsample: float = 0.85,
                 colsample_bytree: float = 0.80,
                 calibration_method: str = "isotonic",
                 random_state: int = 42):
        self.calibration_method = calibration_method if calibration_method in ["isotonic", "sigmoid"] else "isotonic"
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
        self.calibrated_model = None

    def _make_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Balanced class weights with moderate 1.25x safety penalty for fatal crash misses."""
        w = compute_sample_weight("balanced", y)
        w[y == 2] *= 1.25
        return w

    def fit_with_spatial_cv(self,
                            df: pd.DataFrame,
                            feature_cols: list,
                            target_col: str = "severity_code",
                            group_col: str = "spatial_zone_id",
                            n_splits: int = 5) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """Runs spatial cross-validation and records out-of-fold metrics with PR-AUC."""
        self.feature_names = feature_cols
        X = df[feature_cols].values.astype(float)
        y = df[target_col].values

        cv = SpatialGroupedCV(n_splits=n_splits, group_col=group_col)
        oof_preds = np.zeros(len(df))
        oof_probs = np.zeros((len(df), 3))

        logger.info(f"Initiating Spatial Grouped Cross-Validation (K={n_splits})...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            sample_weights = self._make_sample_weights(y_train)
            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(
                X_train, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Fit fold calibrator on training partition for true out-of-fold calibrated probabilities
            calib_fold = CalibratedClassifierCV(fold_model, cv=3, method=self.calibration_method)
            calib_fold.fit(X_train, y_train, sample_weight=sample_weights)
            probs = calib_fold.predict_proba(X_val)

            oof_probs[val_idx] = probs
            oof_preds[val_idx] = np.argmax(probs, axis=1)

        metrics = evaluate_predictions(y, oof_preds, y_proba=oof_probs)

        # Train final model on full dataset
        logger.info("Fitting final production XGBoost model on complete dataset...")
        full_weights = self._make_sample_weights(y)
        self.model.fit(X, y, sample_weight=full_weights, verbose=False)

        # Probability calibration (Isotonic regression or Platt scaling / sigmoid)
        calib_name = "Isotonic regression" if self.calibration_method == "isotonic" else "Platt scaling (sigmoid)"
        logger.info(f"Calibrating XGBoost probability outputs with {calib_name}...")
        calibrator = CalibratedClassifierCV(self.model, cv=3, method=self.calibration_method)
        calibrator.fit(X, y, sample_weight=full_weights)
        self.calibrated_model = calibrator
        logger.info("Calibration complete.")

        return metrics, oof_preds, oof_probs

    def predict_proba_calibrated(self, X: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities if available, else raw XGBoost probabilities."""
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        return self.model.predict_proba(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba_calibrated."""
        return self.predict_proba_calibrated(X)

    def save_model(self, output_path: str = "data/processed/xgboost_severity_model.joblib"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "calibration_method": self.calibration_method,
            "features": self.feature_names
        }, output_path)
        logger.info(f"Model saved to {output_path}")

    @staticmethod
    def load_model(model_path: str = "data/processed/xgboost_severity_model.joblib"):
        data = joblib.load(model_path)
        calib_method = data.get("calibration_method", "isotonic")
        clf = SeverityXGBoost(calibration_method=calib_method)
        clf.model = data["model"]
        clf.calibrated_model = data.get("calibrated_model", None)
        clf.calibration_method = calib_method
        clf.feature_names = data["features"]
        return clf


class OrdinalSeverityModel:
    """Two-stage cumulative binary ordinal classifier.

    Stage 1: predict P(severity >= Grievous) = P(class 1 or 2)
    Stage 2: predict P(severity == Fatal) = P(class 2)
    Derives 3-class probabilities from the two calibrated binary classifiers.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.clf_at_least_grievous = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.06,
            objective="binary:logistic", eval_metric="logloss",
            subsample=0.85, colsample_bytree=0.80,
            random_state=random_state, n_jobs=-1
        )
        self.clf_fatal = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.06,
            objective="binary:logistic", eval_metric="logloss",
            subsample=0.85, colsample_bytree=0.80,
            random_state=random_state, n_jobs=-1
        )
        self.feature_names = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """Fits both binary cumulative thresholds."""
        self.feature_names = feature_names or []
        y_grev = (y >= 1).astype(int)   # "at least Grievous"
        y_fatal = (y == 2).astype(int)  # exactly Fatal

        w_grev = compute_sample_weight("balanced", y_grev)
        w_fatal = compute_sample_weight("balanced", y_fatal)
        w_fatal[y_fatal == 1] *= 2.0    # Extra weight on fatal positives

        logger.info("Fitting Ordinal Stage 1: P(severity >= Grievous)...")
        self.clf_at_least_grievous.fit(X, y_grev, sample_weight=w_grev, verbose=False)
        logger.info("Fitting Ordinal Stage 2: P(severity == Fatal)...")
        self.clf_fatal.fit(X, y_fatal, sample_weight=w_fatal, verbose=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, 3) probabilities by combining ordinal thresholds."""
        p_grev_plus = self.clf_at_least_grievous.predict_proba(X)[:, 1]
        p_fatal = self.clf_fatal.predict_proba(X)[:, 1]

        # Enforce monotonicity: P(fatal) <= P(at least grievous)
        p_fatal = np.minimum(p_fatal, p_grev_plus)
        p_minor = 1.0 - p_grev_plus
        p_grievous = p_grev_plus - p_fatal

        # Clip and renormalize
        probs = np.column_stack([p_minor, p_grievous, p_fatal]).clip(0, 1)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str = "data/processed/ordinal_severity_model.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Ordinal model saved to {path}")

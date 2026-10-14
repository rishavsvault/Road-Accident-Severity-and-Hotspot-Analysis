"""SHAP (SHapley Additive exPlanations) for Global & Local model interpretability."""
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("shap_explainer")

class SeveritySHAPExplainer:
    """Computes TreeSHAP values for the multiclass XGBoost severity model."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        logger.info("Initializing TreeSHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(self, X: pd.DataFrame or np.ndarray):
        """Computes SHAP values across an entire evaluation dataset."""
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names].values
        else:
            X_arr = X
        shap_values = self.explainer.shap_values(X_arr)
        return shap_values

    def get_global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns mean absolute SHAP values for severe/fatal crashes (class 2)."""
        shap_values = self.explain_dataset(X)
        # In multiclass, shap_values is a list of arrays [class_0, class_1, class_2]
        # or a 3D array (samples, features, classes)
        if isinstance(shap_values, list):
            fatal_shap = shap_values[2]
        elif len(shap_values.shape) == 3:
            fatal_shap = shap_values[:, :, 2]
        else:
            fatal_shap = shap_values

        mean_abs = np.mean(np.abs(fatal_shap), axis=0)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "fatal_shap_importance": mean_abs
        }).sort_values(by="fatal_shap_importance", ascending=False).reset_index(drop=True)
        return df_imp

    def explain_single_instance(self, instance_series: pd.Series) -> List[Dict[str, Any]]:
        """Returns top risk-increasing and risk-reducing drivers for a specific crash event."""
        x_row = instance_series[self.feature_names].values.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x_row)

        if isinstance(shap_vals, list):
            fatal_shap = shap_vals[2][0]
        elif len(shap_vals.shape) == 3:
            fatal_shap = shap_vals[0, :, 2]
        else:
            fatal_shap = shap_vals[0]

        contributions = []
        for feat, val, s in zip(self.feature_names, x_row[0], fatal_shap):
            contributions.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(s),
                "impact": "Increases Risk" if s > 0 else "Decreases Risk"
            })

        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        return contributions

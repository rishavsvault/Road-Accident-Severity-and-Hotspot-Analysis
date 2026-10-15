import os
import sys
from pathlib import Path
import joblib
import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union

# Ensure repository root is on sys.path for direct CLI script execution
REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.logger import get_logger
from src.utils.feature_names import to_display_name, to_simple_name

logger = get_logger("shap_explainer")

class SeveritySHAPExplainer:
    """Computes TreeSHAP values for the multiclass XGBoost severity model."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        logger.info("Initializing TreeSHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_dataset(self, X: Union[pd.DataFrame, np.ndarray]):
        """Computes SHAP values across an entire evaluation dataset."""
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names].values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)

        shap_values = self.explainer.shap_values(X_arr)
        # Handle shap.Explanation objects in newer shap versions
        if hasattr(shap_values, "values"):
            shap_values = shap_values.values
        return shap_values

    def _extract_fatal_shap(self, shap_values: Any) -> np.ndarray:
        """Robustly extracts SHAP values for the Fatal class (class 2) across all SHAP formats."""
        if isinstance(shap_values, list):
            # List of arrays [class_0, class_1, class_2]
            idx = min(2, len(shap_values) - 1)
            return shap_values[idx]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # Shape (samples, features, classes) or (samples, classes, features)
                if shap_values.shape[2] == 3 or shap_values.shape[2] == len(self.model.classes_ if hasattr(self.model, "classes_") else [0, 1, 2]):
                    return shap_values[:, :, 2]
                elif shap_values.shape[1] == 3:
                    return shap_values[:, 2, :]
                else:
                    return shap_values[:, :, -1]
            elif shap_values.ndim == 2:
                return shap_values
        return np.asarray(shap_values)

    def get_global_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns mean absolute SHAP values for severe/fatal crashes (class 2)."""
        shap_values = self.explain_dataset(X)
        fatal_shap = self._extract_fatal_shap(shap_values)

        mean_abs = np.mean(np.abs(fatal_shap), axis=0)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "fatal_shap_importance": mean_abs
        }).sort_values(by="fatal_shap_importance", ascending=False).reset_index(drop=True)
        return df_imp

    def explain_single_instance(self, instance_series: pd.Series) -> List[Dict[str, Any]]:
        """Returns top risk-increasing and risk-reducing drivers for a specific crash event."""
        x_row = instance_series[self.feature_names].values.astype(float).reshape(1, -1)
        shap_vals = self.explainer.shap_values(x_row)
        if hasattr(shap_vals, "values"):
            shap_vals = shap_vals.values

        fatal_shap = self._extract_fatal_shap(shap_vals)
        if fatal_shap.ndim == 2:
            fatal_row = fatal_shap[0]
        else:
            fatal_row = fatal_shap

        contributions = []
        for feat, val, s in zip(self.feature_names, x_row[0], fatal_row):
            contributions.append({
                "feature": feat,
                "simple_name": to_simple_name(feat),
                "display_name": to_display_name(feat),
                "value": float(val),
                "shap_value": float(s),
                "impact": "Increases Risk" if s > 0 else "Decreases Risk"
            })

        contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
        return contributions

if __name__ == "__main__":
    model_path = "data/processed/xgboost_severity_model.joblib"
    features_path = "data/processed/crash_level_features.parquet"

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        print(f"[!] Artifacts not found. Please run 'python run_pipeline.py' first.")
    else:
        print(f"[*] Loading model from {model_path}...")
        data = joblib.load(model_path)
        clf_model = data["model"]
        feat_names = data["features"]

        print(f"[*] Loading feature dataset from {features_path}...")
        df_sample = pd.read_parquet(features_path).sample(min(200, 1000), random_state=42)

        explainer = SeveritySHAPExplainer(clf_model, feat_names)

        print("\n--- Top 10 Global SHAP Drivers for Fatal Crashes ---")
        global_imp = explainer.get_global_feature_importance(df_sample)
        global_imp_display = global_imp.head(10).copy()
        global_imp_display["simple_name"] = global_imp_display["feature"].map(to_simple_name)
        print(global_imp_display[["simple_name", "fatal_shap_importance"]].to_string(index=False))

        print("\n--- Local SHAP Waterfall Breakdown (Sample Crash #1) ---")
        first_instance = df_sample.iloc[0]
        local_drivers = explainer.explain_single_instance(first_instance)
        for d in local_drivers[:5]:
            sign = "+" if d["shap_value"] > 0 else ""
            print(f" - {d['simple_name']:<25} = {d['value']:<8.2f} | SHAP: {sign}{d['shap_value']:.4f} ({d['impact']})")

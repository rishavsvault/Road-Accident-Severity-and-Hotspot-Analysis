"""Optuna hyperparameter optimization for XGBoost severity model."""
import optuna
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.features.build_features import FEATURE_COLUMNS
from src.models.xgboost_trainer import SeverityXGBoost

logger = get_logger("tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optuna_tuning(df: pd.DataFrame, n_trials: int = 15) -> dict:
    """Finds optimal hyperparameters optimizing fatal class recall and macro-F1."""
    logger.info(f"Starting Optuna hyperparameter optimization ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.70, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
        }

        trainer = SeverityXGBoost(**params)
        metrics, _, _ = trainer.fit_with_spatial_cv(
            df=df,
            feature_cols=FEATURE_COLUMNS,
            n_splits=3
        )
        # Composite objective: Fatal Recall (primary) + Macro-F1 (secondary)
        score = 0.6 * metrics["fatal_recall"] + 0.4 * metrics["macro_f1"]
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optimal trial score: {study.best_value:.4f}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params

if __name__ == "__main__":
    df_feat = pd.read_parquet("data/processed/crash_level_features.parquet")
    run_optuna_tuning(df_feat)

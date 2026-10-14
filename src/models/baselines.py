"""Baseline models for severity benchmark comparisons."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger("baselines")

class BaselineModels:
    """Provides standard benchmark classifiers with balanced class weighting."""

    @staticmethod
    def get_logistic_regression(random_state: int = 42) -> LogisticRegression:
        logger.info("Initializing Multinomial Logistic Regression baseline...")
        return LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        )

    @staticmethod
    def get_random_forest(random_state: int = 42) -> RandomForestClassifier:
        logger.info("Initializing Random Forest baseline...")
        return RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        )

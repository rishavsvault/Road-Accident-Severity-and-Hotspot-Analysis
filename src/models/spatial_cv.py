"""Spatial Grouped Cross-Validation to eliminate spatial leakage."""
import numpy as np
import pandas as pd
from typing import Generator, Tuple
from sklearn.model_selection import GroupKFold
from src.utils.logger import get_logger

logger = get_logger("spatial_cv")

class SpatialGroupedCV:
    """Enforces spatial isolation between training and validation splits.

    Folds are grouped by planning zones or police station jurisdictions so segments
    from the same geographic corridor do not appear in both train and validation sets.
    """
    def __init__(self, n_splits: int = 5, group_col: str = "spatial_zone_id"):
        self.n_splits = n_splits
        self.group_col = group_col
        self.gkf = GroupKFold(n_splits=n_splits)

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        groups = df[self.group_col].values
        unique_groups = np.unique(groups)
        logger.info(f"Setting up Spatial Grouped CV with {len(unique_groups)} distinct geographic zones across {self.n_splits} folds.")
        return self.gkf.split(df, groups=groups)

"""Data loader for experiment datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ExperimentDataLoader:
    """Load and prepare data for prediction experiments.
    
    Handles:
    - Loading feature matrix from Parquet
    - Filtering by category, week range
    - Splitting into development and test sets
    - Extracting feature matrices and targets
    """
    
    # Default feature columns (L1 + L2, no positional)
    DEFAULT_FEATURES = [
        "log_return",
        "log_return_intraweek",
        "log_range",
        "log_volume",
        "log_avg_daily_volume",
        "intra_week_volatility",
        "log_return_ma4",
        "log_return_ma12",
        "log_volume_ma4",
        "log_volume_ma12",
        "momentum_4w",
        "momentum_12w",
        "volatility_ma4",
        "volatility_ma12",
        "log_volume_delta",
    ]
    
    TARGET_COL = "target_return"
    
    def __init__(
        self,
        feature_matrix_path: str,
        test_weeks: int = 2,
        train_weeks: Optional[int] = None,
        category_filter: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
    ):
        """Initialize data loader.
        
        Args:
            feature_matrix_path: Path to feature_matrix.parquet
            test_weeks: Number of weeks to hold out for final test
            train_weeks: Max weeks for train+val (None = use all available)
            category_filter: List of categories to include (e.g., ["target"])
            features: List of feature column names, or None for defaults
        """
        self.feature_matrix_path = Path(feature_matrix_path)
        self.test_weeks = test_weeks
        self.train_weeks = train_weeks
        self.category_filter = category_filter
        self.features = features or self.DEFAULT_FEATURES
        
        self._df: Optional[pd.DataFrame] = None
        self._dev_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._max_week_idx: int = 0
    
    def load(self) -> "ExperimentDataLoader":
        """Load and prepare the dataset.
        
        Returns:
            self (for chaining)
        """
        # Load parquet
        self._df = pd.read_parquet(self.feature_matrix_path)
        
        # Apply category filter
        if self.category_filter:
            self._df = self._df[self._df["category"].isin(self.category_filter)]
        
        # Determine week range
        self._max_week_idx = self._df["week_idx"].max()
        test_start_week = self._max_week_idx - self.test_weeks + 1
        
        # Split into development and test
        self._test_df = self._df[self._df["week_idx"] >= test_start_week].copy()
        
        # Apply train_weeks limit if specified
        if self.train_weeks is not None:
            train_start_week = test_start_week - self.train_weeks
            self._dev_df = self._df[
                (self._df["week_idx"] >= train_start_week) & 
                (self._df["week_idx"] < test_start_week)
            ].copy()
        else:
            self._dev_df = self._df[self._df["week_idx"] < test_start_week].copy()
        
        return self
    
    def get_development_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get development set (for cross-validation).
        
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        if self._dev_df is None:
            raise RuntimeError("Call load() first")
        
        X = self._dev_df[self.features].values
        y = self._dev_df[self.TARGET_COL].values
        return X, y
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get held-out test set.
        
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        if self._test_df is None:
            raise RuntimeError("Call load() first")
        
        X = self._test_df[self.features].values
        y = self._test_df[self.TARGET_COL].values
        return X, y
    
    def get_test_metadata(self) -> pd.DataFrame:
        """Get metadata for test set (symbol, week_start, etc.).
        
        Useful for analyzing predictions by symbol/time.
        """
        if self._test_df is None:
            raise RuntimeError("Call load() first")
        
        return self._test_df[["symbol", "name", "category", "week_start", "week_idx"]].copy()
    
    def get_development_metadata(self) -> pd.DataFrame:
        """Get metadata for development set (symbol, week_start, etc.).
        
        Useful for out-of-fold prediction analysis.
        """
        if self._dev_df is None:
            raise RuntimeError("Call load() first")
        
        return self._dev_df[["symbol", "name", "category", "week_start", "week_idx"]].copy()
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.features.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata for reporting."""
        if self._df is None:
            raise RuntimeError("Call load() first")
        
        return {
            "total_rows": len(self._df),
            "dev_rows": len(self._dev_df),
            "test_rows": len(self._test_df),
            "n_features": len(self.features),
            "n_symbols": self._df["symbol"].nunique(),
            "dev_symbols": self._dev_df["symbol"].nunique(),
            "test_symbols": self._test_df["symbol"].nunique(),
            "week_range": {
                "min": int(self._df["week_idx"].min()),
                "max": int(self._df["week_idx"].max()),
            },
            "dev_week_range": {
                "min": int(self._dev_df["week_idx"].min()),
                "max": int(self._dev_df["week_idx"].max()),
            },
            "test_week_range": {
                "min": int(self._test_df["week_idx"].min()),
                "max": int(self._test_df["week_idx"].max()),
            },
            "features": self.features,
            "category_filter": self.category_filter,
            "train_weeks_limit": self.train_weeks,
        }

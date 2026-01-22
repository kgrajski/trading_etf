"""Data normalization for features and targets."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class NormalizerParams:
    """Parameters for a fitted normalizer."""
    mean: float
    std: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std}


class DataNormalizer:
    """Z-score normalizer for features and target.
    
    Fits on development (train+val) data, applies to all data.
    Ensures no data leakage from test set.
    
    Usage:
        normalizer = DataNormalizer(normalize_features=True, normalize_target=True)
        normalizer.fit(X_train_val, y_train_val)
        
        X_norm, y_norm = normalizer.transform(X, y)
        y_pred_original = normalizer.inverse_transform_target(y_pred_norm)
    """
    
    def __init__(
        self,
        normalize_features: bool = True,
        normalize_target: bool = True,
    ):
        """Initialize normalizer.
        
        Args:
            normalize_features: Whether to z-score normalize features
            normalize_target: Whether to z-score normalize target
        """
        self.normalize_features = normalize_features
        self.normalize_target = normalize_target
        
        # Feature params (per-column)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        
        # Target params (single value)
        self._target_mean: Optional[float] = None
        self._target_std: Optional[float] = None
        
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataNormalizer":
        """Fit normalizer on development data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            
        Returns:
            self (for chaining)
        """
        if self.normalize_features:
            self._feature_means = np.nanmean(X, axis=0)
            self._feature_stds = np.nanstd(X, axis=0)
            # Prevent division by zero for constant features
            self._feature_stds[self._feature_stds == 0] = 1.0
        
        if self.normalize_target:
            self._target_mean = float(np.nanmean(y))
            self._target_std = float(np.nanstd(y))
            if self._target_std == 0:
                self._target_std = 1.0
        
        self._is_fitted = True
        return self
    
    def transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply normalization.
        
        Args:
            X: Feature matrix
            y: Optional target vector
            
        Returns:
            Tuple of (X_normalized, y_normalized or None)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        X_out = X.copy()
        y_out = y.copy() if y is not None else None
        
        if self.normalize_features:
            X_out = (X_out - self._feature_means) / self._feature_stds
        
        if self.normalize_target and y_out is not None:
            y_out = (y_out - self._target_mean) / self._target_std
        
        return X_out, y_out
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transform features only."""
        X_out, _ = self.transform(X, None)
        return X_out
    
    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """Transform target only."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        if not self.normalize_target:
            return y.copy()
        
        return (y - self._target_mean) / self._target_std
    
    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original scale.
        
        Args:
            y_normalized: Predictions in normalized space
            
        Returns:
            Predictions in original log-return scale
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")
        
        if not self.normalize_target:
            return y_normalized.copy()
        
        return y_normalized * self._target_std + self._target_mean
    
    def get_params(self) -> Dict[str, Any]:
        """Get normalization parameters for saving to config."""
        params = {
            "normalize_features": self.normalize_features,
            "normalize_target": self.normalize_target,
            "is_fitted": self._is_fitted,
        }
        
        if self._is_fitted:
            if self.normalize_features:
                params["feature_means"] = self._feature_means.tolist()
                params["feature_stds"] = self._feature_stds.tolist()
            
            if self.normalize_target:
                params["target_mean"] = self._target_mean
                params["target_std"] = self._target_std
        
        return params
    
    @property
    def target_params(self) -> Optional[NormalizerParams]:
        """Get target normalization parameters."""
        if self._is_fitted and self.normalize_target:
            return NormalizerParams(
                mean=self._target_mean,
                std=self._target_std,
            )
        return None

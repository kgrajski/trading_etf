"""Base model interface for all prediction models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all prediction models.
    
    All models must implement fit() and predict().
    Feature importance is optional (not all models support it).
    """
    
    def __init__(self, **kwargs):
        """Initialize model with optional hyperparameters."""
        self._model = None
        self._feature_names: Optional[List[str]] = None
        self._is_fitted = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass
    
    @property
    def requires_scaling(self) -> bool:
        """Whether this model requires feature scaling.
        
        Override in subclasses that need scaling (e.g., neural nets).
        Linear models with regularization benefit from scaling but sklearn
        handles this internally for Ridge/Lasso.
        """
        return False
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted
    
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "BaseModel":
        """Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: Optional list of feature names for importance
            
        Returns:
            self (for chaining)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dict mapping feature name to importance score, or None if
            not supported by this model type.
        """
        return None
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters.
        
        Returns:
            Dict of hyperparameter names and values.
        """
        if self._model is not None and hasattr(self._model, "get_params"):
            return self._model.get_params()
        return {}
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self._is_fitted})"

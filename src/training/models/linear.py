"""Linear regression models."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.training.models.base import BaseModel


class LinearModel(BaseModel):
    """Ordinary Least Squares linear regression.
    
    Baseline model. No regularization.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = LinearRegression(**kwargs)
    
    @property
    def name(self) -> str:
        return "Linear Regression"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "LinearModel":
        self._feature_names = feature_names
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted:
            return None
        
        coefs = np.abs(self._model.coef_)
        if self._feature_names is not None:
            return dict(zip(self._feature_names, coefs))
        return {f"feature_{i}": c for i, c in enumerate(coefs)}


class RidgeModel(BaseModel):
    """Ridge regression (L2 regularization).
    
    Handles collinearity, shrinks coefficients toward zero.
    """
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._model = Ridge(alpha=alpha, **kwargs)
    
    @property
    def name(self) -> str:
        return "Ridge Regression"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "RidgeModel":
        self._feature_names = feature_names
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted:
            return None
        
        coefs = np.abs(self._model.coef_)
        if self._feature_names is not None:
            return dict(zip(self._feature_names, coefs))
        return {f"feature_{i}": c for i, c in enumerate(coefs)}


class LassoModel(BaseModel):
    """Lasso regression (L1 regularization).
    
    Performs feature selection by driving some coefficients to exactly zero.
    """
    
    def __init__(self, alpha: float = 0.001, max_iter: int = 10000, **kwargs):
        super().__init__(**kwargs)
        # Lower alpha for Lasso since returns are small magnitude
        self._model = Lasso(alpha=alpha, max_iter=max_iter, **kwargs)
    
    @property
    def name(self) -> str:
        return "Lasso Regression"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "LassoModel":
        self._feature_names = feature_names
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted:
            return None
        
        coefs = np.abs(self._model.coef_)
        if self._feature_names is not None:
            return dict(zip(self._feature_names, coefs))
        return {f"feature_{i}": c for i, c in enumerate(coefs)}
    
    def get_selected_features(self) -> Optional[List[str]]:
        """Get features with non-zero coefficients."""
        if not self._is_fitted or self._feature_names is None:
            return None
        
        mask = self._model.coef_ != 0
        return [f for f, m in zip(self._feature_names, mask) if m]

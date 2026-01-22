"""Tree-based ensemble models."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.training.models.base import BaseModel

# XGBoost import with fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class RandomForestModel(BaseModel):
    """Random Forest regressor.
    
    Non-linear, handles interactions, provides feature importance.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
    
    @property
    def name(self) -> str:
        return "Random Forest"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "RandomForestModel":
        self._feature_names = feature_names
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted:
            return None
        
        importances = self._model.feature_importances_
        if self._feature_names is not None:
            return dict(zip(self._feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting regressor.
    
    State-of-the-art tree ensemble method.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
        verbosity: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )
        
        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=verbosity,
            **kwargs,
        )
    
    @property
    def name(self) -> str:
        return "XGBoost"
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "XGBoostModel":
        self._feature_names = feature_names
        self._model.fit(X, y)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if not self._is_fitted:
            return None
        
        importances = self._model.feature_importances_
        if self._feature_names is not None:
            return dict(zip(self._feature_names, importances))
        return {f"feature_{i}": imp for i, imp in enumerate(importances)}

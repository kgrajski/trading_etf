"""Classification models for ETF direction prediction."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.training.models.base import BaseModel

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class LogisticModel(BaseModel):
    """Logistic Regression classifier."""
    
    def __init__(self, **kwargs):
        self.model = LogisticRegression(max_iter=1000, **kwargs)
        self._feature_names: Optional[List[str]] = None
    
    @property
    def name(self) -> str:
        return "Logistic Regression"
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self._feature_names = feature_names
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._feature_names is None:
            return None
        coef = self.model.coef_[0]
        return dict(zip(self._feature_names, coef.tolist()))


class RandomForestClassifierModel(BaseModel):
    """Random Forest classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, n_jobs: int = -1, **kwargs):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=42,
            **kwargs
        )
        self._feature_names: Optional[List[str]] = None
    
    @property
    def name(self) -> str:
        return "Random Forest Classifier"
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self._feature_names = feature_names
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._feature_names is None:
            return None
        return dict(zip(self._feature_names, self.model.feature_importances_.tolist()))


class XGBoostClassifierModel(BaseModel):
    """XGBoost classifier."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost not available")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss',
            **kwargs
        )
        self._feature_names: Optional[List[str]] = None
    
    @property
    def name(self) -> str:
        return "XGBoost Classifier"
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        self._feature_names = feature_names
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self._feature_names is None:
            return None
        return dict(zip(self._feature_names, self.model.feature_importances_.tolist()))

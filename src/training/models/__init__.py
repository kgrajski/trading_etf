"""Model implementations for ETF return prediction."""

from src.training.models.base import BaseModel
from src.training.models.linear import LinearModel, RidgeModel, LassoModel
from src.training.models.tree import RandomForestModel, XGBoostModel
from src.training.models.classification import (
    LogisticModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
)

__all__ = [
    "BaseModel",
    # Regression models
    "LinearModel",
    "RidgeModel",
    "LassoModel",
    "RandomForestModel",
    "XGBoostModel",
    # Classification models
    "LogisticModel",
    "RandomForestClassifierModel",
    "XGBoostClassifierModel",
]

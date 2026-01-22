"""Model factory for creating prediction models by name."""

from typing import Any, Dict, List

from src.training.models.base import BaseModel
from src.training.models.linear import LassoModel, LinearModel, RidgeModel
from src.training.models.tree import RandomForestModel, XGBoostModel
from src.training.models.classification import (
    LogisticModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
)


# Registry of available regression models
REGRESSION_REGISTRY: Dict[str, type] = {
    "linear": LinearModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}

# Registry of available classification models
CLASSIFICATION_REGISTRY: Dict[str, type] = {
    "logistic": LogisticModel,
    "rf_classifier": RandomForestClassifierModel,
    "xgb_classifier": XGBoostClassifierModel,
}

# Combined registry
MODEL_REGISTRY: Dict[str, type] = {**REGRESSION_REGISTRY, **CLASSIFICATION_REGISTRY}


def create_model(model_type: str, **kwargs: Any) -> BaseModel:
    """Create a model instance by type name.
    
    Args:
        model_type: One of the registered model types
        **kwargs: Model-specific hyperparameters
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not registered
    """
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {available}"
        )
    
    return MODEL_REGISTRY[model_type](**kwargs)


def list_models() -> List[str]:
    """List available model types.
    
    Returns:
        List of registered model type names
    """
    return list(MODEL_REGISTRY.keys())


def get_default_models() -> List[str]:
    """Get the default set of regression models for baseline experiments.
    
    Returns:
        List of model types to use in baseline experiments
    """
    return ["linear", "ridge", "lasso", "random_forest", "xgboost"]


def get_default_classifiers() -> List[str]:
    """Get the default set of classification models.
    
    Returns:
        List of classifier types to use in classification experiments
    """
    return ["logistic", "rf_classifier", "xgb_classifier"]

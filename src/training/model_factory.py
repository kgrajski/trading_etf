"""Model factory for creating prediction models by name."""

from typing import Any, Dict, List

from src.training.models.base import BaseModel
from src.training.models.linear import LassoModel, LinearModel, RidgeModel
from src.training.models.tree import RandomForestModel, XGBoostModel


# Registry of available models
MODEL_REGISTRY: Dict[str, type] = {
    "linear": LinearModel,
    "ridge": RidgeModel,
    "lasso": LassoModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


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
    """Get the default set of models for baseline experiments.
    
    Returns:
        List of model types to use in baseline experiments
    """
    return ["linear", "ridge", "lasso", "random_forest", "xgboost"]

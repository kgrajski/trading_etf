"""Cross-validation with parallel fold execution."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from src.training.model_factory import create_model
from src.training.evaluator import Evaluator


@dataclass
class CVFoldResult:
    """Results from a single cross-validation fold."""
    fold: int
    metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]]
    val_indices: Optional[np.ndarray] = None
    val_predictions: Optional[np.ndarray] = None
    val_actuals: Optional[np.ndarray] = None


@dataclass
class CVResult:
    """Aggregated cross-validation results."""
    model_type: str
    model_name: str
    n_folds: int
    fold_results: List[CVFoldResult]
    
    @property
    def mean_metrics(self) -> Dict[str, float]:
        """Compute mean of each metric across folds."""
        if not self.fold_results:
            return {}
        
        metric_names = self.fold_results[0].metrics.keys()
        return {
            name: np.mean([fr.metrics[name] for fr in self.fold_results])
            for name in metric_names
        }
    
    @property
    def std_metrics(self) -> Dict[str, float]:
        """Compute std of each metric across folds."""
        if not self.fold_results:
            return {}
        
        metric_names = self.fold_results[0].metrics.keys()
        return {
            name: np.std([fr.metrics[name] for fr in self.fold_results])
            for name in metric_names
        }
    
    @property
    def mean_feature_importance(self) -> Optional[Dict[str, float]]:
        """Average feature importance across folds."""
        importances = [fr.feature_importance for fr in self.fold_results 
                       if fr.feature_importance is not None]
        if not importances:
            return None
        
        all_features = set()
        for imp in importances:
            all_features.update(imp.keys())
        
        return {
            f: np.mean([imp.get(f, 0) for imp in importances])
            for f in all_features
        }
    
    def get_oof_predictions(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get out-of-fold predictions consolidated across all folds.
        
        Returns:
            Tuple of (indices, predictions, actuals) or None if not collected
        """
        if not self.fold_results or self.fold_results[0].val_indices is None:
            return None
        
        all_indices = []
        all_preds = []
        all_actuals = []
        
        for fr in self.fold_results:
            if fr.val_indices is not None:
                all_indices.extend(fr.val_indices)
                all_preds.extend(fr.val_predictions)
                all_actuals.extend(fr.val_actuals)
        
        return (
            np.array(all_indices),
            np.array(all_preds),
            np.array(all_actuals),
        )


def _run_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    model_params: Dict[str, Any],
    feature_names: List[str],
    collect_predictions: bool = True,
) -> CVFoldResult:
    """Run a single CV fold (designed to be called in parallel).
    
    Args:
        fold_idx: Fold number (0-indexed)
        train_idx: Indices for training data
        val_idx: Indices for validation data
        X: Full feature matrix
        y: Full target vector
        model_type: Type of model to create
        model_params: Model hyperparameters
        feature_names: List of feature names
        collect_predictions: Whether to store predictions for out-of-fold analysis
        
    Returns:
        CVFoldResult with metrics, feature importance, and optionally predictions
    """
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Create and train model
    model = create_model(model_type, **model_params)
    model.fit(X_train, y_train, feature_names=feature_names)
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    evaluator = Evaluator(y_val, y_pred)
    metrics = evaluator.compute_metrics()
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    return CVFoldResult(
        fold=fold_idx,
        metrics=metrics,
        feature_importance=feature_importance,
        val_indices=val_idx if collect_predictions else None,
        val_predictions=y_pred if collect_predictions else None,
        val_actuals=y_val if collect_predictions else None,
    )


class CrossValidator:
    """K-fold cross-validation with parallel execution.
    
    Performs random shuffled K-fold CV on the development set.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """Initialize cross-validator.
        
        Args:
            n_folds: Number of folds (K)
            n_jobs: Number of parallel jobs (-1 = all CPUs)
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
        model_params: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> CVResult:
        """Run cross-validation for a single model type.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            model_type: Type of model to validate
            model_params: Optional hyperparameters
            feature_names: Optional feature names for importance
            
        Returns:
            CVResult with aggregated metrics
        """
        model_params = model_params or {}
        feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        
        # Create K-fold splitter
        kfold = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        
        # Run folds in parallel
        fold_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_single_fold)(
                fold_idx=i,
                train_idx=train_idx,
                val_idx=val_idx,
                X=X,
                y=y,
                model_type=model_type,
                model_params=model_params,
                feature_names=feature_names,
            )
            for i, (train_idx, val_idx) in enumerate(kfold.split(X))
        )
        
        # Get model name
        temp_model = create_model(model_type)
        model_name = temp_model.name
        
        return CVResult(
            model_type=model_type,
            model_name=model_name,
            n_folds=self.n_folds,
            fold_results=fold_results,
        )
    
    def validate_multiple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_types: List[str],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, CVResult]:
        """Run cross-validation for multiple model types.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_types: List of model types to validate
            feature_names: Optional feature names
            
        Returns:
            Dict mapping model_type to CVResult
        """
        results = {}
        for model_type in model_types:
            results[model_type] = self.validate(
                X, y, model_type, feature_names=feature_names
            )
        return results

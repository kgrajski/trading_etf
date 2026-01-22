"""Evaluation metrics for prediction models."""

from typing import Dict, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Evaluator:
    """Compute evaluation metrics for regression predictions.
    
    Metrics:
    - R² (coefficient of determination)
    - MAE (mean absolute error)
    - RMSE (root mean squared error)
    - Directional Accuracy (% correct sign predictions)
    - IC (Information Coefficient = Spearman rank correlation)
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Initialize evaluator with actual and predicted values.
        
        Args:
            y_true: Actual target values
            y_pred: Predicted values
        """
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(self.y_true)}, y_pred={len(self.y_pred)}"
            )
    
    def r2_score(self) -> float:
        """Coefficient of determination (R²).
        
        Proportion of variance in y explained by predictions.
        Range: (-inf, 1], higher is better. 1 = perfect, 0 = mean baseline.
        """
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def mae(self) -> float:
        """Mean Absolute Error.
        
        Average absolute prediction error. In same units as target.
        """
        return np.mean(np.abs(self.y_true - self.y_pred))
    
    def rmse(self) -> float:
        """Root Mean Squared Error.
        
        Penalizes large errors more than MAE.
        """
        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
    
    def directional_accuracy(self) -> float:
        """Percentage of predictions with correct sign.
        
        Trading-relevant: did we predict up/down correctly?
        Range: [0, 1], 0.5 = random guessing.
        """
        correct = (np.sign(self.y_true) == np.sign(self.y_pred))
        return np.mean(correct)
    
    def information_coefficient(self) -> float:
        """Information Coefficient (IC).
        
        Spearman rank correlation between predictions and actuals.
        Standard quant metric. Range: [-1, 1], 0 = no correlation.
        """
        if len(self.y_true) < 3:
            return 0.0
        
        corr, _ = stats.spearmanr(self.y_true, self.y_pred)
        return corr if not np.isnan(corr) else 0.0
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dict with metric names and values
        """
        return {
            "r2": self.r2_score(),
            "mae": self.mae(),
            "rmse": self.rmse(),
            "directional_accuracy": self.directional_accuracy(),
            "ic": self.information_coefficient(),
        }
    
    def summary_string(self) -> str:
        """Get a formatted summary string of metrics."""
        metrics = self.compute_metrics()
        return (
            f"R²={metrics['r2']:.4f}, "
            f"MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, "
            f"Dir.Acc={metrics['directional_accuracy']:.2%}, "
            f"IC={metrics['ic']:.4f}"
        )


class ClassificationEvaluator:
    """Compute evaluation metrics for classification predictions.
    
    Metrics:
    - Accuracy (% correct predictions)
    - Precision (true positives / predicted positives)
    - Recall (true positives / actual positives)
    - F1 Score (harmonic mean of precision and recall)
    - AUC-ROC (area under ROC curve, if probabilities provided)
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ):
        """Initialize evaluator with actual and predicted values.
        
        Args:
            y_true: Actual binary labels (0/1)
            y_pred: Predicted binary labels (0/1)
            y_prob: Optional probability of positive class (for AUC)
        """
        self.y_true = np.asarray(y_true).flatten()
        self.y_pred = np.asarray(y_pred).flatten()
        self.y_prob = np.asarray(y_prob).flatten() if y_prob is not None else None
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(
                f"Length mismatch: y_true={len(self.y_true)}, y_pred={len(self.y_pred)}"
            )
    
    def accuracy(self) -> float:
        """Accuracy: proportion of correct predictions."""
        return accuracy_score(self.y_true, self.y_pred)
    
    def precision(self) -> float:
        """Precision: true positives / predicted positives.
        
        High precision = few false positives.
        """
        return precision_score(self.y_true, self.y_pred, zero_division=0)
    
    def recall(self) -> float:
        """Recall: true positives / actual positives.
        
        High recall = few false negatives.
        """
        return recall_score(self.y_true, self.y_pred, zero_division=0)
    
    def f1(self) -> float:
        """F1 Score: harmonic mean of precision and recall."""
        return f1_score(self.y_true, self.y_pred, zero_division=0)
    
    def auc_roc(self) -> float:
        """Area Under ROC Curve.
        
        Requires probability predictions. Returns 0.5 if not available.
        Range: [0, 1], 0.5 = random guessing, 1 = perfect.
        """
        if self.y_prob is None:
            return 0.5
        
        # Check if we have both classes
        if len(np.unique(self.y_true)) < 2:
            return 0.5
        
        try:
            return roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            return 0.5
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dict with metric names and values
        """
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "auc_roc": self.auc_roc(),
        }
    
    def summary_string(self) -> str:
        """Get a formatted summary string of metrics."""
        metrics = self.compute_metrics()
        return (
            f"Acc={metrics['accuracy']:.2%}, "
            f"Prec={metrics['precision']:.2%}, "
            f"Rec={metrics['recall']:.2%}, "
            f"F1={metrics['f1']:.4f}, "
            f"AUC={metrics['auc_roc']:.4f}"
        )

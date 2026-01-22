#!/usr/bin/env python3
"""
06b-run-classification-experiment.py

Quick classification experiment: predict direction (UP/DOWN) instead of magnitude.

This script:
1. Loads the feature matrix
2. Converts target to binary (1 if alpha > threshold, else 0)
3. Trains classification models
4. Compares to regression baseline
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.training.data_loader import ExperimentDataLoader
from src.training.model_factory import create_model, get_default_classifiers
from src.training.evaluator import ClassificationEvaluator
from src.training.normalizer import DataNormalizer
from src.workflow.workflow_utils import print_summary, setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp006_classification"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME

# Data configuration
FEATURE_MATRIX_PATH = "data/processed/iex/feature_matrix.parquet"
TEST_WEEKS = 2
TRAIN_WEEKS = 104
CATEGORY_FILTER = ["target"]

# Classification threshold: 1 if alpha > THRESHOLD, else 0
# Try multiple thresholds
THRESHOLDS = [0.0, 0.005, 0.01]  # 0%, 0.5%, 1%

# Alpha (market-neutral) configuration
USE_ALPHA_TARGET = True

# Normalization
NORMALIZE_FEATURES = True

# Models
CLASSIFIER_TYPES = get_default_classifiers()


# =============================================================================
# Main
# =============================================================================

@workflow_script("06b-run-classification-experiment")
def main() -> None:
    """Run the classification experiment."""
    
    # Create output directories
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    
    print("=" * 80)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading data...")
    
    loader = ExperimentDataLoader(
        feature_matrix_path=FEATURE_MATRIX_PATH,
        test_weeks=TEST_WEEKS,
        train_weeks=TRAIN_WEEKS,
        category_filter=CATEGORY_FILTER,
    )
    loader.load()
    
    X_train_val, y_train_val = loader.get_development_data()
    X_test, y_test = loader.get_test_data()
    dev_metadata = loader.get_development_metadata()
    test_metadata = loader.get_test_metadata()
    feature_names = loader.get_feature_names()
    
    print(f"  Train+Validation set: {X_train_val.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print()
    
    # Keep original targets for evaluation
    y_train_val_original = y_train_val.copy()
    y_test_original = y_test.copy()
    
    # =========================================================================
    # Step 2: Convert to Alpha
    # =========================================================================
    if USE_ALPHA_TARGET:
        print("Step 2: Converting to alpha (market-neutral)...")
        
        dev_weeks = dev_metadata["week_idx"].values
        test_weeks_idx = test_metadata["week_idx"].values
        
        all_weeks = np.concatenate([dev_weeks, test_weeks_idx])
        all_returns = np.concatenate([y_train_val, y_test])
        
        weekly_mean_lookup = {}
        for week in np.unique(all_weeks):
            mask = all_weeks == week
            weekly_mean_lookup[week] = all_returns[mask].mean()
        
        weekly_means_dev = np.array([weekly_mean_lookup[w] for w in dev_weeks])
        weekly_means_test = np.array([weekly_mean_lookup[w] for w in test_weeks_idx])
        
        y_train_val_alpha = y_train_val - weekly_means_dev
        y_test_alpha = y_test - weekly_means_test
        
        print(f"  Alpha (dev) range: [{y_train_val_alpha.min():.4f}, {y_train_val_alpha.max():.4f}]")
        print()
    else:
        y_train_val_alpha = y_train_val
        y_test_alpha = y_test
    
    # =========================================================================
    # Step 3: Normalize Features
    # =========================================================================
    normalizer = DataNormalizer(normalize_features=NORMALIZE_FEATURES, normalize_target=False)
    normalizer.fit(X_train_val, None)
    X_train_val_norm, _ = normalizer.transform(X_train_val, None)
    X_test_norm, _ = normalizer.transform(X_test, None)
    
    print("Step 3: Features normalized")
    print()
    
    # =========================================================================
    # Step 4: Run Classification for Each Threshold
    # =========================================================================
    all_results = {}
    
    for threshold in THRESHOLDS:
        print("=" * 60)
        print(f"THRESHOLD: {threshold:.1%}")
        print("=" * 60)
        
        # Convert to binary: 1 if alpha > threshold, else 0
        y_train_val_binary = (y_train_val_alpha > threshold).astype(int)
        y_test_binary = (y_test_alpha > threshold).astype(int)
        
        # Class balance
        train_pos_rate = y_train_val_binary.mean()
        test_pos_rate = y_test_binary.mean()
        print(f"  Train class balance: {train_pos_rate:.1%} positive")
        print(f"  Test class balance: {test_pos_rate:.1%} positive")
        print()
        
        threshold_results = {}
        
        for model_type in CLASSIFIER_TYPES:
            print(f"  Training {model_type}...", end=" ", flush=True)
            
            try:
                model = create_model(model_type)
                model.fit(X_train_val_norm, y_train_val_binary, feature_names=feature_names)
                
                # Predictions
                y_pred = model.predict(X_test_norm)
                y_prob = model.predict_proba(X_test_norm)
                
                # Evaluate
                evaluator = ClassificationEvaluator(y_test_binary, y_pred, y_prob)
                metrics = evaluator.compute_metrics()
                
                threshold_results[model_type] = {
                    "model_name": model.name,
                    "metrics": metrics,
                    "feature_importance": model.get_feature_importance(),
                }
                
                print(evaluator.summary_string())
                
            except Exception as e:
                print(f"FAILED: {e}")
                threshold_results[model_type] = {"error": str(e)}
        
        all_results[f"threshold_{threshold}"] = {
            "threshold": threshold,
            "train_pos_rate": train_pos_rate,
            "test_pos_rate": test_pos_rate,
            "models": threshold_results,
        }
        print()
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("Step 5: Saving results...")
    
    # Results JSON
    results_path = EXPERIMENT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"  Results: {results_path}")
    
    # Config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "thresholds": THRESHOLDS,
        "classifiers": CLASSIFIER_TYPES,
        "test_weeks": TEST_WEEKS,
        "train_weeks": TRAIN_WEEKS,
        "use_alpha": USE_ALPHA_TARGET,
        "normalize_features": NORMALIZE_FEATURES,
    }
    config_path = EXPERIMENT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    
    # Generate comparison report
    report = generate_report(all_results, config)
    report_path = EXPERIMENT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    # Find best configuration
    best_acc = 0
    best_config = ""
    for thresh_key, thresh_data in all_results.items():
        for model_type, model_data in thresh_data["models"].items():
            if "metrics" in model_data:
                acc = model_data["metrics"]["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_config = f"{model_data['model_name']} @ {thresh_data['threshold']:.1%}"
    
    print_summary(
        experiment=EXPERIMENT_NAME,
        thresholds_tested=len(THRESHOLDS),
        classifiers_tested=len(CLASSIFIER_TYPES),
        best_accuracy=f"{best_acc:.2%}",
        best_configuration=best_config,
        output_directory=str(EXPERIMENT_DIR),
    )


def generate_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append(f"# Classification Experiment Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Target:** Alpha (market-neutral) > threshold → 1, else → 0")
    lines.append(f"- **Thresholds tested:** {config['thresholds']}")
    lines.append(f"- **Classifiers:** {config['classifiers']}")
    lines.append(f"- **Test weeks:** {config['test_weeks']}")
    lines.append("")
    
    # Results by threshold
    for thresh_key, thresh_data in results.items():
        threshold = thresh_data["threshold"]
        lines.append(f"## Threshold: {threshold:.1%}")
        lines.append("")
        lines.append(f"- Train positive rate: {thresh_data['train_pos_rate']:.1%}")
        lines.append(f"- Test positive rate: {thresh_data['test_pos_rate']:.1%}")
        lines.append("")
        
        lines.append("| Model | Accuracy | Precision | Recall | F1 | AUC |")
        lines.append("|-------|----------|-----------|--------|-----|-----|")
        
        for model_type, model_data in thresh_data["models"].items():
            if "metrics" in model_data:
                m = model_data["metrics"]
                lines.append(
                    f"| {model_data['model_name']} | "
                    f"{m['accuracy']:.2%} | "
                    f"{m['precision']:.2%} | "
                    f"{m['recall']:.2%} | "
                    f"{m['f1']:.4f} | "
                    f"{m['auc_roc']:.4f} |"
                )
            else:
                lines.append(f"| {model_type} | ERROR | - | - | - | - |")
        
        lines.append("")
    
    # Comparison with regression baseline
    lines.append("## Comparison with Regression (exp004_alpha_norm)")
    lines.append("")
    lines.append("| Approach | Directional Accuracy |")
    lines.append("|----------|---------------------|")
    lines.append("| Regression (Lasso) | 76.0% |")
    
    # Best classification result
    best_acc = 0
    best_name = ""
    for thresh_data in results.values():
        for model_data in thresh_data["models"].values():
            if "metrics" in model_data:
                if model_data["metrics"]["accuracy"] > best_acc:
                    best_acc = model_data["metrics"]["accuracy"]
                    best_name = model_data["model_name"]
    
    lines.append(f"| Classification ({best_name}) | {best_acc:.1%} |")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `06b-run-classification-experiment.py`*")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()

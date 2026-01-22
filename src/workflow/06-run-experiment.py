#!/usr/bin/env python3
"""
06-run-experiment.py

Run baseline prediction experiment with cross-validation.

This script:
1. Loads the feature matrix
2. Splits into development (CV) and test sets
3. Runs 5-fold cross-validation for multiple models
4. Trains final models on full development set
5. Evaluates on held-out test set
6. Generates visualizations and reports
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.training.data_loader import ExperimentDataLoader
from src.training.model_factory import create_model, get_default_models
from src.training.cross_validator import CrossValidator
from src.training.evaluator import Evaluator
from src.training.visualizer import ExperimentVisualizer
from src.training.prediction_analyzer import PredictionAnalyzer
from src.training.normalizer import DataNormalizer
from src.workflow.workflow_utils import print_summary, setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp005_two_stage"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME

# Data configuration
FEATURE_MATRIX_PATH = "data/processed/iex/feature_matrix.parquet"
TEST_WEEKS = 2  # Hold out last 2 weeks (in 2026)
TRAIN_WEEKS = 104  # Limit train+val to 2 years
CATEGORY_FILTER = ["target"]  # Exclude macro symbols for alpha model

# Two-stage modeling configuration
USE_TWO_STAGE = True  # Stage 1: predict market (beta), Stage 2: predict alpha
MARKET_MODEL_TYPE = "ridge"  # Simple model for market prediction

# Alpha (market-neutral) configuration
USE_ALPHA_TARGET = True  # Convert target to market-neutral (subtract weekly mean)

# Normalization configuration
NORMALIZE_FEATURES = True  # Z-score normalize features
NORMALIZE_TARGET = True    # Z-score normalize target (after alpha if enabled)

# Cross-validation configuration
N_FOLDS = 5
N_JOBS = -1  # Use all CPUs

# Models to evaluate
MODEL_TYPES = get_default_models()  # linear, ridge, lasso, random_forest, xgboost


# =============================================================================
# Main
# =============================================================================

@workflow_script("06-run-experiment")
def main() -> None:
    """Run the baseline experiment."""
    
    # Create output directories
    plots_dir = EXPERIMENT_DIR / "plots"
    predictions_dir = EXPERIMENT_DIR / "predictions"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
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
    
    X_train_val, y_train_val = loader.get_development_data()  # For CV (train + validation)
    X_test, y_test = loader.get_test_data()  # Held out completely
    dev_metadata = loader.get_development_metadata()
    test_metadata = loader.get_test_metadata()
    feature_names = loader.get_feature_names()
    data_metadata = loader.get_metadata()
    
    print(f"  Train+Validation set: {X_train_val.shape[0]:,} samples, {X_train_val.shape[1]} features")
    print(f"  Test set (held out): {X_test.shape[0]:,} samples")
    print(f"  Features: {feature_names}")
    print()
    
    # Keep original data for final output (predictions in original scale)
    y_train_val_original = y_train_val.copy()
    y_test_original = y_test.copy()
    
    # =========================================================================
    # Step 1b: Two-Stage Market Model (if enabled)
    # =========================================================================
    market_model_results = None
    weekly_means_dev = None
    weekly_means_test = None
    weekly_means_test_predicted = None
    
    if USE_TWO_STAGE and USE_ALPHA_TARGET:
        print("Step 1b: Building Market Model (Stage 1)...")
        
        # Get week indices
        dev_weeks = dev_metadata["week_idx"].values
        test_week_indices = test_metadata["week_idx"].values
        
        # Compute ACTUAL weekly means (for training and dev evaluation)
        all_weeks = np.concatenate([dev_weeks, test_week_indices])
        all_returns = np.concatenate([y_train_val, y_test])
        
        weekly_mean_lookup = {}
        for week in np.unique(all_weeks):
            mask = all_weeks == week
            weekly_mean_lookup[week] = all_returns[mask].mean()
        
        # Load ALL symbols for market features (aggregate across all symbols per week)
        all_loader = ExperimentDataLoader(
            feature_matrix_path=FEATURE_MATRIX_PATH,
            test_weeks=TEST_WEEKS,
            train_weeks=TRAIN_WEEKS,
            category_filter=None,  # ALL symbols (target + macro)
        )
        all_loader.load()
        X_all_dev, _ = all_loader.get_development_data()
        X_all_test, _ = all_loader.get_test_data()
        all_dev_meta = all_loader.get_development_metadata()
        all_test_meta = all_loader.get_test_metadata()
        all_features = all_loader.get_feature_names()
        
        # Aggregate ALL symbols' features to one row per week (mean across all symbols)
        all_dev_df = pd.DataFrame(X_all_dev, columns=all_features)
        all_dev_df["week_idx"] = all_dev_meta["week_idx"].values
        X_market_dev = all_dev_df.groupby("week_idx").mean()
        
        all_test_df = pd.DataFrame(X_all_test, columns=all_features)
        all_test_df["week_idx"] = all_test_meta["week_idx"].values
        X_market_test = all_test_df.groupby("week_idx").mean()
        
        # Market target: weekly cross-sectional mean return (from target ETFs)
        y_market_dev = pd.Series({w: weekly_mean_lookup[w] for w in X_market_dev.index})
        y_market_test_actual = pd.Series({w: weekly_mean_lookup[w] for w in X_market_test.index})
        
        print(f"  Market model: {X_market_dev.shape[0]} weeks dev, {X_market_test.shape[0]} weeks test")
        print(f"  Market features (aggregated): {len(all_features)}")
        
        # Normalize market features
        market_normalizer = DataNormalizer(normalize_features=True, normalize_target=True)
        market_normalizer.fit(X_market_dev.values, y_market_dev.values)
        X_market_dev_norm, y_market_dev_norm = market_normalizer.transform(
            X_market_dev.values, y_market_dev.values
        )
        X_market_test_norm, _ = market_normalizer.transform(X_market_test.values, None)
        
        # Train market model
        market_model = create_model(MARKET_MODEL_TYPE)
        market_model.fit(X_market_dev_norm, y_market_dev_norm, feature_names=all_features)
        
        # Predict market return (beta) for test weeks
        y_market_pred_norm = market_model.predict(X_market_test_norm)
        y_market_pred = market_normalizer.inverse_transform_target(y_market_pred_norm)
        
        # Also get dev predictions for diagnostics
        y_market_dev_pred_norm = market_model.predict(X_market_dev_norm)
        y_market_dev_pred = market_normalizer.inverse_transform_target(y_market_dev_pred_norm)
        
        # Evaluate market model
        market_evaluator = Evaluator(y_market_test_actual.values, y_market_pred)
        market_metrics = market_evaluator.compute_metrics()
        
        print(f"  Market model test: {market_evaluator.summary_string()}")
        
        # Store results for report
        market_model_results = {
            "model_type": MARKET_MODEL_TYPE,
            "n_weeks_dev": len(y_market_dev),
            "n_weeks_test": len(y_market_test_actual),
            "n_features": len(all_features),
            "test_metrics": market_metrics,
            "feature_importance": market_model.get_feature_importance(),
            "test_actual": y_market_test_actual.to_dict(),
            "test_predicted": dict(zip(X_market_test.index.tolist(), y_market_pred.tolist())),
        }
        
        # Build weekly mean arrays
        # For dev: use ACTUAL means (we know them during training)
        weekly_means_dev = np.array([weekly_mean_lookup[w] for w in dev_weeks])
        
        # For test: use PREDICTED means (this is the key difference!)
        test_week_to_pred = dict(zip(X_market_test.index.tolist(), y_market_pred.tolist()))
        weekly_means_test_predicted = np.array([test_week_to_pred[w] for w in test_week_indices])
        
        # Also keep actual for comparison
        weekly_means_test = np.array([weekly_mean_lookup[w] for w in test_week_indices])
        
        print(f"  Predicted beta for test weeks: {y_market_pred}")
        print(f"  Actual beta for test weeks: {y_market_test_actual.values}")
        print()
        
        # Transform to alpha (using ACTUAL means for dev, we'll use predicted for test later)
        y_train_val = y_train_val - weekly_means_dev
        y_test = y_test - weekly_means_test  # Still use actual for alpha training target
        
        print(f"  Alpha (dev) range: [{y_train_val.min():.6f}, {y_train_val.max():.6f}]")
        print(f"  Alpha (test) range: [{y_test.min():.6f}, {y_test.max():.6f}]")
        print()
        
    elif USE_ALPHA_TARGET:
        # Original single-stage alpha (uses actual weekly means)
        print("Step 1b: Converting target to market-neutral (alpha)...")
        
        dev_weeks = dev_metadata["week_idx"].values
        test_week_indices = test_metadata["week_idx"].values
        
        all_weeks = np.concatenate([dev_weeks, test_week_indices])
        all_returns = np.concatenate([y_train_val, y_test])
        
        weekly_mean_lookup = {}
        for week in np.unique(all_weeks):
            mask = all_weeks == week
            weekly_mean_lookup[week] = all_returns[mask].mean()
        
        weekly_means_dev = np.array([weekly_mean_lookup[w] for w in dev_weeks])
        weekly_means_test = np.array([weekly_mean_lookup[w] for w in test_week_indices])
        
        y_train_val = y_train_val - weekly_means_dev
        y_test = y_test - weekly_means_test
        
        print(f"  Weekly mean range: [{min(weekly_mean_lookup.values()):.6f}, {max(weekly_mean_lookup.values()):.6f}]")
        print(f"  Alpha (dev) range: [{y_train_val.min():.6f}, {y_train_val.max():.6f}]")
        print(f"  Alpha (test) range: [{y_test.min():.6f}, {y_test.max():.6f}]")
        print()
    
    # =========================================================================
    # Step 1c: Normalize Features and Target (if enabled)
    # =========================================================================
    normalizer = DataNormalizer(
        normalize_features=NORMALIZE_FEATURES,
        normalize_target=NORMALIZE_TARGET,
    )
    
    if NORMALIZE_FEATURES or NORMALIZE_TARGET:
        print("Step 1b: Normalizing data...")
        normalizer.fit(X_train_val, y_train_val)
        
        X_train_val, y_train_val = normalizer.transform(X_train_val, y_train_val)
        X_test, _ = normalizer.transform(X_test, None)  # Don't transform y_test yet
        
        if NORMALIZE_FEATURES:
            print(f"  Features: Z-score normalized")
        if NORMALIZE_TARGET:
            params = normalizer.target_params
            print(f"  Target: Z-score normalized (mean={params.mean:.6f}, std={params.std:.6f})")
        print()
    
    # =========================================================================
    # Step 2: Cross-Validation (Train/Validation splits)
    # =========================================================================
    print("Step 2: Running cross-validation (train/validation splits)...")
    print(f"  Models: {MODEL_TYPES}")
    print(f"  Folds: {N_FOLDS} (each fold: 80% train, 20% validation)")
    print(f"  Parallel jobs: {N_JOBS if N_JOBS > 0 else 'all CPUs'}")
    if NORMALIZE_TARGET:
        print(f"  Note: CV metrics are in NORMALIZED space")
    print()
    
    cv = CrossValidator(n_folds=N_FOLDS, n_jobs=N_JOBS)
    cv_results = {}
    
    for model_type in MODEL_TYPES:
        print(f"  Running CV for {model_type}...", end=" ", flush=True)
        result = cv.validate(X_train_val, y_train_val, model_type, feature_names=feature_names)
        cv_results[model_type] = result
        
        mean_r2 = result.mean_metrics["r2"]
        std_r2 = result.std_metrics["r2"]
        mean_ic = result.mean_metrics["ic"]
        print(f"R²={mean_r2:.4f}±{std_r2:.4f}, IC={mean_ic:.4f}")
    
    print()
    
    # =========================================================================
    # Step 3: Train Final Models on Full Train+Val, Evaluate on Held-Out Test
    # =========================================================================
    print("Step 3: Training final models (on full train+val) and evaluating on held-out test...")
    if NORMALIZE_TARGET:
        print("  Note: Predictions inverse-transformed to original log-return scale for evaluation")
    
    test_results = {}
    predictions = {}  # Store predictions in ORIGINAL scale (log-return)
    feature_importances = {}
    
    for model_type in MODEL_TYPES:
        print(f"  {model_type}...", end=" ", flush=True)
        
        # Train on full train+validation set (normalized if enabled)
        model = create_model(model_type)
        model.fit(X_train_val, y_train_val, feature_names=feature_names)
        
        # Predict on test set (normalized features)
        y_pred_norm = model.predict(X_test)
        
        # Inverse transform predictions to original scale
        # Step 1: Inverse Z-score normalization
        y_pred_alpha = normalizer.inverse_transform_target(y_pred_norm)
        
        # Step 2: Inverse alpha transform (add back weekly mean)
        if USE_ALPHA_TARGET:
            if USE_TWO_STAGE and weekly_means_test_predicted is not None:
                # Two-stage: use PREDICTED market return (beta)
                y_pred = y_pred_alpha + weekly_means_test_predicted
            elif weekly_means_test is not None:
                # Single-stage: use ACTUAL market return
                y_pred = y_pred_alpha + weekly_means_test
            else:
                y_pred = y_pred_alpha
        else:
            y_pred = y_pred_alpha
        
        predictions[model_type] = y_pred
        
        # Evaluate in ORIGINAL scale (log-return)
        evaluator = Evaluator(y_test_original, y_pred)
        metrics = evaluator.compute_metrics()
        test_results[model_type] = {
            "model_name": model.name,
            "metrics": metrics,
        }
        
        # Get feature importance
        importance = model.get_feature_importance()
        if importance:
            feature_importances[model_type] = importance
        
        print(evaluator.summary_string())
    
    print()
    
    # =========================================================================
    # Step 4: Generate Visualizations
    # =========================================================================
    print("Step 4: Generating visualizations...")
    
    visualizer = ExperimentVisualizer(plots_dir)
    
    # CV comparison plot
    cv_plot = visualizer.plot_cv_comparison(cv_results, "cv_comparison")
    print(f"  CV comparison: {cv_plot}")
    
    # Test scatter plots for each model
    for model_type, y_pred in predictions.items():
        model_name = test_results[model_type]["model_name"]
        scatter_plot = visualizer.plot_scatter_predictions(
            y_test, y_pred, model_name, test_metadata,
            filename=f"test_scatter_{model_type}",
        )
        print(f"  Scatter ({model_type}): {scatter_plot}")
    
    # Feature importance plots
    for model_type, importance in feature_importances.items():
        model_name = test_results[model_type]["model_name"]
        imp_plot = visualizer.plot_feature_importance(
            importance, model_name,
            filename=f"feature_importance_{model_type}",
        )
        print(f"  Feature importance ({model_type}): {imp_plot}")
    
    # Inspector
    inspector_path = visualizer.create_inspector()
    print(f"  Inspector: {inspector_path}")
    print()
    
    # =========================================================================
    # Step 5: Save Results
    # =========================================================================
    print("Step 5: Saving results...")
    
    # CV results JSON
    cv_json = {
        model_type: {
            "model_name": result.model_name,
            "n_folds": result.n_folds,
            "mean_metrics": result.mean_metrics,
            "std_metrics": result.std_metrics,
        }
        for model_type, result in cv_results.items()
    }
    cv_path = EXPERIMENT_DIR / "cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_json, f, indent=2)
    print(f"  CV results: {cv_path}")
    
    # Test results JSON
    test_json = {
        model_type: result
        for model_type, result in test_results.items()
    }
    test_path = EXPERIMENT_DIR / "test_results.json"
    with open(test_path, "w") as f:
        json.dump(test_json, f, indent=2)
    print(f"  Test results: {test_path}")
    
    # Predictions CSV with both log returns and percentages (ORIGINAL scale)
    pred_df = test_metadata.copy()
    pred_df["actual_log"] = y_test_original  # Use original, not normalized
    pred_df["actual_pct"] = (np.exp(y_test_original) - 1) * 100
    for model_type, y_pred in predictions.items():
        # predictions dict already contains inverse-transformed values
        pred_df[f"pred_{model_type}_log"] = y_pred
        pred_df[f"pred_{model_type}_pct"] = (np.exp(y_pred) - 1) * 100
        pred_df[f"error_{model_type}_pct"] = pred_df[f"pred_{model_type}_pct"] - pred_df["actual_pct"]
    pred_df["dataset"] = "test"
    pred_path = predictions_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path}")
    
    # Out-of-fold predictions from CV for ALL models (for visualization)
    # Note: OOF predictions are in normalized space, need to inverse transform
    oof_df = None
    for model_type in MODEL_TYPES:
        oof_result = cv_results[model_type].get_oof_predictions()
        if oof_result is not None:
            oof_indices, oof_preds_norm, oof_actuals_norm = oof_result
            
            # Inverse transform to original scale
            # Step 1: Inverse Z-score normalization
            oof_preds_alpha = normalizer.inverse_transform_target(oof_preds_norm)
            oof_actuals_alpha = normalizer.inverse_transform_target(oof_actuals_norm)
            
            # Step 2: Inverse alpha transform (add back weekly mean)
            if USE_ALPHA_TARGET and weekly_means_dev is not None:
                oof_preds = oof_preds_alpha + weekly_means_dev[oof_indices]
                oof_actuals = oof_actuals_alpha + weekly_means_dev[oof_indices]
            else:
                oof_preds = oof_preds_alpha
                oof_actuals = oof_actuals_alpha
            
            if oof_df is None:
                # First model - create base DataFrame
                oof_df = dev_metadata.iloc[oof_indices].copy()
                oof_df["actual_log"] = oof_actuals
                oof_df["actual_pct"] = (np.exp(oof_actuals) - 1) * 100
                oof_df["dataset"] = "dev"
            
            # Add this model's predictions (in original scale)
            oof_df[f"pred_{model_type}_log"] = oof_preds
            oof_df[f"pred_{model_type}_pct"] = (np.exp(oof_preds) - 1) * 100
            oof_df[f"error_{model_type}_pct"] = oof_df[f"pred_{model_type}_pct"] - oof_df["actual_pct"]
    
    if oof_df is not None:
        oof_path = predictions_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        print(f"  OOF predictions (all models): {oof_path}")
    
    # Experiment config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "feature_matrix_path": FEATURE_MATRIX_PATH,
            "test_weeks": TEST_WEEKS,
            "train_weeks": TRAIN_WEEKS,
            "category_filter": CATEGORY_FILTER,
            **data_metadata,
        },
        "two_stage": USE_TWO_STAGE,
        "market_model": market_model_results if USE_TWO_STAGE else None,
        "alpha_target": USE_ALPHA_TARGET,
        "normalization": normalizer.get_params(),
        "cross_validation": {
            "n_folds": N_FOLDS,
        },
        "models": MODEL_TYPES,
    }
    config_path = EXPERIMENT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    
    # Generate markdown report
    report = generate_report(
        experiment_name=EXPERIMENT_NAME,
        data_metadata=data_metadata,
        cv_results=cv_results,
        test_results=test_results,
        feature_importances=feature_importances,
        n_folds=N_FOLDS,
        market_model_results=market_model_results,
    )
    report_path = EXPERIMENT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")
    
    print()
    
    # =========================================================================
    # Step 6: Prediction Analysis (Per-Symbol & Directional)
    # =========================================================================
    print("Step 6: Running prediction analysis...")
    
    # Use best model (Ridge) for detailed analysis
    best_model_type = "ridge"
    analyzer = PredictionAnalyzer(pred_df, model_col=f"pred_{best_model_type}_log")
    
    # Generate plots
    dir_plot = analyzer.plot_directional_distributions()
    dir_plot_path = plots_dir / "directional_distributions.html"
    dir_plot.write_html(str(dir_plot_path))
    print(f"  Directional distributions: {dir_plot_path}")
    
    symbol_plot = analyzer.plot_per_symbol_rankings(top_n=20)
    symbol_plot_path = plots_dir / "per_symbol_rankings.html"
    symbol_plot.write_html(str(symbol_plot_path))
    print(f"  Per-symbol rankings: {symbol_plot_path}")
    
    # Generate analysis report
    analysis_report = analyzer.generate_report()
    analysis_path = EXPERIMENT_DIR / "prediction_analysis.md"
    with open(analysis_path, "w") as f:
        f.write(analysis_report)
    print(f"  Analysis report: {analysis_path}")
    
    # Save per-symbol rankings CSV
    rankings = analyzer.per_symbol_performance()
    rankings_path = predictions_dir / "per_symbol_performance.csv"
    rankings.to_csv(rankings_path, index=False)
    print(f"  Per-symbol CSV: {rankings_path}")
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    best_model = max(test_results.items(), key=lambda x: x[1]["metrics"]["r2"])
    best_name = best_model[1]["model_name"]
    best_r2 = best_model[1]["metrics"]["r2"]
    best_ic = best_model[1]["metrics"]["ic"]
    best_da = best_model[1]["metrics"]["directional_accuracy"]
    
    print_summary(
        experiment=EXPERIMENT_NAME,
        train_val_samples=f"{X_train_val.shape[0]:,}",
        test_samples=f"{X_test.shape[0]:,}",
        models_evaluated=len(MODEL_TYPES),
        cv_folds=N_FOLDS,
        best_model=best_name,
        best_test_r2=f"{best_r2:.4f}",
        best_test_ic=f"{best_ic:.4f}",
        best_directional_accuracy=f"{best_da:.2%}",
        output_directory=str(EXPERIMENT_DIR),
    )


def generate_report(
    experiment_name: str,
    data_metadata: Dict[str, Any],
    cv_results: Dict[str, Any],
    test_results: Dict[str, Any],
    feature_importances: Dict[str, Dict[str, float]],
    n_folds: int = 5,
    market_model_results: Dict[str, Any] = None,
) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append(f"# Experiment Report: {experiment_name}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Data summary
    lines.append("## Data Summary")
    lines.append("")
    lines.append(f"- **Total samples:** {data_metadata['total_rows']:,}")
    lines.append(f"- **Train+Validation samples:** {data_metadata['dev_rows']:,} (used for {N_FOLDS}-fold CV)")
    lines.append(f"- **Test samples:** {data_metadata['test_rows']:,} (held out, final evaluation only)")
    lines.append(f"- **Features:** {data_metadata['n_features']}")
    lines.append(f"- **Symbols:** {data_metadata['n_symbols']}")
    lines.append("")
    lines.append("**Split Strategy:** Train → Validation (via K-fold CV) → Test")
    lines.append("")
    if USE_TWO_STAGE:
        lines.append("**Architecture:** Two-Stage (Market β + Alpha α)")
        lines.append("")
    if USE_ALPHA_TARGET:
        lines.append("**Target:** Alpha (market-neutral, weekly cross-sectional mean subtracted)")
        lines.append("")
    if NORMALIZE_FEATURES:
        lines.append("**Features:** Z-score normalized")
    if NORMALIZE_TARGET:
        lines.append("**Target:** Z-score normalized (after alpha if enabled)")
    if NORMALIZE_FEATURES or NORMALIZE_TARGET:
        lines.append("")
    
    # Market model results (if two-stage)
    if market_model_results:
        lines.append("## Stage 1: Market Model (β)")
        lines.append("")
        lines.append(f"- **Model:** {market_model_results['model_type']}")
        lines.append(f"- **Training weeks:** {market_model_results['n_weeks_dev']}")
        lines.append(f"- **Test weeks:** {market_model_results['n_weeks_test']}")
        lines.append(f"- **Macro features:** {market_model_results['n_features']}")
        lines.append("")
        
        m = market_model_results['test_metrics']
        lines.append("**Test Performance:**")
        lines.append("")
        lines.append(f"| R² | MAE | RMSE | Dir. Acc. | IC |")
        lines.append(f"|----|----|------|-----------|-----|")
        lines.append(f"| {m['r2']:.4f} | {m['mae']:.6f} | {m['rmse']:.6f} | {m['directional_accuracy']:.2%} | {m['ic']:.4f} |")
        lines.append("")
        
        # Show actual vs predicted for test weeks
        lines.append("**Test Week Predictions:**")
        lines.append("")
        lines.append("| Week | Actual β | Predicted β | Error |")
        lines.append("|------|----------|-------------|-------|")
        for week in sorted(market_model_results['test_actual'].keys()):
            actual = market_model_results['test_actual'][week]
            pred = market_model_results['test_predicted'][week]
            error = pred - actual
            lines.append(f"| {week} | {actual:.4f} | {pred:.4f} | {error:+.4f} |")
        lines.append("")
        
        # Feature importance for market model
        if market_model_results.get('feature_importance'):
            imp = market_model_results['feature_importance']
            sorted_imp = sorted(imp.items(), key=lambda x: -abs(x[1]))[:5]
            lines.append("**Top 5 Market Features:**")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for feat, val in sorted_imp:
                lines.append(f"| {feat} | {val:.4f} |")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Stage 2: Alpha Model (α)")
        lines.append("")
    
    # CV results
    lines.append("## Cross-Validation Results (5-fold)")
    lines.append("")
    lines.append("| Model | R² | MAE | RMSE | Dir. Acc. | IC |")
    lines.append("|-------|----|----|------|-----------|-----|")
    
    for model_type, result in cv_results.items():
        m = result.mean_metrics
        s = result.std_metrics
        lines.append(
            f"| {result.model_name} | "
            f"{m['r2']:.4f}±{s['r2']:.4f} | "
            f"{m['mae']:.6f}±{s['mae']:.6f} | "
            f"{m['rmse']:.6f}±{s['rmse']:.6f} | "
            f"{m['directional_accuracy']:.2%}±{s['directional_accuracy']:.2%} | "
            f"{m['ic']:.4f}±{s['ic']:.4f} |"
        )
    
    lines.append("")
    
    # Test results
    lines.append("## Test Set Results (Held-out)")
    lines.append("")
    lines.append("| Model | R² | MAE | RMSE | Dir. Acc. | IC |")
    lines.append("|-------|----|----|------|-----------|-----|")
    
    for model_type, result in test_results.items():
        m = result["metrics"]
        lines.append(
            f"| {result['model_name']} | "
            f"{m['r2']:.4f} | "
            f"{m['mae']:.6f} | "
            f"{m['rmse']:.6f} | "
            f"{m['directional_accuracy']:.2%} | "
            f"{m['ic']:.4f} |"
        )
    
    lines.append("")
    
    # Best model
    best = max(test_results.items(), key=lambda x: x[1]["metrics"]["r2"])
    lines.append(f"**Best Model (by R²):** {best[1]['model_name']}")
    lines.append("")
    
    # Feature importance (for best tree model)
    for model_type in ["xgboost", "random_forest"]:
        if model_type in feature_importances:
            importance = feature_importances[model_type]
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:10]
            
            lines.append(f"## Top 10 Features ({test_results[model_type]['model_name']})")
            lines.append("")
            lines.append("| Feature | Importance |")
            lines.append("|---------|------------|")
            for feat, imp in sorted_imp:
                lines.append(f"| {feat} | {imp:.4f} |")
            lines.append("")
            break
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `06-run-experiment.py`*")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()

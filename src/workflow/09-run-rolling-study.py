#!/usr/bin/env python3
"""
09-run-rolling-study.py

Run a true rolling window study: for each week, train on all prior data
and predict one week ahead.

This script:
1. Loads the feature matrix
2. Iterates through weeks starting from MIN_HISTORY_WEEKS
3. For each week: trains model, predicts, evaluates
4. Stores all predictions and metrics
5. Generates summary report
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.model_factory import create_model
from src.training.evaluator import Evaluator
from src.training.normalizer import DataNormalizer
from src.workflow.workflow_utils import print_summary, setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp009_rolling_all_models"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME

# Data configuration
FEATURE_MATRIX_PATH = "data/processed/iex/feature_matrix.parquet"
CATEGORY_FILTER = ["target"]  # Only target ETFs

# Rolling window configuration
MIN_HISTORY_WEEKS = 52  # Start predicting after 1 year of data
MAX_TRAIN_WEEKS = 104  # None = expanding window, or int for fixed window (104 = 2 years)

# Model configuration - now multiple models
MODEL_TYPES = ["linear", "ridge", "lasso", "random_forest", "xgboost"]
NORMALIZE_FEATURES = True
NORMALIZE_TARGET = True

# Feature columns (must match what's in feature matrix)
FEATURE_COLUMNS = [
    "log_return", "log_return_intraweek", "log_range", "log_volume",
    "log_avg_daily_volume", "intra_week_volatility", "log_return_ma4",
    "log_return_ma12", "log_volume_ma4", "log_volume_ma12", "momentum_4w",
    "momentum_12w", "volatility_ma4", "volatility_ma12", "log_volume_delta",
]


# =============================================================================
# Helper Functions
# =============================================================================

def load_feature_matrix() -> pd.DataFrame:
    """Load and filter the feature matrix."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    
    # Filter to target symbols only
    if CATEGORY_FILTER:
        df = df[df["category"].isin(CATEGORY_FILTER)].copy()
    
    # Sort by week for consistent ordering
    df = df.sort_values(["week_idx", "symbol"]).reset_index(drop=True)
    
    return df


def compute_alpha_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add alpha (market-neutral) target column."""
    # Compute weekly cross-sectional mean
    weekly_means = df.groupby("week_idx")["target_return"].transform("mean")
    df = df.copy()
    df["alpha"] = df["target_return"] - weekly_means
    df["weekly_mean"] = weekly_means
    return df


def get_train_test_split(
    df: pd.DataFrame,
    test_week: int,
    min_history: int,
    max_train: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data for a specific test week.
    
    Args:
        df: Full feature matrix with alpha column
        test_week: Week index to predict
        min_history: Minimum weeks of history required
        max_train: Maximum training weeks (None = expanding)
    
    Returns:
        (train_df, test_df)
    """
    # Training: all weeks before test_week
    if max_train is not None:
        min_train_week = max(0, test_week - max_train)
        train_df = df[(df["week_idx"] >= min_train_week) & (df["week_idx"] < test_week)]
    else:
        train_df = df[df["week_idx"] < test_week]
    
    # Test: just the test week
    test_df = df[df["week_idx"] == test_week]
    
    return train_df, test_df


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str,
    normalize_features: bool = True,
    normalize_target: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Train model and predict for test week.
    
    Returns:
        (predictions, feature_importance)
    """
    # Prepare data
    X_train = train_df[feature_cols].values
    y_train = train_df["alpha"].values
    X_test = test_df[feature_cols].values
    
    # Normalize
    normalizer = DataNormalizer(
        normalize_features=normalize_features,
        normalize_target=normalize_target,
    )
    normalizer.fit(X_train, y_train)
    X_train_norm, y_train_norm = normalizer.transform(X_train, y_train)
    X_test_norm, _ = normalizer.transform(X_test, None)
    
    # Train
    model = create_model(model_type)
    model.fit(X_train_norm, y_train_norm, feature_names=feature_cols)
    
    # Predict
    y_pred_norm = model.predict(X_test_norm)
    y_pred = normalizer.inverse_transform_target(y_pred_norm)
    
    # Feature importance
    importance = model.get_feature_importance() or {}
    
    return y_pred, importance


def train_and_predict_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_types: List[str],
    normalize_features: bool = True,
    normalize_target: bool = True,
) -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
    """Train all models and predict for test week.
    
    Returns:
        Dict mapping model_type -> (predictions, feature_importance)
    """
    # Prepare data once
    X_train = train_df[feature_cols].values
    y_train = train_df["alpha"].values
    X_test = test_df[feature_cols].values
    
    # Normalize once
    normalizer = DataNormalizer(
        normalize_features=normalize_features,
        normalize_target=normalize_target,
    )
    normalizer.fit(X_train, y_train)
    X_train_norm, y_train_norm = normalizer.transform(X_train, y_train)
    X_test_norm, _ = normalizer.transform(X_test, None)
    
    results = {}
    for model_type in model_types:
        # Train
        model = create_model(model_type)
        model.fit(X_train_norm, y_train_norm, feature_names=feature_cols)
        
        # Predict
        y_pred_norm = model.predict(X_test_norm)
        y_pred = normalizer.inverse_transform_target(y_pred_norm)
        
        # Feature importance
        importance = model.get_feature_importance() or {}
        
        results[model_type] = (y_pred, importance)
    
    return results


def evaluate_week(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics for a single week."""
    evaluator = Evaluator(y_true, y_pred)
    return evaluator.compute_metrics()


# =============================================================================
# Main
# =============================================================================

@workflow_script("09-run-rolling-study")
def main() -> None:
    """Run the rolling window study."""
    
    # Create output directories
    plots_dir = EXPERIMENT_DIR / "plots"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(EXPERIMENT_DIR / "plots" / "symbol_dashboards", exist_ok=True)
    
    print("=" * 80)
    print(f"ROLLING WINDOW STUDY: {EXPERIMENT_NAME}")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading feature matrix...")
    df = load_feature_matrix()
    
    # Add alpha target
    df = compute_alpha_target(df)
    
    # Get week range
    all_weeks = sorted(df["week_idx"].unique())
    first_week = all_weeks[0]
    last_week = all_weeks[-1]
    
    # Determine which weeks to predict
    start_predict_week = first_week + MIN_HISTORY_WEEKS
    predict_weeks = [w for w in all_weeks if w >= start_predict_week]
    
    print(f"  Total weeks in data: {len(all_weeks)} (weeks {first_week} to {last_week})")
    print(f"  Weeks to predict: {len(predict_weeks)} (starting at week {start_predict_week})")
    print(f"  Unique symbols: {df['symbol'].nunique()}")
    print()
    
    # =========================================================================
    # Step 2: Rolling Window Loop
    # =========================================================================
    print("Step 2: Running rolling predictions...")
    print(f"  Models: {MODEL_TYPES}")
    print(f"  Window: {'Expanding' if MAX_TRAIN_WEEKS is None else f'Fixed {MAX_TRAIN_WEEKS} weeks'}")
    print()
    
    # Storage for each model
    all_predictions = {m: [] for m in MODEL_TYPES}
    weekly_metrics = {m: [] for m in MODEL_TYPES}
    weekly_timings = []
    
    for i, test_week in enumerate(predict_weeks):
        week_start_time = time.time()
        
        # Get train/test split
        train_df, test_df = get_train_test_split(
            df, test_week, MIN_HISTORY_WEEKS, MAX_TRAIN_WEEKS
        )
        
        if len(test_df) == 0:
            continue
        
        # Train and predict all models
        model_results = train_and_predict_all_models(
            train_df, test_df, FEATURE_COLUMNS, MODEL_TYPES,
            NORMALIZE_FEATURES, NORMALIZE_TARGET,
        )
        
        # Get actuals
        y_true = test_df["alpha"].values
        
        # Process each model's results
        model_metrics_this_week = {}
        for model_type, (y_pred, importance) in model_results.items():
            # Evaluate
            metrics = evaluate_week(y_true, y_pred)
            model_metrics_this_week[model_type] = metrics
            
            # Store predictions
            week_preds = test_df[["symbol", "week_idx", "week_start", "alpha", "weekly_mean", "target_return"]].copy()
            week_preds["predicted_alpha"] = y_pred
            week_preds["predicted_return"] = y_pred + week_preds["weekly_mean"]
            week_preds["model"] = model_type
            all_predictions[model_type].append(week_preds)
            
            # Store metrics
            weekly_metrics[model_type].append({
                "week_idx": test_week,
                "week_start": test_df["week_start"].iloc[0] if "week_start" in test_df.columns else None,
                "n_symbols": len(test_df),
                "n_train_samples": len(train_df),
                "n_train_weeks": train_df["week_idx"].nunique(),
                **metrics,
            })
        
        week_time = time.time() - week_start_time
        weekly_timings.append(week_time)
        
        # Progress (show lasso as reference)
        if (i + 1) % 10 == 0 or (i + 1) == len(predict_weeks):
            lasso_da = model_metrics_this_week["lasso"]["directional_accuracy"]
            print(f"  Week {i+1}/{len(predict_weeks)}: Lasso Dir.Acc={lasso_da:.1%}, "
                  f"n={len(test_df)}, time={week_time:.2f}s")
    
    print()
    
    # =========================================================================
    # Step 3: Aggregate Results
    # =========================================================================
    print("Step 3: Aggregating results...")
    
    # Combine predictions and metrics for each model
    model_summaries = {}
    
    for model_type in MODEL_TYPES:
        predictions_df = pd.concat(all_predictions[model_type], ignore_index=True)
        metrics_df = pd.DataFrame(weekly_metrics[model_type])
        
        model_summaries[model_type] = {
            "predictions_df": predictions_df,
            "metrics_df": metrics_df,
            "n_weeks": len(metrics_df),
            "n_predictions": len(predictions_df),
            "n_symbols": predictions_df["symbol"].nunique(),
            "mean_directional_accuracy": metrics_df["directional_accuracy"].mean(),
            "std_directional_accuracy": metrics_df["directional_accuracy"].std(),
            "median_directional_accuracy": metrics_df["directional_accuracy"].median(),
            "mean_ic": metrics_df["ic"].mean(),
            "std_ic": metrics_df["ic"].std(),
            "mean_r2": metrics_df["r2"].mean(),
            "pct_weeks_above_50": (metrics_df["directional_accuracy"] > 0.5).mean(),
            "pct_weeks_above_60": (metrics_df["directional_accuracy"] > 0.6).mean(),
        }
    
    # Print comparison table
    print()
    print("  Model Comparison:")
    print("  " + "-" * 60)
    print(f"  {'Model':<15} {'Dir.Acc':<10} {'IC':<10} {'>50%':<10} {'>60%':<10}")
    print("  " + "-" * 60)
    for model_type in MODEL_TYPES:
        s = model_summaries[model_type]
        print(f"  {model_type:<15} {s['mean_directional_accuracy']:.1%}      {s['mean_ic']:.3f}     "
              f"{s['pct_weeks_above_50']:.1%}     {s['pct_weeks_above_60']:.1%}")
    print("  " + "-" * 60)
    print()
    
    # Find best model
    best_model = max(MODEL_TYPES, key=lambda m: model_summaries[m]["mean_directional_accuracy"])
    best_summary = model_summaries[best_model]
    
    print(f"  Best model: {best_model} ({best_summary['mean_directional_accuracy']:.1%} dir.acc)")
    print()
    
    # Use best model for detailed outputs
    predictions_df = model_summaries[best_model]["predictions_df"]
    metrics_df = model_summaries[best_model]["metrics_df"]
    
    # Add timing info
    metrics_df["time_seconds"] = weekly_timings[:len(metrics_df)]
    
    # Overall summary (for best model, plus comparison)
    overall_metrics = {
        "best_model": best_model,
        "n_weeks": best_summary["n_weeks"],
        "n_predictions": best_summary["n_predictions"],
        "n_symbols": best_summary["n_symbols"],
        "mean_directional_accuracy": best_summary["mean_directional_accuracy"],
        "std_directional_accuracy": best_summary["std_directional_accuracy"],
        "median_directional_accuracy": best_summary["median_directional_accuracy"],
        "mean_ic": best_summary["mean_ic"],
        "std_ic": best_summary["std_ic"],
        "mean_r2": best_summary["mean_r2"],
        "pct_weeks_above_50": best_summary["pct_weeks_above_50"],
        "pct_weeks_above_60": best_summary["pct_weeks_above_60"],
        "total_time_seconds": sum(weekly_timings),
        "avg_time_per_week": np.mean(weekly_timings),
        "model_comparison": {
            m: {
                "mean_directional_accuracy": model_summaries[m]["mean_directional_accuracy"],
                "mean_ic": model_summaries[m]["mean_ic"],
                "pct_weeks_above_50": model_summaries[m]["pct_weeks_above_50"],
                "pct_weeks_above_60": model_summaries[m]["pct_weeks_above_60"],
            }
            for m in MODEL_TYPES
        },
    }
    
    # Per-symbol aggregation (using best model)
    symbol_metrics = predictions_df.groupby("symbol").apply(
        lambda g: pd.Series({
            "n_weeks": len(g),
            "mean_actual_alpha": g["alpha"].mean(),
            "mean_pred_alpha": g["predicted_alpha"].mean(),
            "directional_accuracy": (np.sign(g["alpha"]) == np.sign(g["predicted_alpha"])).mean(),
            "mae": np.abs(g["alpha"] - g["predicted_alpha"]).mean(),
            "correlation": g["alpha"].corr(g["predicted_alpha"]) if len(g) > 2 else 0,
        }),
        include_groups=False,
    ).reset_index()
    
    # =========================================================================
    # Step 4: Save Results
    # =========================================================================
    print("Step 4: Saving results...")
    
    # Predictions (parquet for efficiency)
    pred_path = EXPERIMENT_DIR / "weekly_predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    print(f"  Predictions: {pred_path}")
    
    # Weekly metrics (CSV for easy viewing)
    metrics_path = EXPERIMENT_DIR / "weekly_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Weekly metrics: {metrics_path}")
    
    # Per-symbol metrics
    symbol_path = EXPERIMENT_DIR / "per_symbol_metrics.csv"
    symbol_metrics.to_csv(symbol_path, index=False)
    print(f"  Per-symbol metrics: {symbol_path}")
    
    # Overall summary
    summary_path = EXPERIMENT_DIR / "overall_summary.json"
    with open(summary_path, "w") as f:
        json.dump(overall_metrics, f, indent=2)
    print(f"  Overall summary: {summary_path}")
    
    # Config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "min_history_weeks": MIN_HISTORY_WEEKS,
        "max_train_weeks": MAX_TRAIN_WEEKS,
        "model_types": MODEL_TYPES,
        "best_model": best_model,
        "normalize_features": NORMALIZE_FEATURES,
        "normalize_target": NORMALIZE_TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "category_filter": CATEGORY_FILTER,
    }
    config_path = EXPERIMENT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    
    # Report
    report = generate_report(overall_metrics, metrics_df, config)
    report_path = EXPERIMENT_DIR / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")
    
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_summary(
        experiment=EXPERIMENT_NAME,
        weeks_evaluated=overall_metrics["n_weeks"],
        total_predictions=f"{overall_metrics['n_predictions']:,}",
        models_tested=len(MODEL_TYPES),
        best_model=best_model,
        best_directional_accuracy=f"{overall_metrics['mean_directional_accuracy']:.1%}",
        best_ic=f"{overall_metrics['mean_ic']:.3f}",
        pct_weeks_above_60=f"{overall_metrics['pct_weeks_above_60']:.1%}",
        total_time=f"{overall_metrics['total_time_seconds']:.1f}s",
        output_directory=str(EXPERIMENT_DIR),
    )


def generate_report(
    overall: Dict[str, Any],
    weekly: pd.DataFrame,
    config: Dict[str, Any],
) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append("# Rolling Window Study Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Configuration
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Models:** {config.get('model_types', [config.get('model_type', 'unknown')])}")
    window_str = "Expanding" if config['max_train_weeks'] is None else f"Fixed {config['max_train_weeks']} weeks"
    lines.append(f"- **Window:** {window_str}")
    lines.append(f"- **Min history:** {config['min_history_weeks']} weeks")
    lines.append(f"- **Features:** {len(config['feature_columns'])}")
    lines.append(f"- **Normalization:** Features={config['normalize_features']}, Target={config['normalize_target']}")
    lines.append("")
    
    # Model comparison table (if available)
    if "model_comparison" in overall:
        lines.append("## Model Comparison")
        lines.append("")
        lines.append("| Model | Dir. Accuracy | IC | Weeks > 50% | Weeks > 60% |")
        lines.append("|-------|---------------|-----|-------------|-------------|")
        for model, stats in overall["model_comparison"].items():
            lines.append(
                f"| {model} | {stats['mean_directional_accuracy']:.1%} | "
                f"{stats['mean_ic']:.3f} | {stats['pct_weeks_above_50']:.1%} | "
                f"{stats['pct_weeks_above_60']:.1%} |"
            )
        lines.append("")
        lines.append(f"**Best Model:** {overall.get('best_model', 'N/A')}")
        lines.append("")
    
    # Overall results
    lines.append("## Overall Results")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Weeks evaluated | {overall['n_weeks']} |")
    lines.append(f"| Total predictions | {overall['n_predictions']:,} |")
    lines.append(f"| Unique symbols | {overall['n_symbols']} |")
    lines.append(f"| **Mean Directional Accuracy** | **{overall['mean_directional_accuracy']:.1%}** |")
    lines.append(f"| Std Directional Accuracy | {overall['std_directional_accuracy']:.1%} |")
    lines.append(f"| Median Directional Accuracy | {overall['median_directional_accuracy']:.1%} |")
    lines.append(f"| Mean IC | {overall['mean_ic']:.3f} |")
    lines.append(f"| Mean R² | {overall['mean_r2']:.4f} |")
    lines.append(f"| % weeks > 50% accuracy | {overall['pct_weeks_above_50']:.1%} |")
    lines.append(f"| % weeks > 60% accuracy | {overall['pct_weeks_above_60']:.1%} |")
    lines.append("")
    
    # Best and worst weeks
    lines.append("## Best and Worst Weeks")
    lines.append("")
    
    best_weeks = weekly.nlargest(5, "directional_accuracy")
    lines.append("### Top 5 Weeks")
    lines.append("")
    lines.append("| Week | Date | Dir. Acc | IC | n |")
    lines.append("|------|------|----------|-----|---|")
    for _, row in best_weeks.iterrows():
        if pd.notna(row["week_start"]):
            date_str = str(row["week_start"])[:10]
        else:
            date_str = str(int(row["week_idx"]))
        lines.append(f"| {int(row['week_idx'])} | {date_str} | {row['directional_accuracy']:.1%} | {row['ic']:.3f} | {int(row['n_symbols'])} |")
    lines.append("")
    
    worst_weeks = weekly.nsmallest(5, "directional_accuracy")
    lines.append("### Bottom 5 Weeks")
    lines.append("")
    lines.append("| Week | Date | Dir. Acc | IC | n |")
    lines.append("|------|------|----------|-----|---|")
    for _, row in worst_weeks.iterrows():
        if pd.notna(row["week_start"]):
            date_str = str(row["week_start"])[:10]
        else:
            date_str = str(int(row["week_idx"]))
        lines.append(f"| {int(row['week_idx'])} | {date_str} | {row['directional_accuracy']:.1%} | {row['ic']:.3f} | {int(row['n_symbols'])} |")
    lines.append("")
    
    # Timing
    lines.append("## Performance")
    lines.append("")
    lines.append(f"- Total time: {overall['total_time_seconds']:.1f} seconds")
    lines.append(f"- Average per week: {overall['avg_time_per_week']:.2f} seconds")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `09-run-rolling-study.py`*")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()

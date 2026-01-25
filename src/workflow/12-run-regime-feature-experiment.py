#!/usr/bin/env python3
"""
12-run-regime-feature-experiment.py

Experiment: Add regime indicators as features to the rolling window model.

Hypothesis: Adding VIX regime, market trend, and dispersion as features
will improve IC and directional accuracy over the baseline (exp009).

This is exp011 in the research grid.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.training import DataNormalizer, Evaluator, create_model
from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp011_regime_features"
MIN_TRAIN_WEEKS = 52
MAX_TRAIN_WEEKS = 104  # Fixed window
MODEL_TYPES = ["linear", "ridge", "lasso"]  # Focus on linear models (they won)

# Feature matrix paths (long format - same as exp009)
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"

# Weekly data for computing regime features (macro symbols)
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Base feature columns (from exp009) - without positional encodings
BASE_FEATURE_COLUMNS = [
    "log_return",
    "log_return_intraweek",
    "log_range",
    "log_volume",
    "log_avg_daily_volume",
    "intra_week_volatility",
    "log_return_ma4",
    "log_return_ma12",
    "log_volume_ma4",
    "log_volume_ma12",
    "momentum_4w",
    "momentum_12w",
    "volatility_ma4",
    "volatility_ma12",
    "log_volume_delta",
]

# Regime feature columns (will be added)
REGIME_FEATURE_COLUMNS = [
    "vix_level",           # VIXY close (continuous, normalized)
    "spy_momentum_4w",     # SPY 4-week momentum
    "cross_dispersion",    # Cross-sectional dispersion of returns
    "market_breadth",      # % of ETFs with positive returns
]


def load_macro_data() -> dict[str, pd.DataFrame]:
    """Load macro symbol weekly data for regime computation."""
    macro_data = {}
    
    for symbol in ["VIXY", "SPY"]:
        csv_path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=["week_start"])
            df = df.set_index("week_start").sort_index()
            macro_data[symbol] = df
            logging.info(f"Loaded {symbol}: {len(df)} weeks")
        else:
            logging.warning(f"Missing macro data: {csv_path}")
    
    return macro_data


def compute_regime_features(
    feature_df: pd.DataFrame,
    macro_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute regime features for each week in the feature matrix.
    
    These are computed CAUSALLY - using only data available at prediction time.
    """
    # Get unique weeks with week_idx mapping
    week_mapping = feature_df[["week_start", "week_idx"]].drop_duplicates()
    week_mapping = week_mapping.sort_values("week_idx")
    weeks = week_mapping["week_start"].tolist()
    
    regime_records = []
    
    for _, row in week_mapping.iterrows():
        week = row["week_start"]
        week_idx = row["week_idx"]
        record = {"week_idx": week_idx, "week_start": week}
        
        # VIX level (use VIXY as proxy) - normalized by its own history
        if "VIXY" in macro_data:
            vixy = macro_data["VIXY"]
            if week in vixy.index:
                record["vix_level"] = np.log(vixy.loc[week, "close"])  # Log for normalization
            else:
                # Find most recent available
                prior = vixy.index[vixy.index <= week]
                if len(prior) > 0:
                    record["vix_level"] = np.log(vixy.loc[prior[-1], "close"])
                else:
                    record["vix_level"] = np.nan
        else:
            record["vix_level"] = np.nan
        
        # SPY momentum
        if "SPY" in macro_data:
            spy = macro_data["SPY"]
            if week in spy.index:
                # Use pre-computed momentum if available
                if "momentum_4w" in spy.columns:
                    record["spy_momentum_4w"] = spy.loc[week, "momentum_4w"]
                else:
                    record["spy_momentum_4w"] = np.nan
            else:
                # Find most recent
                prior = spy.index[spy.index <= week]
                if len(prior) > 0 and "momentum_4w" in spy.columns:
                    record["spy_momentum_4w"] = spy.loc[prior[-1], "momentum_4w"]
                else:
                    record["spy_momentum_4w"] = np.nan
        else:
            record["spy_momentum_4w"] = np.nan
        
        # Cross-sectional features computed from feature_df
        week_data = feature_df[feature_df["week_start"] == week]
        if len(week_data) > 1:
            record["cross_dispersion"] = week_data["log_return"].std()
            record["market_breadth"] = (week_data["log_return"] > 0).mean()
        else:
            record["cross_dispersion"] = np.nan
            record["market_breadth"] = np.nan
        
        regime_records.append(record)
    
    regime_df = pd.DataFrame(regime_records)
    return regime_df


def run_rolling_study_with_regime_features(
    feature_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    model_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling window study with regime features added.
    """
    # Merge regime features into feature_df by week_idx
    feature_df = feature_df.copy()
    feature_df = feature_df.merge(
        regime_df[["week_idx"] + REGIME_FEATURE_COLUMNS],
        on="week_idx",
        how="left"
    )
    
    # All feature columns
    all_feature_cols = BASE_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS
    
    # Filter to target ETFs only
    target_df = feature_df[feature_df["category"] == "target"].copy()
    
    # Compute alpha (market-neutral target)
    weekly_means = target_df.groupby("week_idx")["target_return"].transform("mean")
    target_df["alpha"] = target_df["target_return"] - weekly_means
    
    # Get unique week indices sorted
    week_indices = sorted(target_df["week_idx"].unique())
    logging.info(f"Total weeks: {len(week_indices)}")
    
    # Find first prediction week (need MIN_TRAIN_WEEKS of history)
    start_idx = MIN_TRAIN_WEEKS
    prediction_week_indices = [w for w in week_indices if w >= start_idx]
    logging.info(f"Will predict {len(prediction_week_indices)} weeks")
    
    weekly_results = []
    all_predictions = []
    
    for i, pred_week_idx in enumerate(prediction_week_indices):
        # Determine training window
        if MAX_TRAIN_WEEKS is not None:
            min_train_week_idx = max(0, pred_week_idx - MAX_TRAIN_WEEKS)
        else:
            min_train_week_idx = 0
        
        # Get training data
        train_mask = (target_df["week_idx"] >= min_train_week_idx) & (target_df["week_idx"] < pred_week_idx)
        train_data = target_df[train_mask].dropna(subset=all_feature_cols + ["alpha"])
        
        # Get test data (the prediction week)
        test_mask = target_df["week_idx"] == pred_week_idx
        test_data = target_df[test_mask].dropna(subset=all_feature_cols + ["alpha"])
        
        if len(train_data) < 100 or len(test_data) < 10:
            continue
        
        X_train = train_data[all_feature_cols].values
        y_train = train_data["alpha"].values
        X_test = test_data[all_feature_cols].values
        y_test = test_data["alpha"].values
        
        # Get week_start for this prediction week
        week_start = test_data["week_start"].iloc[0]
        
        # Normalize features
        normalizer = DataNormalizer(normalize_features=True, normalize_target=True)
        normalizer.fit(X_train, y_train)
        
        X_train_norm = normalizer.transform_features(X_train)
        y_train_norm = normalizer.transform_target(y_train)
        X_test_norm = normalizer.transform_features(X_test)
        
        # Train and predict with each model
        for model_type in model_types:
            model = create_model(model_type)
            model.fit(X_train_norm, y_train_norm)
            
            y_pred_norm = model.predict(X_test_norm)
            y_pred_alpha = normalizer.inverse_transform_target(y_pred_norm)
            
            # Evaluate
            evaluator = Evaluator(y_test, y_pred_alpha)
            metrics = evaluator.compute_metrics()
            
            weekly_results.append({
                "week_idx": pred_week_idx,
                "week_start": week_start,
                "model": model_type,
                "n_symbols": len(y_test),
                "n_train_samples": len(y_train),
                "n_train_weeks": pred_week_idx - min_train_week_idx,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "directional_accuracy": metrics["directional_accuracy"],
                "information_coefficient": metrics["ic"],
            })
            
            # Store predictions
            for symbol, actual, pred in zip(
                test_data["symbol"].values,
                y_test,
                y_pred_alpha
            ):
                all_predictions.append({
                    "week_idx": pred_week_idx,
                    "week_start": week_start,
                    "symbol": symbol,
                    "model": model_type,
                    "actual_alpha": actual,
                    "predicted_alpha": pred,
                })
        
        if (i + 1) % 20 == 0:
            logging.info(f"Processed {i + 1}/{len(prediction_week_indices)} weeks")
    
    weekly_df = pd.DataFrame(weekly_results)
    predictions_df = pd.DataFrame(all_predictions)
    
    return weekly_df, predictions_df


def generate_report(
    weekly_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate experiment report."""
    report_lines = [
        f"# Experiment Report: {EXPERIMENT_NAME}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Hypothesis",
        "",
        "Adding regime indicators (VIX level, SPY momentum, cross-dispersion, market breadth)",
        "as features will improve IC and directional accuracy over the baseline (exp009).",
        "",
        "## Configuration",
        "",
        f"- Training window: Fixed {MAX_TRAIN_WEEKS} weeks",
        f"- Min history: {MIN_TRAIN_WEEKS} weeks",
        f"- Models: {', '.join(MODEL_TYPES)}",
        f"- Base features: {len(BASE_FEATURE_COLUMNS)}",
        f"- Regime features: {len(REGIME_FEATURE_COLUMNS)}",
        f"- Total features: {len(BASE_FEATURE_COLUMNS) + len(REGIME_FEATURE_COLUMNS)}",
        "",
        "## Regime Features Added",
        "",
        "| Feature | Description |",
        "|---------|-------------|",
        "| vix_level | Log VIXY closing price (VIX proxy) |",
        "| spy_momentum_4w | SPY 4-week momentum |",
        "| cross_dispersion | Std of ETF returns that week |",
        "| market_breadth | % of ETFs with positive returns |",
        "",
        "## Results by Model",
        "",
    ]
    
    # Summary by model
    summary = weekly_df.groupby("model").agg({
        "directional_accuracy": ["mean", "std"],
        "information_coefficient": ["mean", "std"],
        "r2": "mean",
        "n_symbols": "mean",
    }).round(4)
    
    report_lines.append("| Model | Dir.Acc (mean±std) | IC (mean±std) | R² | Symbols |")
    report_lines.append("|-------|-------------------|---------------|-----|---------|")
    
    for model in MODEL_TYPES:
        if model in summary.index:
            row = summary.loc[model]
            dir_acc_mean = row[("directional_accuracy", "mean")]
            dir_acc_std = row[("directional_accuracy", "std")]
            ic_mean = row[("information_coefficient", "mean")]
            ic_std = row[("information_coefficient", "std")]
            r2 = row[("r2", "mean")]
            n_sym = row[("n_symbols", "mean")]
            
            report_lines.append(
                f"| {model} | {dir_acc_mean:.1%} ± {dir_acc_std:.1%} | "
                f"{ic_mean:.3f} ± {ic_std:.3f} | {r2:.4f} | {n_sym:.0f} |"
            )
    
    # Comparison to baseline
    report_lines.extend([
        "",
        "## Comparison to Baseline (exp009)",
        "",
        "| Metric | exp009 (baseline) | exp011 (regime features) | Delta |",
        "|--------|-------------------|--------------------------|-------|",
    ])
    
    # exp009 baseline values (from research log)
    baseline_dir_acc = 0.509
    baseline_ic = 0.030
    
    best_model = weekly_df.groupby("model")["information_coefficient"].mean().idxmax()
    new_dir_acc = weekly_df[weekly_df["model"] == best_model]["directional_accuracy"].mean()
    new_ic = weekly_df[weekly_df["model"] == best_model]["information_coefficient"].mean()
    
    delta_dir_acc = new_dir_acc - baseline_dir_acc
    delta_ic = new_ic - baseline_ic
    
    report_lines.append(
        f"| Dir.Acc | {baseline_dir_acc:.1%} | {new_dir_acc:.1%} | "
        f"{'+' if delta_dir_acc > 0 else ''}{delta_dir_acc:.1%} |"
    )
    report_lines.append(
        f"| IC | {baseline_ic:.3f} | {new_ic:.3f} | "
        f"{'+' if delta_ic > 0 else ''}{delta_ic:.3f} |"
    )
    
    # Kill criteria evaluation
    report_lines.extend([
        "",
        "## Kill Criteria Evaluation",
        "",
    ])
    
    if new_ic < 0.02:
        report_lines.append("❌ **IC < 0.02**: Signal too weak, consider abandoning this branch")
    else:
        report_lines.append("✓ IC ≥ 0.02: Signal meets minimum threshold")
    
    if new_dir_acc < 0.52:
        report_lines.append("❌ **Dir.Acc < 52%**: Barely above random")
    else:
        report_lines.append("✓ Dir.Acc ≥ 52%: Above random threshold")
    
    if delta_ic < 0.01 and delta_dir_acc < 0.02:
        report_lines.append("❌ **No meaningful improvement over baseline**")
    else:
        report_lines.append("✓ Improvement over baseline detected")
    
    # Decision
    report_lines.extend([
        "",
        "## Decision",
        "",
    ])
    
    if delta_ic > 0.01 or delta_dir_acc > 0.02:
        report_lines.append("**CONTINUE**: Regime features show promise. Next: regime-specific models.")
    elif new_ic > 0.02 and new_dir_acc > 0.50:
        report_lines.append("**HOLD**: No improvement, but baseline is viable. Try classification.")
    else:
        report_lines.append("**PIVOT**: This approach exhausted. Move to alternative targets or simplify.")
    
    report_lines.append("")
    report_lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    logging.info(f"Report saved: {report_path}")


@workflow_script("12-run-regime-feature-experiment")
def main() -> None:
    """Run regime feature experiment."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load feature matrix
    logging.info(f"Loading feature matrix: {FEATURE_MATRIX_PATH}")
    feature_df = pd.read_parquet(FEATURE_MATRIX_PATH)
    logging.info(f"Feature matrix: {len(feature_df)} rows, {feature_df['week_idx'].nunique()} weeks")
    
    # Load macro data for regime computation
    logging.info("Loading macro data for regime features...")
    macro_data = load_macro_data()
    
    # Compute regime features
    logging.info("Computing regime features...")
    regime_df = compute_regime_features(feature_df, macro_data)
    logging.info(f"Regime features computed for {len(regime_df)} weeks")
    
    # Run rolling study with regime features
    logging.info("Running rolling study with regime features...")
    weekly_df, predictions_df = run_rolling_study_with_regime_features(
        feature_df, regime_df, MODEL_TYPES
    )
    
    # Save results
    weekly_df.to_csv(OUTPUT_DIR / "weekly_metrics.csv", index=False)
    predictions_df.to_parquet(OUTPUT_DIR / "weekly_predictions.parquet", index=False)
    
    # Save config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "min_train_weeks": MIN_TRAIN_WEEKS,
        "max_train_weeks": MAX_TRAIN_WEEKS,
        "model_types": MODEL_TYPES,
        "base_features": BASE_FEATURE_COLUMNS,
        "regime_features": REGIME_FEATURE_COLUMNS,
        "generated": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate report
    generate_report(weekly_df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 60)
    
    for model in MODEL_TYPES:
        model_df = weekly_df[weekly_df["model"] == model]
        print(f"\n{model.upper()}:")
        print(f"  Dir.Acc: {model_df['directional_accuracy'].mean():.1%} ± {model_df['directional_accuracy'].std():.1%}")
        print(f"  IC:      {model_df['information_coefficient'].mean():.3f} ± {model_df['information_coefficient'].std():.3f}")
    
    print("\n" + "=" * 60)
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()

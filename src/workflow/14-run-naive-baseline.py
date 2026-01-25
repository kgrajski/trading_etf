#!/usr/bin/env python3
"""
14-run-naive-baseline.py

Experiment: Compare ML models against naive baseline strategies.

Hypotheses tested:
1. Momentum: Buy last week's winners (top decile by return)
2. Mean-Reversion: Buy last week's losers (bottom decile by return)
3. Random: Random ranking (control)

This is exp014 in the research grid.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp014_naive_baseline"
MIN_HISTORY_WEEKS = 52

# Feature matrix path (long format)
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Strategies to test
STRATEGIES = ["momentum", "mean_reversion", "random"]


def compute_strategy_predictions(
    df: pd.DataFrame,
    strategy: str,
    week_idx: int,
) -> pd.DataFrame:
    """
    Generate predictions for a given week using a naive strategy.
    
    Args:
        df: Feature matrix with all data
        strategy: One of 'momentum', 'mean_reversion', 'random'
        week_idx: Week to predict
    
    Returns:
        DataFrame with symbol, predicted_rank, actual_alpha
    """
    # Get previous week's returns (the signal)
    prev_week = df[df["week_idx"] == week_idx - 1][["symbol", "log_return"]].copy()
    prev_week = prev_week.rename(columns={"log_return": "prev_return"})
    
    # Get current week's data (the outcome)
    curr_week = df[df["week_idx"] == week_idx][["symbol", "log_return"]].copy()
    
    # Compute alpha (market-neutral)
    curr_mean = curr_week["log_return"].mean()
    curr_week["actual_alpha"] = curr_week["log_return"] - curr_mean
    
    # Merge
    merged = curr_week.merge(prev_week, on="symbol", how="inner")
    
    if len(merged) < 10:
        return pd.DataFrame()
    
    # Generate predictions based on strategy
    if strategy == "momentum":
        # Higher previous return → predict higher future alpha
        merged["predicted_alpha"] = merged["prev_return"]
    
    elif strategy == "mean_reversion":
        # Lower previous return → predict higher future alpha
        merged["predicted_alpha"] = -merged["prev_return"]
    
    elif strategy == "random":
        # Random ranking
        np.random.seed(week_idx)  # Reproducible randomness
        merged["predicted_alpha"] = np.random.randn(len(merged))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return merged[["symbol", "predicted_alpha", "actual_alpha"]]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    if len(y_true) < 3:
        return {
            "directional_accuracy": 0.5,
            "information_coefficient": 0.0,
            "n_symbols": len(y_true),
        }
    
    # Directional accuracy
    correct = (np.sign(y_true) == np.sign(y_pred))
    dir_acc = np.mean(correct)
    
    # Information coefficient (Spearman)
    ic, _ = stats.spearmanr(y_true, y_pred)
    if np.isnan(ic):
        ic = 0.0
    
    return {
        "directional_accuracy": dir_acc,
        "information_coefficient": ic,
        "n_symbols": len(y_true),
    }


def run_rolling_study(
    df: pd.DataFrame,
    strategy: str,
) -> pd.DataFrame:
    """
    Run rolling window study for a naive strategy.
    """
    # Filter to target ETFs
    target_df = df[df["category"] == "target"].copy()
    
    # Get unique weeks
    week_indices = sorted(target_df["week_idx"].unique())
    prediction_week_indices = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    logging.info(f"Strategy: {strategy}, predicting {len(prediction_week_indices)} weeks")
    
    weekly_results = []
    
    for week_idx in prediction_week_indices:
        predictions = compute_strategy_predictions(target_df, strategy, week_idx)
        
        if len(predictions) < 10:
            continue
        
        metrics = evaluate_predictions(
            predictions["actual_alpha"].values,
            predictions["predicted_alpha"].values,
        )
        
        # Get week_start for this week
        week_start = target_df[target_df["week_idx"] == week_idx]["week_start"].iloc[0]
        
        weekly_results.append({
            "week_idx": week_idx,
            "week_start": week_start,
            "strategy": strategy,
            **metrics,
        })
    
    return pd.DataFrame(weekly_results)


def generate_report(
    results_df: pd.DataFrame,
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
        "Test whether naive trading strategies outperform our ML models:",
        "- **Momentum:** Buy last week's winners (top performers)",
        "- **Mean-Reversion:** Buy last week's losers (bottom performers)",
        "- **Random:** Random selection (control)",
        "",
        "## Results Summary",
        "",
        "| Strategy | Dir.Acc (mean±std) | IC (mean±std) | Symbols |",
        "|----------|-------------------|---------------|---------|",
    ]
    
    for strategy in STRATEGIES:
        strat_df = results_df[results_df["strategy"] == strategy]
        dir_acc_mean = strat_df["directional_accuracy"].mean()
        dir_acc_std = strat_df["directional_accuracy"].std()
        ic_mean = strat_df["information_coefficient"].mean()
        ic_std = strat_df["information_coefficient"].std()
        n_sym = strat_df["n_symbols"].mean()
        
        report_lines.append(
            f"| {strategy} | {dir_acc_mean:.1%} ± {dir_acc_std:.1%} | "
            f"{ic_mean:.3f} ± {ic_std:.3f} | {n_sym:.0f} |"
        )
    
    # Comparison to ML baseline
    report_lines.extend([
        "",
        "## Comparison to ML Models",
        "",
        "| Approach | Dir.Acc | IC | Notes |",
        "|----------|---------|-----|-------|",
        "| exp009 (Linear/Ridge) | 50.9% | 0.030 | ML baseline |",
        "| exp011 (+ Regime features) | 51.2% | 0.038 | Best ML so far |",
    ])
    
    for strategy in STRATEGIES:
        strat_df = results_df[results_df["strategy"] == strategy]
        dir_acc = strat_df["directional_accuracy"].mean()
        ic = strat_df["information_coefficient"].mean()
        
        if strategy == "momentum":
            note = "Prior week return as signal"
        elif strategy == "mean_reversion":
            note = "Inverse of prior week return"
        else:
            note = "Random baseline"
        
        report_lines.append(f"| {strategy} | {dir_acc:.1%} | {ic:.3f} | {note} |")
    
    # Interpretation
    report_lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    
    # Find best strategy
    summary = results_df.groupby("strategy")["information_coefficient"].mean()
    best_strategy = summary.idxmax()
    best_ic = summary.max()
    
    ml_baseline_ic = 0.038  # exp011
    
    if best_ic > ml_baseline_ic:
        report_lines.append(
            f"**{best_strategy.upper()} BEATS ML!** (IC={best_ic:.3f} vs {ml_baseline_ic:.3f})"
        )
        report_lines.append("")
        report_lines.append(
            "This suggests we should use the naive strategy as a baseline feature "
            "or reconsider our modeling approach."
        )
    elif best_ic > 0.02:
        report_lines.append(
            f"**{best_strategy}** shows some signal (IC={best_ic:.3f}) but ML is better."
        )
        report_lines.append("")
        report_lines.append(
            "Consider combining naive signals with ML predictions."
        )
    else:
        report_lines.append(
            "No naive strategy shows meaningful signal. ML approach is justified."
        )
    
    # Kill criteria
    report_lines.extend([
        "",
        "## Decision",
        "",
    ])
    
    if best_ic > ml_baseline_ic:
        report_lines.append(
            f"**PIVOT:** Use {best_strategy} as baseline or primary signal."
        )
    else:
        report_lines.append(
            "**CONTINUE:** ML models outperform naive baselines."
        )
    
    report_lines.append("")
    report_lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    logging.info(f"Report saved: {report_path}")


@workflow_script("14-run-naive-baseline")
def main() -> None:
    """Run naive baseline experiment."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load feature matrix
    logging.info(f"Loading feature matrix: {FEATURE_MATRIX_PATH}")
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    logging.info(f"Feature matrix: {len(df)} rows, {df['week_idx'].nunique()} weeks")
    
    # Run rolling study for each strategy
    all_results = []
    
    for strategy in STRATEGIES:
        results = run_rolling_study(df, strategy)
        all_results.append(results)
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "weekly_metrics.csv", index=False)
    
    # Save config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "strategies": STRATEGIES,
        "min_history_weeks": MIN_HISTORY_WEEKS,
        "generated": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate report
    generate_report(results_df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 60)
    
    print("\nNaive Strategy Results:")
    print("-" * 40)
    
    for strategy in STRATEGIES:
        strat_df = results_df[results_df["strategy"] == strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Dir.Acc: {strat_df['directional_accuracy'].mean():.1%} ± {strat_df['directional_accuracy'].std():.1%}")
        print(f"  IC:      {strat_df['information_coefficient'].mean():.3f} ± {strat_df['information_coefficient'].std():.3f}")
    
    print("\n" + "-" * 40)
    print("ML Baseline (exp011): Dir.Acc=51.2%, IC=0.038")
    print("=" * 60)
    
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()

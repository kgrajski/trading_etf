#!/usr/bin/env python3
"""
17-analyze-losers.py

Deep dive into the "losers" that trigger mean-reversion entries:
1. Distribution of loser returns (how extreme are they?)
2. What types of ETFs are chronic losers?
3. Do extreme losers behave differently than moderate losers?
4. Test filtering strategies (skip worst X%, take next Y%)

Goal: Understand if we can classify "safer" losers.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp017_loser_analysis"
MIN_HISTORY_WEEKS = 52

# Feature matrix path
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME


def load_data() -> pd.DataFrame:
    """Load feature matrix."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[df["category"] == "target"].copy()
    return df


def analyze_loser_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the distribution of weekly returns for the worst performers.
    """
    week_indices = sorted(df["week_idx"].unique())
    prediction_weeks = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    records = []
    
    for week_idx in prediction_weeks:
        # Get prior week's returns
        prior_week = df[df["week_idx"] == week_idx - 1][["symbol", "log_return", "name"]].copy()
        prior_week = prior_week.sort_values("log_return")
        
        n_symbols = len(prior_week)
        if n_symbols < 20:
            continue
        
        # Get current week's returns (outcome)
        curr_week = df[df["week_idx"] == week_idx][["symbol", "log_return"]].copy()
        curr_week = curr_week.rename(columns={"log_return": "next_return"})
        
        merged = prior_week.merge(curr_week, on="symbol", how="inner")
        merged["percentile"] = merged["log_return"].rank(pct=True)
        
        for _, row in merged.iterrows():
            records.append({
                "week_idx": week_idx,
                "symbol": row["symbol"],
                "name": row["name"],
                "prior_return": row["log_return"],
                "next_return": row["next_return"],
                "percentile": row["percentile"],
            })
    
    return pd.DataFrame(records)


def analyze_by_percentile_bucket(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by percentile buckets and compute mean-reversion success.
    """
    # Create buckets
    bins = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 1.0]
    labels = ["0-1%", "1-2%", "2-5%", "5-10%", "10-20%", "20-50%", 
              "50-80%", "80-90%", "90-95%", "95-98%", "98-99%", "99-100%"]
    
    analysis_df["bucket"] = pd.cut(analysis_df["percentile"], bins=bins, labels=labels)
    
    summary = analysis_df.groupby("bucket", observed=True).agg({
        "prior_return": ["mean", "std", "count"],
        "next_return": ["mean", "std"],
    }).round(4)
    
    # Flatten column names
    summary.columns = ["prior_mean", "prior_std", "count", "next_mean", "next_std"]
    summary = summary.reset_index()
    
    # Compute mean-reversion ratio
    summary["reversion_ratio"] = -summary["next_mean"] / summary["prior_mean"]
    
    return summary


def analyze_chronic_losers(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """
    Which symbols appear most often in the bottom 5%?
    """
    bottom_5 = analysis_df[analysis_df["percentile"] <= 0.05]
    
    chronic = bottom_5.groupby(["symbol", "name"]).agg({
        "prior_return": ["count", "mean"],
        "next_return": "mean",
    }).round(4)
    
    chronic.columns = ["times_in_bottom5", "avg_prior_return", "avg_next_return"]
    chronic = chronic.reset_index()
    chronic = chronic.sort_values("times_in_bottom5", ascending=False)
    
    return chronic


def test_filtered_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Test different filtering strategies:
    - Skip worst 1%, take next 5%
    - Skip worst 2%, take next 5%
    - Take percentile 2-7%
    - etc.
    """
    week_indices = sorted(df["week_idx"].unique())
    prediction_weeks = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    strategies = [
        {"name": "bottom_5", "skip": 0, "take": 5, "desc": "Bottom 5 (current)"},
        {"name": "skip1_take5", "skip": 1, "take": 5, "desc": "Skip worst 1%, take next 5%"},
        {"name": "skip2_take5", "skip": 2, "take": 5, "desc": "Skip worst 2%, take next 5%"},
        {"name": "skip5_take5", "skip": 5, "take": 5, "desc": "Skip worst 5%, take next 5%"},
        {"name": "pct_1to5", "pct_low": 0.01, "pct_high": 0.05, "desc": "Percentile 1-5%"},
        {"name": "pct_2to7", "pct_low": 0.02, "pct_high": 0.07, "desc": "Percentile 2-7%"},
        {"name": "pct_5to10", "pct_low": 0.05, "pct_high": 0.10, "desc": "Percentile 5-10%"},
    ]
    
    results = []
    
    for strat in strategies:
        trades = []
        
        for week_idx in prediction_weeks:
            # Get prior week's returns
            prior_week = df[df["week_idx"] == week_idx - 1][["symbol", "log_return"]].copy()
            prior_week = prior_week.sort_values("log_return")
            n_symbols = len(prior_week)
            
            if n_symbols < 20:
                continue
            
            # Apply selection strategy
            if "skip" in strat:
                selected = prior_week.iloc[strat["skip"]:strat["skip"]+strat["take"]]
            else:
                # Percentile-based
                prior_week["pct"] = prior_week["log_return"].rank(pct=True)
                selected = prior_week[
                    (prior_week["pct"] >= strat["pct_low"]) & 
                    (prior_week["pct"] < strat["pct_high"])
                ]
            
            # Get next week returns
            curr_week = df[df["week_idx"] == week_idx][["symbol", "log_return"]].copy()
            curr_week = curr_week.rename(columns={"log_return": "next_return"})
            
            merged = selected.merge(curr_week, on="symbol", how="inner")
            
            for _, row in merged.iterrows():
                trades.append({
                    "strategy": strat["name"],
                    "week_idx": week_idx,
                    "symbol": row["symbol"],
                    "prior_return": row["log_return"],
                    "next_return": row["next_return"],
                })
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            results.append({
                "strategy": strat["name"],
                "description": strat["desc"],
                "n_trades": len(trades_df),
                "avg_prior_return": trades_df["prior_return"].mean(),
                "avg_next_return": trades_df["next_return"].mean(),
                "next_return_std": trades_df["next_return"].std(),
                "win_rate": (trades_df["next_return"] > 0).mean(),
                "sharpe": trades_df["next_return"].mean() / trades_df["next_return"].std() * np.sqrt(52),
            })
    
    return pd.DataFrame(results)


def generate_report(
    bucket_summary: pd.DataFrame,
    chronic_losers: pd.DataFrame,
    strategy_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate analysis report."""
    report_lines = [
        f"# Loser Analysis Report: {EXPERIMENT_NAME}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Question",
        "",
        "Are all 'losers' created equal? Can we identify safer mean-reversion candidates?",
        "",
        "---",
        "",
        "## 1. Return by Percentile Bucket",
        "",
        "How does the next-week return vary by how extreme the prior loss was?",
        "",
        "| Bucket | Prior Return | Count | Next Return | Reversion Ratio |",
        "|--------|--------------|-------|-------------|-----------------|",
    ]
    
    for _, row in bucket_summary.iterrows():
        report_lines.append(
            f"| {row['bucket']} | {row['prior_mean']:.2%} | {row['count']:.0f} | "
            f"{row['next_mean']:.2%} | {row['reversion_ratio']:.2f} |"
        )
    
    # Chronic losers
    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Chronic Losers (Bottom 5% Most Often)",
        "",
        "Which ETFs appear repeatedly in the bottom 5%?",
        "",
        "| Symbol | Name | Times in Bottom 5% | Avg Prior Return | Avg Next Return |",
        "|--------|------|-------------------|------------------|-----------------|",
    ])
    
    for _, row in chronic_losers.head(20).iterrows():
        name = row["name"][:40] if isinstance(row["name"], str) else "N/A"
        report_lines.append(
            f"| {row['symbol']} | {name} | {row['times_in_bottom5']:.0f} | "
            f"{row['avg_prior_return']:.2%} | {row['avg_next_return']:.2%} |"
        )
    
    # Strategy comparison
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. Filtering Strategy Comparison",
        "",
        "Does skipping the most extreme losers improve results?",
        "",
        "| Strategy | Description | Trades | Avg Prior | Avg Next | Win Rate | Sharpe |",
        "|----------|-------------|--------|-----------|----------|----------|--------|",
    ])
    
    for _, row in strategy_comparison.iterrows():
        report_lines.append(
            f"| {row['strategy']} | {row['description']} | {row['n_trades']:.0f} | "
            f"{row['avg_prior_return']:.2%} | {row['avg_next_return']:.2%} | "
            f"{row['win_rate']:.1%} | {row['sharpe']:.2f} |"
        )
    
    # Conclusions
    report_lines.extend([
        "",
        "---",
        "",
        "## 4. Data Availability",
        "",
        "**Pre-market data:** Not available with Alpaca IEX (free tier).",
        "We have daily OHLCV only, no intraday or pre-market.",
        "",
        "**What we can see:**",
        "- Weekly open, high, low, close, volume",
        "- Monday open vs Friday close gap (implicitly)",
        "",
        "---",
        "",
        "## 5. Conclusions",
        "",
    ])
    
    # Find best strategy
    best = strategy_comparison.loc[strategy_comparison["sharpe"].idxmax()]
    
    report_lines.append(f"**Best filtering strategy:** `{best['strategy']}`")
    report_lines.append(f"- Sharpe: {best['sharpe']:.2f}")
    report_lines.append(f"- Win rate: {best['win_rate']:.1%}")
    report_lines.append(f"- Avg next-week return: {best['avg_next_return']:.2%}")
    
    report_lines.append("")
    report_lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    logging.info(f"Report saved: {report_path}")


@workflow_script("17-analyze-losers")
def main() -> None:
    """Analyze loser characteristics."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    df = load_data()
    logging.info(f"Feature matrix: {len(df)} rows, {df['week_idx'].nunique()} weeks")
    
    # Analyze loser distribution
    logging.info("Analyzing loser distribution...")
    analysis_df = analyze_loser_distribution(df)
    analysis_df.to_csv(OUTPUT_DIR / "loser_analysis.csv", index=False)
    
    # Bucket analysis
    logging.info("Analyzing by percentile bucket...")
    bucket_summary = analyze_by_percentile_bucket(analysis_df)
    bucket_summary.to_csv(OUTPUT_DIR / "bucket_summary.csv", index=False)
    
    # Chronic losers
    logging.info("Identifying chronic losers...")
    chronic_losers = analyze_chronic_losers(analysis_df)
    chronic_losers.to_csv(OUTPUT_DIR / "chronic_losers.csv", index=False)
    
    # Test filtering strategies
    logging.info("Testing filtering strategies...")
    strategy_comparison = test_filtered_strategies(df)
    strategy_comparison.to_csv(OUTPUT_DIR / "strategy_comparison.csv", index=False)
    
    # Generate report
    generate_report(bucket_summary, chronic_losers, strategy_comparison, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"LOSER ANALYSIS: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    print("\nPercentile Bucket Analysis (prior return → next return):")
    print("-" * 50)
    for _, row in bucket_summary.iterrows():
        print(f"  {row['bucket']:<10}: {row['prior_mean']:>7.2%} → {row['next_mean']:>7.2%} "
              f"(n={row['count']:.0f})")
    
    print("\n" + "-" * 50)
    print("Strategy Comparison:")
    print("-" * 50)
    for _, row in strategy_comparison.iterrows():
        print(f"  {row['strategy']:<15}: Win={row['win_rate']:.1%}, "
              f"Sharpe={row['sharpe']:.2f}, Next={row['avg_next_return']:.2%}")
    
    print("\n" + "-" * 50)
    print("Top 10 Chronic Losers:")
    for _, row in chronic_losers.head(10).iterrows():
        print(f"  {row['symbol']:<6}: {row['times_in_bottom5']:.0f}x in bottom 5%, "
              f"avg next: {row['avg_next_return']:.2%}")
    
    print("=" * 70)
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()

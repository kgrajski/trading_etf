#!/usr/bin/env python3
"""
08-analyze-symbol-characteristics.py

Investigate relationship between symbol characteristics (volume, volatility)
and prediction quality. Helps decide whether symbol filtering would help.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.workflow.workflow_utils import print_summary, setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp003_ftr_tgt_norm"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME
FEATURE_MATRIX_PATH = Path("data/processed/iex/feature_matrix.parquet")
OUTPUT_DIR = EXPERIMENT_DIR / "symbol_analysis"


# =============================================================================
# Main
# =============================================================================

@workflow_script("08-analyze-symbol-characteristics")
def main() -> None:
    """Analyze relationship between symbol characteristics and prediction quality."""
    
    print("=" * 80)
    print(f"SYMBOL CHARACTERISTIC ANALYSIS: {EXPERIMENT_NAME}")
    print("=" * 80)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading data...")
    
    # Load feature matrix for symbol characteristics
    features_df = pd.read_parquet(FEATURE_MATRIX_PATH)
    features_df = features_df[features_df["category"] == "target"]  # ETFs only
    print(f"  Feature matrix: {len(features_df):,} rows")
    
    # Load per-symbol prediction performance
    perf_path = EXPERIMENT_DIR / "predictions" / "per_symbol_performance.csv"
    perf_df = pd.read_csv(perf_path)
    print(f"  Symbol performance: {len(perf_df)} symbols")
    print()
    
    # =========================================================================
    # Step 2: Compute Symbol Characteristics
    # =========================================================================
    print("Step 2: Computing symbol characteristics...")
    
    # Aggregate features per symbol
    char_df = features_df.groupby("symbol").agg({
        "log_volume": "mean",
        "log_avg_daily_volume": "mean",
        "intra_week_volatility": "mean",
        "log_return": ["mean", "std"],
        "week_idx": "count",
    }).reset_index()
    
    # Flatten column names
    char_df.columns = [
        "symbol", "avg_log_volume", "avg_log_daily_volume", 
        "avg_volatility", "avg_return", "return_std", "n_weeks"
    ]
    
    print(f"  Computed characteristics for {len(char_df)} symbols")
    print()
    
    # =========================================================================
    # Step 3: Merge with Prediction Performance
    # =========================================================================
    print("Step 3: Merging characteristics with prediction performance...")
    
    # Rename n_weeks in char_df to avoid collision
    char_df = char_df.rename(columns={"n_weeks": "n_weeks_history"})
    
    merged_df = perf_df.merge(char_df, on="symbol", how="inner")
    print(f"  Merged: {len(merged_df)} symbols")
    print(f"  Columns: {list(merged_df.columns)}")
    print()
    
    # =========================================================================
    # Step 4: Correlation Analysis
    # =========================================================================
    print("Step 4: Computing correlations...")
    
    characteristics = ["avg_log_volume", "avg_volatility", "return_std", "n_weeks_history"]
    metrics = ["dir_accuracy", "mae_pct"]
    
    correlations = []
    for char in characteristics:
        for metric in metrics:
            valid = merged_df[[char, metric]].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[char], valid[metric])
                correlations.append({
                    "characteristic": char,
                    "metric": metric,
                    "correlation": r,
                    "p_value": p,
                    "significant": p < 0.05,
                    "n": len(valid),
                })
    
    corr_df = pd.DataFrame(correlations)
    print("\n  Correlation Results:")
    print("  " + "-" * 70)
    for _, row in corr_df.iterrows():
        sig = "✓" if row["significant"] else " "
        print(f"  {row['characteristic']:20s} vs {row['metric']:15s}: "
              f"r={row['correlation']:+.3f}, p={row['p_value']:.4f} {sig}")
    print()
    
    # =========================================================================
    # Step 5: Create Visualizations
    # =========================================================================
    print("Step 5: Creating visualizations...")
    
    # 2x2 scatter matrix
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Volume vs Directional Accuracy",
            "Volatility vs Directional Accuracy",
            "Volume vs MAE",
            "Volatility vs MAE",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )
    
    # Volume vs Dir Acc
    r_vol_dir = corr_df[(corr_df["characteristic"] == "avg_log_volume") & 
                        (corr_df["metric"] == "dir_accuracy")]["correlation"].values[0]
    fig.add_trace(
        go.Scatter(
            x=merged_df["avg_log_volume"],
            y=merged_df["dir_accuracy"] * 100,
            mode="markers",
            marker=dict(size=5, color="#3498DB", opacity=0.6),
            text=merged_df["symbol"],
            hovertemplate="<b>%{text}</b><br>Volume: %{x:.2f}<br>Dir.Acc: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="x domain", yref="y domain",
        text=f"r = {r_vol_dir:.3f}", showarrow=False,
        font=dict(size=12, color="white"), row=1, col=1,
    )
    
    # Volatility vs Dir Acc
    r_vol_dir2 = corr_df[(corr_df["characteristic"] == "avg_volatility") & 
                         (corr_df["metric"] == "dir_accuracy")]["correlation"].values[0]
    fig.add_trace(
        go.Scatter(
            x=merged_df["avg_volatility"],
            y=merged_df["dir_accuracy"] * 100,
            mode="markers",
            marker=dict(size=5, color="#E74C3C", opacity=0.6),
            text=merged_df["symbol"],
            hovertemplate="<b>%{text}</b><br>Volatility: %{x:.4f}<br>Dir.Acc: %{y:.1f}%<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="x2 domain", yref="y2 domain",
        text=f"r = {r_vol_dir2:.3f}", showarrow=False,
        font=dict(size=12, color="white"), row=1, col=2,
    )
    
    # Volume vs MAE
    r_vol_mae = corr_df[(corr_df["characteristic"] == "avg_log_volume") & 
                        (corr_df["metric"] == "mae_pct")]["correlation"].values[0]
    fig.add_trace(
        go.Scatter(
            x=merged_df["avg_log_volume"],
            y=merged_df["mae_pct"],
            mode="markers",
            marker=dict(size=5, color="#3498DB", opacity=0.6),
            text=merged_df["symbol"],
            hovertemplate="<b>%{text}</b><br>Volume: %{x:.2f}<br>MAE: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="x3 domain", yref="y3 domain",
        text=f"r = {r_vol_mae:.3f}", showarrow=False,
        font=dict(size=12, color="white"), row=2, col=1,
    )
    
    # Volatility vs MAE
    r_vol_mae2 = corr_df[(corr_df["characteristic"] == "avg_volatility") & 
                         (corr_df["metric"] == "mae_pct")]["correlation"].values[0]
    fig.add_trace(
        go.Scatter(
            x=merged_df["avg_volatility"],
            y=merged_df["mae_pct"],
            mode="markers",
            marker=dict(size=5, color="#E74C3C", opacity=0.6),
            text=merged_df["symbol"],
            hovertemplate="<b>%{text}</b><br>Volatility: %{x:.4f}<br>MAE: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="x4 domain", yref="y4 domain",
        text=f"r = {r_vol_mae2:.3f}", showarrow=False,
        font=dict(size=12, color="white"), row=2, col=2,
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Symbol Characteristics vs Prediction Quality</b>",
            x=0.5,
            xanchor="center",
        ),
        height=700,
        template="plotly_dark",
        showlegend=False,
    )
    
    fig.update_xaxes(title_text="Avg Log Volume", row=1, col=1)
    fig.update_xaxes(title_text="Avg Volatility", row=1, col=2)
    fig.update_xaxes(title_text="Avg Log Volume", row=2, col=1)
    fig.update_xaxes(title_text="Avg Volatility", row=2, col=2)
    fig.update_yaxes(title_text="Dir. Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Dir. Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="MAE (%)", row=2, col=1)
    fig.update_yaxes(title_text="MAE (%)", row=2, col=2)
    
    scatter_path = OUTPUT_DIR / "characteristics_vs_performance.html"
    fig.write_html(str(scatter_path))
    print(f"  Scatter plot: {scatter_path}")
    
    # =========================================================================
    # Step 6: Stratified Analysis
    # =========================================================================
    print("\nStep 6: Stratified analysis...")
    
    # Split into terciles by volume
    merged_df["volume_tercile"] = pd.qcut(
        merged_df["avg_log_volume"], 3, labels=["Low", "Medium", "High"]
    )
    
    # Split into terciles by volatility
    merged_df["volatility_tercile"] = pd.qcut(
        merged_df["avg_volatility"], 3, labels=["Low", "Medium", "High"]
    )
    
    print("\n  Performance by Volume Tercile:")
    print("  " + "-" * 50)
    vol_stats = merged_df.groupby("volume_tercile").agg({
        "dir_accuracy": ["mean", "std", "count"],
        "mae_pct": ["mean", "std"],
    }).round(4)
    print(vol_stats.to_string())
    
    print("\n  Performance by Volatility Tercile:")
    print("  " + "-" * 50)
    vol_stats2 = merged_df.groupby("volatility_tercile").agg({
        "dir_accuracy": ["mean", "std", "count"],
        "mae_pct": ["mean", "std"],
    }).round(4)
    print(vol_stats2.to_string())
    
    # =========================================================================
    # Step 7: Save Results
    # =========================================================================
    print("\nStep 7: Saving results...")
    
    # Save merged data
    merged_path = OUTPUT_DIR / "symbol_characteristics_merged.csv"
    merged_df.to_csv(merged_path, index=False)
    print(f"  Merged data: {merged_path}")
    
    # Save correlation results
    corr_path = OUTPUT_DIR / "correlation_results.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"  Correlations: {corr_path}")
    
    # Generate report
    report_lines = [
        "# Symbol Characteristic Analysis",
        "",
        f"**Experiment:** {EXPERIMENT_NAME}",
        f"**Symbols analyzed:** {len(merged_df)}",
        "",
        "## Correlation Summary",
        "",
        "| Characteristic | Metric | Correlation | P-value | Significant |",
        "|----------------|--------|-------------|---------|-------------|",
    ]
    
    for _, row in corr_df.iterrows():
        sig = "✓" if row["significant"] else ""
        report_lines.append(
            f"| {row['characteristic']} | {row['metric']} | "
            f"{row['correlation']:+.3f} | {row['p_value']:.4f} | {sig} |"
        )
    
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Positive correlation with Dir.Acc**: Higher values → better predictions",
        "- **Negative correlation with MAE**: Higher values → lower error",
        "- **|r| < 0.1**: Negligible relationship",
        "- **|r| 0.1-0.3**: Weak relationship",
        "- **|r| > 0.3**: Moderate relationship",
        "",
        "## Recommendation",
        "",
    ])
    
    # Add recommendation based on results
    strong_corrs = corr_df[corr_df["correlation"].abs() > 0.15]
    if len(strong_corrs) > 0:
        report_lines.append("Based on the correlations, consider filtering by:")
        for _, row in strong_corrs.iterrows():
            direction = "higher" if (row["correlation"] > 0 and "accuracy" in row["metric"]) or \
                                   (row["correlation"] < 0 and "mae" in row["metric"]) else "lower"
            report_lines.append(f"- **{row['characteristic']}**: {direction} values may improve results")
    else:
        report_lines.append("No strong correlations found. Symbol filtering may not significantly improve results.")
    
    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"  Report: {report_path}")
    
    print()
    print_summary(
        experiment=EXPERIMENT_NAME,
        symbols_analyzed=len(merged_df),
        correlations_computed=len(corr_df),
        significant_correlations=corr_df["significant"].sum(),
        output_directory=str(OUTPUT_DIR),
    )


if __name__ == "__main__":
    main()

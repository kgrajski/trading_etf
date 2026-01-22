#!/usr/bin/env python3
"""
10-visualize-rolling-study.py

Generate visualizations for the rolling window study.

This script:
1. Loads weekly metrics and predictions
2. Creates KPI time series dashboard
3. Creates metric distribution plots
4. Creates per-symbol dashboards
5. Creates inspector for navigation
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.workflow_utils import setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp009_rolling_all_models"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME
PLOTS_DIR = EXPERIMENT_DIR / "plots"


# =============================================================================
# Visualization Functions
# =============================================================================

def create_kpi_timeseries(metrics_df: pd.DataFrame) -> go.Figure:
    """Create 4-panel KPI time series dashboard."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Directional Accuracy Over Time",
            "Information Coefficient (IC) Over Time",
            "R² Over Time",
            "Symbols Predicted Per Week",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    # Parse dates
    if "week_start" in metrics_df.columns:
        x_vals = pd.to_datetime(metrics_df["week_start"])
    else:
        x_vals = metrics_df["week_idx"]
    
    # Rolling averages (4-week)
    da_rolling = metrics_df["directional_accuracy"].rolling(4, min_periods=1).mean()
    ic_rolling = metrics_df["ic"].rolling(4, min_periods=1).mean()
    r2_rolling = metrics_df["r2"].rolling(4, min_periods=1).mean()
    
    # Panel 1: Directional Accuracy
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=metrics_df["directional_accuracy"],
            mode="lines+markers",
            name="Weekly",
            line=dict(color="#3498db", width=1),
            marker=dict(size=4),
            hovertemplate="Week: %{x}<br>Dir.Acc: %{y:.1%}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=da_rolling,
            mode="lines",
            name="4-week avg",
            line=dict(color="#e74c3c", width=2),
            hovertemplate="4-week avg: %{y:.1%}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0.6, line_dash="dot", line_color="green", row=1, col=1)
    
    # Panel 2: IC
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=metrics_df["ic"],
            mode="lines+markers",
            name="Weekly IC",
            line=dict(color="#9b59b6", width=1),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate="Week: %{x}<br>IC: %{y:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=ic_rolling,
            mode="lines",
            name="4-week avg IC",
            line=dict(color="#e74c3c", width=2),
            showlegend=False,
            hovertemplate="4-week avg: %{y:.3f}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Panel 3: R²
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=metrics_df["r2"],
            mode="lines+markers",
            name="Weekly R²",
            line=dict(color="#27ae60", width=1),
            marker=dict(size=4),
            showlegend=False,
            hovertemplate="Week: %{x}<br>R²: %{y:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=r2_rolling,
            mode="lines",
            name="4-week avg R²",
            line=dict(color="#e74c3c", width=2),
            showlegend=False,
            hovertemplate="4-week avg: %{y:.4f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Panel 4: Symbols per week
    fig.add_trace(
        go.Bar(
            x=x_vals, y=metrics_df["n_symbols"],
            name="Symbols",
            marker_color="#f39c12",
            showlegend=False,
            hovertemplate="Week: %{x}<br>Symbols: %{y}<extra></extra>",
        ),
        row=2, col=2,
    )
    
    # Layout
    mean_da = metrics_df["directional_accuracy"].mean()
    mean_ic = metrics_df["ic"].mean()
    
    fig.update_layout(
        title=dict(
            text=f"Rolling Window Study: KPI Time Series<br>"
                 f"<sub>Mean Dir.Acc: {mean_da:.1%} | Mean IC: {mean_ic:.3f} | {len(metrics_df)} weeks</sub>",
            x=0.5,
        ),
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
    )
    
    # Y-axis labels
    fig.update_yaxes(title_text="Directional Accuracy", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=2, col=1)
    fig.update_yaxes(title_text="# Symbols", row=2, col=2)
    
    # Sync x-axes
    fig.update_xaxes(matches="x1")
    
    return fig


def create_kpi_distributions(metrics_df: pd.DataFrame) -> go.Figure:
    """Create histograms of KPI distributions."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Directional Accuracy Distribution",
            "IC Distribution",
            "R² Distribution",
            "Cumulative Accuracy Over Time",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    # Panel 1: Directional Accuracy histogram
    fig.add_trace(
        go.Histogram(
            x=metrics_df["directional_accuracy"],
            nbinsx=20,
            name="Dir.Acc",
            marker_color="#3498db",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_vline(x=metrics_df["directional_accuracy"].mean(), 
                  line_dash="solid", line_color="green", row=1, col=1)
    
    # Panel 2: IC histogram
    fig.add_trace(
        go.Histogram(
            x=metrics_df["ic"],
            nbinsx=20,
            name="IC",
            marker_color="#9b59b6",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.add_vline(x=metrics_df["ic"].mean(), 
                  line_dash="solid", line_color="green", row=1, col=2)
    
    # Panel 3: R² histogram
    fig.add_trace(
        go.Histogram(
            x=metrics_df["r2"],
            nbinsx=20,
            name="R²",
            marker_color="#27ae60",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_vline(x=metrics_df["r2"].mean(), 
                  line_dash="solid", line_color="green", row=2, col=1)
    
    # Panel 4: Cumulative accuracy
    # Compute running accuracy
    cumsum_correct = (metrics_df["directional_accuracy"] * metrics_df["n_symbols"]).cumsum()
    cumsum_total = metrics_df["n_symbols"].cumsum()
    cumulative_acc = cumsum_correct / cumsum_total
    
    if "week_start" in metrics_df.columns:
        x_vals = pd.to_datetime(metrics_df["week_start"])
    else:
        x_vals = metrics_df["week_idx"]
    
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=cumulative_acc,
            mode="lines",
            name="Cumulative",
            line=dict(color="#e74c3c", width=2),
            showlegend=False,
            hovertemplate="Cumulative Acc: %{y:.1%}<extra></extra>",
        ),
        row=2, col=2,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=2, col=2)
    
    # Stats annotation
    stats_text = (
        f"Dir.Acc: {metrics_df['directional_accuracy'].mean():.1%} ± {metrics_df['directional_accuracy'].std():.1%}<br>"
        f"IC: {metrics_df['ic'].mean():.3f} ± {metrics_df['ic'].std():.3f}<br>"
        f"R²: {metrics_df['r2'].mean():.4f} ± {metrics_df['r2'].std():.4f}<br>"
        f"Weeks > 50%: {(metrics_df['directional_accuracy'] > 0.5).mean():.1%}<br>"
        f"Weeks > 60%: {(metrics_df['directional_accuracy'] > 0.6).mean():.1%}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=1.0, y=1.0,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
    )
    
    fig.update_layout(
        title=dict(
            text="Rolling Window Study: KPI Distributions",
            x=0.5,
        ),
        height=600,
    )
    
    fig.update_xaxes(title_text="Directional Accuracy", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="IC", row=1, col=2)
    fig.update_xaxes(title_text="R²", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Accuracy", tickformat=".0%", row=2, col=2)
    
    return fig


def create_symbol_dashboard(
    symbol: str,
    predictions_df: pd.DataFrame,
    symbol_metrics: pd.Series,
) -> go.Figure:
    """Create dashboard for a single symbol."""
    
    symbol_data = predictions_df[predictions_df["symbol"] == symbol].sort_values("week_idx")
    
    if len(symbol_data) < 2:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Predicted vs Actual Alpha",
            "Prediction Error Over Time",
            "Alpha Distribution",
            "Directional Accuracy by Period",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    if "week_start" in symbol_data.columns:
        x_vals = pd.to_datetime(symbol_data["week_start"])
    else:
        x_vals = symbol_data["week_idx"]
    
    # Panel 1: Predicted vs Actual
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=symbol_data["alpha"],
            mode="lines",
            name="Actual α",
            line=dict(color="#3498db", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=symbol_data["predicted_alpha"],
            mode="lines",
            name="Predicted α",
            line=dict(color="#e74c3c", width=2, dash="dash"),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    
    # Panel 2: Prediction error
    error = symbol_data["predicted_alpha"] - symbol_data["alpha"]
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=error,
            mode="lines+markers",
            name="Error",
            line=dict(color="#9b59b6", width=1),
            marker=dict(size=4),
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Panel 3: Alpha distribution
    fig.add_trace(
        go.Histogram(x=symbol_data["alpha"], name="Actual", 
                     marker_color="#3498db", opacity=0.7, showlegend=False),
        row=2, col=1,
    )
    fig.add_trace(
        go.Histogram(x=symbol_data["predicted_alpha"], name="Predicted",
                     marker_color="#e74c3c", opacity=0.7, showlegend=False),
        row=2, col=1,
    )
    fig.update_layout(barmode="overlay")
    
    # Panel 4: Rolling directional accuracy
    correct = (np.sign(symbol_data["alpha"]) == np.sign(symbol_data["predicted_alpha"])).astype(int)
    rolling_acc = correct.rolling(10, min_periods=1).mean()
    
    fig.add_trace(
        go.Scatter(
            x=x_vals, y=rolling_acc,
            mode="lines",
            name="10-week rolling acc",
            line=dict(color="#27ae60", width=2),
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=2)
    
    # Layout
    da = symbol_metrics["directional_accuracy"]
    mae = symbol_metrics["mae"]
    n = symbol_metrics["n_weeks"]
    
    fig.update_layout(
        title=dict(
            text=f"{symbol}: Rolling Window Performance<br>"
                 f"<sub>Dir.Acc: {da:.1%} | MAE: {mae:.4f} | Weeks: {n}</sub>",
            x=0.5,
        ),
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    fig.update_xaxes(matches="x1")
    fig.update_yaxes(title_text="Alpha", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", tickformat=".0%", row=2, col=2)
    
    return fig


def create_inspector(metrics_df: pd.DataFrame, symbol_metrics: pd.DataFrame) -> str:
    """Create HTML inspector for navigation."""
    
    mean_da = metrics_df["directional_accuracy"].mean()
    mean_ic = metrics_df["ic"].mean()
    n_weeks = len(metrics_df)
    
    # Sort symbols by directional accuracy
    symbols_sorted = symbol_metrics.sort_values("directional_accuracy", ascending=False)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Rolling Study Inspector - {EXPERIMENT_NAME}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; }}
        .stats {{ display: flex; gap: 20px; margin-top: 10px; }}
        .stat {{ background: rgba(255,255,255,0.2); padding: 10px 15px; border-radius: 5px; }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ margin-top: 0; color: #2c3e50; }}
        .plot-link {{ display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
        .plot-link:hover {{ background: #2980b9; }}
        .symbol-list {{ display: flex; flex-wrap: wrap; gap: 5px; max-height: 400px; overflow-y: auto; }}
        .symbol-link {{ display: inline-block; padding: 5px 10px; background: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 3px; font-size: 0.9em; }}
        .symbol-link:hover {{ background: #3498db; color: white; }}
        .symbol-link .acc {{ font-size: 0.8em; color: #7f8c8d; }}
        .symbol-link:hover .acc {{ color: #ecf0f1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Rolling Window Study: {EXPERIMENT_NAME}</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{mean_da:.1%}</div>
                <div>Mean Dir.Acc</div>
            </div>
            <div class="stat">
                <div class="stat-value">{mean_ic:.3f}</div>
                <div>Mean IC</div>
            </div>
            <div class="stat">
                <div class="stat-value">{n_weeks}</div>
                <div>Weeks</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(symbol_metrics)}</div>
                <div>Symbols</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Main Dashboards</h2>
        <a href="kpi_timeseries.html" class="plot-link">📈 KPI Time Series</a>
        <a href="kpi_distributions.html" class="plot-link">📊 KPI Distributions</a>
    </div>
    
    <div class="section">
        <h2>Symbol Dashboards (sorted by Dir.Acc)</h2>
        <div class="symbol-list">
"""
    
    for _, row in symbols_sorted.iterrows():
        symbol = row["symbol"]
        acc = row["directional_accuracy"]
        html += f'            <a href="symbol_dashboards/{symbol}.html" class="symbol-link">{symbol} <span class="acc">({acc:.0%})</span></a>\n'
    
    html += """        </div>
    </div>
</body>
</html>
"""
    return html


# =============================================================================
# Main
# =============================================================================

@workflow_script("10-visualize-rolling-study")
def main() -> None:
    """Generate visualizations for rolling study."""
    
    print("=" * 80)
    print(f"VISUALIZING: {EXPERIMENT_NAME}")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading data...")
    
    metrics_df = pd.read_csv(EXPERIMENT_DIR / "weekly_metrics.csv")
    predictions_df = pd.read_parquet(EXPERIMENT_DIR / "weekly_predictions.parquet")
    symbol_metrics = pd.read_csv(EXPERIMENT_DIR / "per_symbol_metrics.csv")
    
    print(f"  Weeks: {len(metrics_df)}")
    print(f"  Predictions: {len(predictions_df):,}")
    print(f"  Symbols: {len(symbol_metrics)}")
    print()
    
    # =========================================================================
    # Step 2: Main Dashboards
    # =========================================================================
    print("Step 2: Creating main dashboards...")
    
    # KPI time series
    fig_ts = create_kpi_timeseries(metrics_df)
    ts_path = PLOTS_DIR / "kpi_timeseries.html"
    fig_ts.write_html(str(ts_path))
    print(f"  KPI time series: {ts_path}")
    
    # KPI distributions
    fig_dist = create_kpi_distributions(metrics_df)
    dist_path = PLOTS_DIR / "kpi_distributions.html"
    fig_dist.write_html(str(dist_path))
    print(f"  KPI distributions: {dist_path}")
    
    print()
    
    # =========================================================================
    # Step 3: Symbol Dashboards
    # =========================================================================
    print("Step 3: Creating symbol dashboards...")
    
    symbol_dir = PLOTS_DIR / "symbol_dashboards"
    os.makedirs(symbol_dir, exist_ok=True)
    
    symbols = symbol_metrics["symbol"].unique()
    created = 0
    
    for i, symbol in enumerate(symbols):
        sym_metrics = symbol_metrics[symbol_metrics["symbol"] == symbol].iloc[0]
        fig = create_symbol_dashboard(symbol, predictions_df, sym_metrics)
        
        if fig is not None:
            fig.write_html(str(symbol_dir / f"{symbol}.html"))
            created += 1
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(symbols)}")
    
    print(f"  Created {created} symbol dashboards")
    print()
    
    # =========================================================================
    # Step 4: Inspector
    # =========================================================================
    print("Step 4: Creating inspector...")
    
    inspector_html = create_inspector(metrics_df, symbol_metrics)
    inspector_path = PLOTS_DIR / "_inspector.html"
    with open(inspector_path, "w") as f:
        f.write(inspector_html)
    print(f"  Inspector: {inspector_path}")
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print(f"Open: {inspector_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

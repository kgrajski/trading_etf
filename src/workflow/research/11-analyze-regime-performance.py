#!/usr/bin/env python3
"""
11-analyze-regime-performance.py

Post-hoc analysis of rolling study results stratified by market regime.

This script:
1. Loads weekly metrics from a rolling study (e.g., exp009)
2. Computes regime features for each week (VIX, trend, dispersion)
3. Stratifies performance metrics by regime
4. Generates visualizations showing when the signal works
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

# Source experiment to analyze
SOURCE_EXPERIMENT = "exp009_rolling_all_models"
SOURCE_DIR = Path("experiments") / SOURCE_EXPERIMENT

# Output directory
OUTPUT_DIR = SOURCE_DIR / "regime_analysis"

# Macro data for regime computation
MACRO_DATA_DIR = Path("data/historical/iex/weekly")

# Regime definitions
VIX_TERCILES = [0.33, 0.67]  # Low/Medium/High
TREND_LOOKBACK = 4  # weeks for momentum calculation


# =============================================================================
# Regime Computation Functions
# =============================================================================

def load_macro_weekly_data() -> Dict[str, pd.DataFrame]:
    """Load weekly data for macro symbols used in regime computation."""
    macro_symbols = {
        "VIXY": "VIX proxy (short-term)",
        "SPY": "S&P 500 proxy",
    }
    
    data = {}
    for symbol in macro_symbols:
        path = MACRO_DATA_DIR / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["week_start"])
            df = df.sort_values("week_start").reset_index(drop=True)
            data[symbol] = df
    
    return data


def compute_regime_features(
    weekly_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    macro_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add regime features to weekly metrics.
    
    Regimes computed:
    1. VIX level tercile (low/medium/high volatility environment)
    2. Market trend (SPY 4-week momentum: up/flat/down)
    3. Cross-sectional dispersion (std of ETF returns)
    4. Market breadth (% of ETFs with positive return)
    """
    df = weekly_metrics.copy()
    
    # Parse week_start if needed
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"])
    
    # --- VIX Regime ---
    if "VIXY" in macro_data:
        vixy = macro_data["VIXY"][["week_start", "close"]].copy()
        vixy = vixy.rename(columns={"close": "vixy_close"})
        df = df.merge(vixy, on="week_start", how="left")
        
        # Compute VIX terciles based on rolling history (causal)
        df["vixy_rank"] = df["vixy_close"].expanding().apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 10 else 0.5
        )
        df["vix_regime"] = pd.cut(
            df["vixy_rank"], 
            bins=[-0.01, 0.33, 0.67, 1.01],
            labels=["low", "medium", "high"]
        )
    else:
        df["vix_regime"] = "unknown"
    
    # --- Market Trend Regime ---
    if "SPY" in macro_data:
        spy = macro_data["SPY"][["week_start", "close"]].copy()
        spy = spy.rename(columns={"close": "spy_close"})
        spy["spy_momentum"] = spy["spy_close"].pct_change(TREND_LOOKBACK)
        df = df.merge(spy[["week_start", "spy_momentum"]], on="week_start", how="left")
        
        # Trend regime based on momentum
        df["trend_regime"] = pd.cut(
            df["spy_momentum"],
            bins=[-np.inf, -0.02, 0.02, np.inf],
            labels=["down", "flat", "up"]
        )
    else:
        df["trend_regime"] = "unknown"
    
    # --- Cross-sectional Dispersion ---
    # Compute from predictions data
    week_dispersion = predictions.groupby("week_idx")["alpha"].std().reset_index()
    week_dispersion = week_dispersion.rename(columns={"alpha": "cross_sectional_std"})
    df = df.merge(week_dispersion, on="week_idx", how="left")
    
    # Terciles of dispersion
    df["dispersion_rank"] = df["cross_sectional_std"].expanding().apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 10 else 0.5
    )
    df["dispersion_regime"] = pd.cut(
        df["dispersion_rank"],
        bins=[-0.01, 0.33, 0.67, 1.01],
        labels=["low", "medium", "high"]
    )
    
    # --- Market Breadth ---
    week_breadth = predictions.groupby("week_idx").apply(
        lambda g: (g["alpha"] > 0).mean(),
        include_groups=False
    ).reset_index()
    week_breadth.columns = ["week_idx", "breadth"]
    df = df.merge(week_breadth, on="week_idx", how="left")
    
    df["breadth_regime"] = pd.cut(
        df["breadth"],
        bins=[-0.01, 0.4, 0.6, 1.01],
        labels=["bearish", "neutral", "bullish"]
    )
    
    return df


# =============================================================================
# Analysis Functions
# =============================================================================

def stratify_by_regime(
    df: pd.DataFrame,
    regime_col: str,
    metric_cols: List[str] = ["directional_accuracy", "ic", "r2"],
) -> pd.DataFrame:
    """Compute aggregate metrics stratified by regime."""
    
    results = []
    for regime in df[regime_col].dropna().unique():
        subset = df[df[regime_col] == regime]
        row = {
            "regime": regime,
            "n_weeks": len(subset),
            "pct_of_total": len(subset) / len(df),
        }
        for col in metric_cols:
            if col in subset.columns:
                row[f"mean_{col}"] = subset[col].mean()
                row[f"std_{col}"] = subset[col].std()
        results.append(row)
    
    return pd.DataFrame(results).sort_values("regime")


def create_regime_performance_plot(
    df: pd.DataFrame,
    regime_col: str,
    title: str,
) -> go.Figure:
    """Create bar chart of performance by regime."""
    
    stats = stratify_by_regime(df, regime_col)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Directional Accuracy", "Information Coefficient", "# Weeks"],
        horizontal_spacing=0.1,
    )
    
    colors = {"low": "#27ae60", "medium": "#f39c12", "high": "#e74c3c",
              "down": "#e74c3c", "flat": "#f39c12", "up": "#27ae60",
              "bearish": "#e74c3c", "neutral": "#f39c12", "bullish": "#27ae60"}
    
    regimes = stats["regime"].tolist()
    
    # Directional Accuracy
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=stats["mean_directional_accuracy"],
            error_y=dict(type="data", array=stats["std_directional_accuracy"]),
            marker_color=[colors.get(r, "#3498db") for r in regimes],
            name="Dir. Acc",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
    
    # IC
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=stats["mean_ic"],
            error_y=dict(type="data", array=stats["std_ic"]),
            marker_color=[colors.get(r, "#3498db") for r in regimes],
            name="IC",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # N weeks
    fig.add_trace(
        go.Bar(
            x=regimes,
            y=stats["n_weeks"],
            marker_color=[colors.get(r, "#3498db") for r in regimes],
            name="Weeks",
            showlegend=False,
        ),
        row=1, col=3,
    )
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=400,
    )
    
    fig.update_yaxes(title_text="Dir. Accuracy", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="IC", row=1, col=2)
    fig.update_yaxes(title_text="# Weeks", row=1, col=3)
    
    return fig


def create_summary_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create heatmap of IC by two regime dimensions."""
    
    # VIX vs Trend
    pivot = df.pivot_table(
        values="ic",
        index="vix_regime",
        columns="trend_regime",
        aggfunc="mean",
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="VIX: %{y}<br>Trend: %{x}<br>IC: %{z:.3f}<extra></extra>",
    ))
    
    fig.update_layout(
        title=dict(text="Mean IC by VIX Level × Market Trend", x=0.5),
        xaxis_title="Market Trend (SPY 4-week)",
        yaxis_title="VIX Level",
        height=400,
        width=500,
    )
    
    return fig


def generate_report(
    df: pd.DataFrame,
    regime_stats: Dict[str, pd.DataFrame],
) -> str:
    """Generate markdown report."""
    
    lines = []
    lines.append("# Regime-Stratified Performance Analysis")
    lines.append("")
    lines.append(f"**Source:** {SOURCE_EXPERIMENT}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overall baseline
    lines.append("## Baseline (All Weeks)")
    lines.append("")
    lines.append(f"- Weeks analyzed: {len(df)}")
    lines.append(f"- Mean Dir. Accuracy: {df['directional_accuracy'].mean():.1%}")
    lines.append(f"- Mean IC: {df['ic'].mean():.3f}")
    lines.append("")
    
    # Each regime dimension
    for regime_name, stats in regime_stats.items():
        lines.append(f"## Performance by {regime_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Regime | Weeks | Dir. Acc | IC |")
        lines.append("|--------|-------|----------|-----|")
        for _, row in stats.iterrows():
            lines.append(
                f"| {row['regime']} | {row['n_weeks']} ({row['pct_of_total']:.0%}) | "
                f"{row['mean_directional_accuracy']:.1%} | {row['mean_ic']:.3f} |"
            )
        lines.append("")
    
    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    
    # Find best/worst regimes
    for regime_name, stats in regime_stats.items():
        if len(stats) > 1:
            best = stats.loc[stats["mean_ic"].idxmax()]
            worst = stats.loc[stats["mean_ic"].idxmin()]
            if best["mean_ic"] - worst["mean_ic"] > 0.05:
                lines.append(
                    f"- **{regime_name.replace('_', ' ').title()}**: "
                    f"Best in '{best['regime']}' (IC={best['mean_ic']:.3f}), "
                    f"Worst in '{worst['regime']}' (IC={worst['mean_ic']:.3f})"
                )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `11-analyze-regime-performance.py`*")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================

@workflow_script("11-analyze-regime-performance")
def main() -> None:
    """Run regime-stratified analysis."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print(f"REGIME ANALYSIS: {SOURCE_EXPERIMENT}")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Step 1: Loading data...")
    
    weekly_metrics = pd.read_csv(SOURCE_DIR / "weekly_metrics.csv")
    predictions = pd.read_parquet(SOURCE_DIR / "weekly_predictions.parquet")
    
    print(f"  Weekly metrics: {len(weekly_metrics)} weeks")
    print(f"  Predictions: {len(predictions):,} rows")
    print()
    
    # =========================================================================
    # Step 2: Load Macro Data for Regime Computation
    # =========================================================================
    print("Step 2: Loading macro data...")
    
    macro_data = load_macro_weekly_data()
    print(f"  Loaded: {list(macro_data.keys())}")
    print()
    
    # =========================================================================
    # Step 3: Compute Regime Features
    # =========================================================================
    print("Step 3: Computing regime features...")
    
    df = compute_regime_features(weekly_metrics, predictions, macro_data)
    
    print(f"  VIX regimes: {df['vix_regime'].value_counts().to_dict()}")
    print(f"  Trend regimes: {df['trend_regime'].value_counts().to_dict()}")
    print(f"  Dispersion regimes: {df['dispersion_regime'].value_counts().to_dict()}")
    print(f"  Breadth regimes: {df['breadth_regime'].value_counts().to_dict()}")
    print()
    
    # =========================================================================
    # Step 4: Stratified Analysis
    # =========================================================================
    print("Step 4: Stratifying performance by regime...")
    
    regime_cols = ["vix_regime", "trend_regime", "dispersion_regime", "breadth_regime"]
    regime_stats = {}
    
    for col in regime_cols:
        stats = stratify_by_regime(df, col)
        regime_stats[col] = stats
        print(f"\n  {col}:")
        for _, row in stats.iterrows():
            print(f"    {row['regime']}: IC={row['mean_ic']:.3f}, Dir.Acc={row['mean_directional_accuracy']:.1%} (n={row['n_weeks']})")
    
    print()
    
    # =========================================================================
    # Step 5: Generate Visualizations
    # =========================================================================
    print("Step 5: Generating visualizations...")
    
    # Performance by each regime
    for col in regime_cols:
        fig = create_regime_performance_plot(df, col, f"Performance by {col.replace('_', ' ').title()}")
        path = OUTPUT_DIR / f"performance_by_{col}.html"
        fig.write_html(str(path))
        print(f"  {path}")
    
    # Heatmap
    fig_heatmap = create_summary_heatmap(df)
    heatmap_path = OUTPUT_DIR / "ic_heatmap_vix_trend.html"
    fig_heatmap.write_html(str(heatmap_path))
    print(f"  {heatmap_path}")
    
    print()
    
    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    print("Step 6: Saving results...")
    
    # Enhanced metrics with regimes
    df.to_csv(OUTPUT_DIR / "weekly_metrics_with_regimes.csv", index=False)
    print(f"  {OUTPUT_DIR / 'weekly_metrics_with_regimes.csv'}")
    
    # Regime stats
    for col, stats in regime_stats.items():
        stats.to_csv(OUTPUT_DIR / f"stats_by_{col}.csv", index=False)
    
    # Report
    report = generate_report(df, regime_stats)
    report_path = OUTPUT_DIR / "regime_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  {report_path}")
    
    print()
    print("=" * 80)
    print("COMPLETE")
    print(f"Open: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()

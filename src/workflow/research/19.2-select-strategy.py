#!/usr/bin/env python3
"""
19.2-select-strategy.py

Strategy Selector with Visual Exploration.

This script analyzes grid search results from 19.1b and helps identify
the best strategies using intelligent multi-criteria filtering rather
than simply picking max return.

Features:
- Distribution histograms (Return, Sortino, Sharpe, Top-5 Concentration)
- Scatter plot explorer (Return vs Sortino, Return vs Concentration)
- Filter funnel visualization
- Top configs table with full metrics
- Interactive HTML dashboard

Can read from:
1. Final grid_results.csv (after 19.1b completes)
2. Partial Ray trial folders (while 19.1b is running)

Outputs:
- experiments/exp019_2_selector/
  - explorer.html
  - top_configs.csv
  - selection_summary.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.workflow_utils import PROJECT_ROOT, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp019_2_selector"

# Input sources
GRID_RESULTS_CSV = PROJECT_ROOT / "experiments" / "exp019_1b_grid" / "grid_results.csv"
RAY_RESULTS_DIR = Path.home() / "ray_results" / "exp019_1b_grid"

# Output
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Selection filters (configurable)
FILTERS = {
    "is_degraded_max": 0,           # Must be stable (0 = not degraded)
    "top5_concentration_max": 50.0, # Not lottery-dependent (< 50%)
    "recent_max_hold_pnl_min": 0.0, # Mean-reversion thesis still works
    "profit_factor_min": 1.0,       # Wins must outpace losses
    "win_rate_min": 50.0,           # More winners than losers
}

# Ranking
RANK_BY = "sortino_overall"  # Primary ranking metric after filtering
TOP_N = 20  # Show top N configs


# =============================================================================
# Data Loading
# =============================================================================

def load_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load results from final CSV."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_from_ray_trials(ray_dir: Path, trial_prefix: str = "ray_tune_trainable_63e46") -> pd.DataFrame:
    """Load partial results from Ray trial folders."""
    if not ray_dir.exists():
        return pd.DataFrame()
    
    rows = []
    trial_dirs = [d for d in ray_dir.iterdir() if d.is_dir() and trial_prefix in d.name]
    
    print(f"  Found {len(trial_dirs)} trial folders...")
    
    for trial_dir in trial_dirs:
        params_file = trial_dir / "params.json"
        result_file = trial_dir / "result.json"
        
        if not params_file.exists() or not result_file.exists():
            continue
        
        try:
            with open(params_file) as f:
                params = json.load(f)
            with open(result_file) as f:
                result = json.load(f)
            
            # Extract config params
            row = {
                "stop_loss_pcnt": params.get("stop_loss_pcnt"),
                "profit_exit_pcnt": params.get("profit_exit_pcnt"),
                "max_hold_weeks": params.get("max_hold_weeks"),
                "boost_direction": params.get("boost_direction"),
                "min_loss_pcnt": params.get("min_loss_pcnt"),
            }
            
            # Extract metrics from result
            for key in ["total_trades", "winning_trades", "losing_trades", "win_rate",
                        "total_pnl", "total_return_pcnt", "max_drawdown_pcnt",
                        "sharpe_overall", "sortino_overall", "top5_concentration",
                        "profit_factor", "sharpe_recent", "sortino_recent",
                        "top5_concentration_recent", "profit_factor_recent",
                        "bull_weeks", "bear_weeks", "bull_trades", "bear_trades",
                        "bull_pnl", "bear_pnl", "prior_trades", "recent_trades",
                        "prior_win_rate", "recent_win_rate", "win_rate_delta",
                        "prior_pnl", "recent_pnl", "prior_max_hold_pnl",
                        "recent_max_hold_pnl", "is_degraded"]:
                row[key] = result.get(key)
            
            rows.append(row)
        except (json.JSONDecodeError, KeyError):
            continue
    
    return pd.DataFrame(rows)


def load_results() -> pd.DataFrame:
    """Load results from best available source."""
    # Try CSV first (final results)
    if GRID_RESULTS_CSV.exists():
        print(f"Loading from CSV: {GRID_RESULTS_CSV}")
        df = load_from_csv(GRID_RESULTS_CSV)
        if len(df) > 0:
            return df
    
    # Fall back to Ray trials (partial results)
    print(f"Loading from Ray trials: {RAY_RESULTS_DIR}")
    return load_from_ray_trials(RAY_RESULTS_DIR)


# =============================================================================
# Filtering
# =============================================================================

def apply_filters(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Apply filters progressively, returning stats at each stage."""
    stages = {"all": df.copy()}
    current = df.copy()
    
    # Filter 1: Not degraded
    if "is_degraded" in current.columns:
        current = current[current["is_degraded"] <= FILTERS["is_degraded_max"]]
    stages["stable"] = current.copy()
    
    # Filter 2: Low concentration (not lottery)
    if "top5_concentration" in current.columns:
        current = current[current["top5_concentration"] <= FILTERS["top5_concentration_max"]]
    stages["diversified"] = current.copy()
    
    # Filter 3: Recent max_hold positive
    if "recent_max_hold_pnl" in current.columns:
        current = current[current["recent_max_hold_pnl"] >= FILTERS["recent_max_hold_pnl_min"]]
    stages["thesis_works"] = current.copy()
    
    # Filter 4: Profit factor
    if "profit_factor" in current.columns:
        current = current[current["profit_factor"] >= FILTERS["profit_factor_min"]]
    stages["profitable"] = current.copy()
    
    # Filter 5: Win rate
    if "win_rate" in current.columns:
        current = current[current["win_rate"] >= FILTERS["win_rate_min"]]
    stages["final"] = current.copy()
    
    return stages


# =============================================================================
# Visualization
# =============================================================================

def create_distribution_panel(df: pd.DataFrame, top_configs: pd.DataFrame) -> go.Figure:
    """Create 4-panel histogram showing distributions."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Total Return % Distribution",
            "Sortino Ratio Distribution",
            "Top-5 Concentration % Distribution",
            "Profit Factor Distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    metrics = [
        ("total_return_pcnt", 1, 1, "#2196F3"),
        ("sortino_overall", 1, 2, "#4CAF50"),
        ("top5_concentration", 2, 1, "#FF9800"),
        ("profit_factor", 2, 2, "#9C27B0"),
    ]
    
    for metric, row, col, color in metrics:
        if metric not in df.columns:
            continue
        
        values = df[metric].dropna()
        
        # Main histogram
        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=50,
                marker_color=color,
                opacity=0.7,
                name=metric,
                showlegend=False,
            ),
            row=row, col=col,
        )
        
        # Mark top configs
        if len(top_configs) > 0 and metric in top_configs.columns:
            top_values = top_configs[metric].dropna()
            for i, val in enumerate(top_values[:5]):
                fig.add_vline(
                    x=val, row=row, col=col,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"#{i+1}" if i == 0 else None,
                )
    
    fig.update_layout(
        title="<b>Strategy Distribution Overview</b><br><sub>Red lines = Top 5 filtered configs</sub>",
        height=600,
        showlegend=False,
    )
    
    return fig


def create_scatter_explorer(df: pd.DataFrame, top_configs: pd.DataFrame) -> go.Figure:
    """Create scatter plot explorer."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Return vs Sortino (colored by stability)",
            "Return vs Top-5 Concentration",
        ),
        horizontal_spacing=0.1,
    )
    
    if "total_return_pcnt" not in df.columns:
        return fig
    
    # Create config label for hover
    df = df.copy()
    df["config_label"] = df.apply(
        lambda r: f"SL={r.get('stop_loss_pcnt', '?')}% TP={r.get('profit_exit_pcnt', '?')}% "
                  f"H={r.get('max_hold_weeks', '?')}w {r.get('boost_direction', '?')} "
                  f"MinL={r.get('min_loss_pcnt', '?')}%",
        axis=1
    )
    
    # Scatter 1: Return vs Sortino
    if "sortino_overall" in df.columns:
        # Stable configs
        stable = df[df.get("is_degraded", 1) == 0] if "is_degraded" in df.columns else df
        degraded = df[df.get("is_degraded", 0) == 1] if "is_degraded" in df.columns else pd.DataFrame()
        
        if len(stable) > 0:
            fig.add_trace(
                go.Scatter(
                    x=stable["total_return_pcnt"],
                    y=stable["sortino_overall"],
                    mode="markers",
                    marker=dict(color="#4CAF50", size=6, opacity=0.6),
                    name="Stable",
                    text=stable["config_label"],
                    hovertemplate="<b>%{text}</b><br>Return: %{x:.1f}%<br>Sortino: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
        
        if len(degraded) > 0:
            fig.add_trace(
                go.Scatter(
                    x=degraded["total_return_pcnt"],
                    y=degraded["sortino_overall"],
                    mode="markers",
                    marker=dict(color="#F44336", size=6, opacity=0.6),
                    name="Degraded",
                    text=degraded["config_label"],
                    hovertemplate="<b>%{text}</b><br>Return: %{x:.1f}%<br>Sortino: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
        
        # Highlight top configs
        if len(top_configs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=top_configs["total_return_pcnt"],
                    y=top_configs["sortino_overall"],
                    mode="markers",
                    marker=dict(color="gold", size=12, symbol="star", line=dict(color="black", width=1)),
                    name="Top Picks",
                    text=top_configs["config_label"] if "config_label" in top_configs.columns else None,
                    hovertemplate="<b>★ TOP PICK</b><br>%{text}<br>Return: %{x:.1f}%<br>Sortino: %{y:.2f}<extra></extra>",
                ),
                row=1, col=1,
            )
    
    # Scatter 2: Return vs Concentration
    if "top5_concentration" in df.columns:
        colors = df["is_degraded"].map({0: "#4CAF50", 1: "#F44336"}) if "is_degraded" in df.columns else "#2196F3"
        
        fig.add_trace(
            go.Scatter(
                x=df["total_return_pcnt"],
                y=df["top5_concentration"],
                mode="markers",
                marker=dict(color=colors, size=6, opacity=0.6),
                name="All Configs",
                text=df["config_label"],
                hovertemplate="<b>%{text}</b><br>Return: %{x:.1f}%<br>Top-5 Conc: %{y:.1f}%<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )
        
        # Add filter threshold line
        fig.add_hline(
            y=FILTERS["top5_concentration_max"], row=1, col=2,
            line=dict(color="orange", width=2, dash="dash"),
            annotation_text=f"Filter: <{FILTERS['top5_concentration_max']}%",
        )
        
        # Highlight top configs
        if len(top_configs) > 0 and "top5_concentration" in top_configs.columns:
            fig.add_trace(
                go.Scatter(
                    x=top_configs["total_return_pcnt"],
                    y=top_configs["top5_concentration"],
                    mode="markers",
                    marker=dict(color="gold", size=12, symbol="star", line=dict(color="black", width=1)),
                    name="Top Picks",
                    showlegend=False,
                ),
                row=1, col=2,
            )
    
    fig.update_xaxes(title_text="Total Return %", row=1, col=1)
    fig.update_yaxes(title_text="Sortino Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Total Return %", row=1, col=2)
    fig.update_yaxes(title_text="Top-5 Concentration %", row=1, col=2)
    
    fig.update_layout(
        title="<b>Scatter Plot Explorer</b><br><sub>Hover for config details | Stars = Top filtered picks</sub>",
        height=500,
    )
    
    return fig


def create_filter_funnel(stages: Dict[str, pd.DataFrame]) -> go.Figure:
    """Create filter funnel visualization."""
    stage_names = ["All Configs", "Stable (not degraded)", "Diversified (<50% conc)", 
                   "Thesis Works (max_hold>0)", "Profitable (PF>1)", "Final (WR>50%)"]
    stage_keys = ["all", "stable", "diversified", "thesis_works", "profitable", "final"]
    
    counts = [len(stages.get(k, pd.DataFrame())) for k in stage_keys]
    
    fig = go.Figure(go.Funnel(
        y=stage_names,
        x=counts,
        textinfo="value+percent initial",
        marker=dict(color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]),
    ))
    
    fig.update_layout(
        title="<b>Filter Funnel</b><br><sub>How many configs pass each filter</sub>",
        height=400,
    )
    
    return fig


def create_top_configs_table(top_configs: pd.DataFrame) -> str:
    """Create HTML table of top configs."""
    if len(top_configs) == 0:
        return "<p>No configs passed all filters.</p>"
    
    # Select columns to display
    display_cols = [
        "stop_loss_pcnt", "profit_exit_pcnt", "max_hold_weeks", "boost_direction", "min_loss_pcnt",
        "total_return_pcnt", "sortino_overall", "sharpe_overall", "top5_concentration",
        "profit_factor", "win_rate", "recent_max_hold_pnl", "is_degraded",
    ]
    
    available_cols = [c for c in display_cols if c in top_configs.columns]
    display_df = top_configs[available_cols].head(TOP_N)
    
    # Format columns
    format_map = {
        "total_return_pcnt": lambda x: f"{x:.1f}%",
        "sortino_overall": lambda x: f"{x:.2f}",
        "sharpe_overall": lambda x: f"{x:.2f}",
        "top5_concentration": lambda x: f"{x:.1f}%",
        "profit_factor": lambda x: f"{x:.2f}",
        "win_rate": lambda x: f"{x:.1f}%",
        "recent_max_hold_pnl": lambda x: f"${x:,.0f}",
        "stop_loss_pcnt": lambda x: f"{x:.0f}%",
        "profit_exit_pcnt": lambda x: f"{x:.0f}%",
        "max_hold_weeks": lambda x: f"{x:.0f}w",
        "min_loss_pcnt": lambda x: f"{x:.0f}%",
    }
    
    for col, fmt in format_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(fmt)
    
    # Rename columns for display
    rename_map = {
        "stop_loss_pcnt": "SL",
        "profit_exit_pcnt": "TP",
        "max_hold_weeks": "Hold",
        "boost_direction": "Boost",
        "min_loss_pcnt": "MinL",
        "total_return_pcnt": "Return",
        "sortino_overall": "Sortino",
        "sharpe_overall": "Sharpe",
        "top5_concentration": "Top5%",
        "profit_factor": "PF",
        "win_rate": "WinRate",
        "recent_max_hold_pnl": "Recent MH P&L",
        "is_degraded": "Degraded",
    }
    
    display_df = display_df.rename(columns=rename_map)
    
    # Convert to HTML
    html = display_df.to_html(index=False, classes="config-table", escape=False)
    
    return html


def create_explorer_dashboard(
    df: pd.DataFrame,
    stages: Dict[str, pd.DataFrame],
    top_configs: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Create interactive HTML explorer dashboard."""
    
    # Create figures
    dist_fig = create_distribution_panel(df, top_configs)
    scatter_fig = create_scatter_explorer(df, top_configs)
    funnel_fig = create_filter_funnel(stages)
    top_table = create_top_configs_table(top_configs)
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Selector - {EXPERIMENT_NAME}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
            margin: 20px; 
            background: #f5f5f5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .card {{ 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            margin: 20px 0; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .filters-box {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .filters-box code {{
            background: #bbdefb;
            padding: 2px 6px;
            border-radius: 4px;
        }}
        .config-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .config-table th {{
            background: #2196F3;
            color: white;
            padding: 12px 8px;
            text-align: left;
        }}
        .config-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
        }}
        .config-table tr:hover {{
            background: #f5f5f5;
        }}
        .config-table tr:nth-child(1) {{
            background: #fff9c4;
            font-weight: bold;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #2196F3;
        }}
        .stat-box .number {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-box .label {{ font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Strategy Selector</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="number">{len(df):,}</div>
                <div class="label">Total Configs</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(stages.get('stable', [])):,}</div>
                <div class="label">Stable (not degraded)</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(stages.get('final', [])):,}</div>
                <div class="label">Passed All Filters</div>
            </div>
            <div class="stat-box">
                <div class="number">{len(top_configs):,}</div>
                <div class="label">Top Candidates</div>
            </div>
        </div>
        
        <div class="filters-box">
            <strong>Active Filters:</strong><br>
            <code>is_degraded = 0</code> (stable) |
            <code>top5_concentration &lt; {FILTERS['top5_concentration_max']}%</code> (diversified) |
            <code>recent_max_hold_pnl &gt; ${FILTERS['recent_max_hold_pnl_min']}</code> (thesis works) |
            <code>profit_factor &gt; {FILTERS['profit_factor_min']}</code> |
            <code>win_rate &gt; {FILTERS['win_rate_min']}%</code>
            <br><br>
            <strong>Ranked by:</strong> <code>{RANK_BY}</code> (descending)
        </div>
        
        <div class="card">
            <h2>🏆 Top {TOP_N} Strategies (After Filtering)</h2>
            {top_table}
        </div>
        
        <div class="card">
            <h2>📊 Distribution Overview</h2>
            {dist_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <div class="card">
            <h2>🔍 Scatter Plot Explorer</h2>
            {scatter_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
        
        <div class="card">
            <h2>🔽 Filter Funnel</h2>
            {funnel_fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </div>
</body>
</html>"""
    
    output_path = output_dir / "explorer.html"
    output_path.write_text(html_content)
    return output_path


# =============================================================================
# Main
# =============================================================================

@workflow_script("19.2-select-strategy")
def main():
    """Run strategy selector."""
    
    print("=" * 70)
    print("19.2 STRATEGY SELECTOR")
    print("=" * 70)
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading results...")
    df = load_results()
    
    if len(df) == 0:
        print("❌ No results found. Run 19.1b first.")
        return
    
    print(f"  Loaded {len(df):,} configurations")
    print()
    
    # Apply filters
    print("Applying filters...")
    stages = apply_filters(df)
    
    for stage_name, stage_df in stages.items():
        print(f"  {stage_name}: {len(stage_df):,} configs")
    print()
    
    # Get final filtered set and rank
    final_df = stages["final"]
    
    if len(final_df) == 0:
        print("⚠️  No configs passed all filters. Relaxing filters...")
        # Fall back to just stable configs
        final_df = stages.get("stable", df)
    
    # Rank by chosen metric
    if RANK_BY in final_df.columns:
        top_configs = final_df.nlargest(TOP_N, RANK_BY)
    else:
        top_configs = final_df.head(TOP_N)
    
    # Print top configs
    print(f"Top {len(top_configs)} configs by {RANK_BY}:")
    print("-" * 70)
    for i, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(f"  #{i}: SL={row.get('stop_loss_pcnt', '?')}% TP={row.get('profit_exit_pcnt', '?')}% "
              f"Hold={row.get('max_hold_weeks', '?')}w Boost={row.get('boost_direction', '?')} "
              f"MinL={row.get('min_loss_pcnt', '?')}%")
        print(f"      Return={row.get('total_return_pcnt', 0):.1f}% Sortino={row.get('sortino_overall', 0):.2f} "
              f"Top5={row.get('top5_concentration', 0):.1f}% PF={row.get('profit_factor', 0):.2f}")
    print()
    
    # Create dashboard
    print("Creating explorer dashboard...")
    dashboard_path = create_explorer_dashboard(df, stages, top_configs, OUTPUT_DIR)
    
    # Save top configs
    top_configs.to_csv(OUTPUT_DIR / "top_configs.csv", index=False)
    
    # Save summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_configs": len(df),
        "filters": FILTERS,
        "rank_by": RANK_BY,
        "stages": {k: len(v) for k, v in stages.items()},
        "champion": top_configs.iloc[0].to_dict() if len(top_configs) > 0 else None,
    }
    
    with open(OUTPUT_DIR / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Dashboard: {dashboard_path}")
    print(f"  Top Configs: {OUTPUT_DIR / 'top_configs.csv'}")
    print(f"  Summary: {OUTPUT_DIR / 'selection_summary.json'}")
    
    if len(top_configs) > 0:
        champ = top_configs.iloc[0]
        print()
        print("🏆 CHAMPION CONFIG:")
        print(f"   SL={champ.get('stop_loss_pcnt', '?')}% TP={champ.get('profit_exit_pcnt', '?')}% "
              f"Hold={champ.get('max_hold_weeks', '?')}w Boost={champ.get('boost_direction', '?')} "
              f"MinL={champ.get('min_loss_pcnt', '?')}%")


if __name__ == "__main__":
    main()

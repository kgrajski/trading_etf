#!/usr/bin/env python3
"""
20b-analyze-robustness.py

Analyze and visualize results from robustness check (script 20).

Generates:
- Equity curve overlay (colored by start regime)
- Return distribution by config and start regime
- Summary statistics table
- Regime sensitivity analysis

Inputs:
- experiments/exp020_robustness_{source}/robustness_results.csv
- experiments/exp020_robustness_{source}/equity_curves.csv

Outputs:
- experiments/exp020_robustness_{source}/robustness_dashboard.html
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.workflow_utils import PROJECT_ROOT

# =============================================================================
# Configuration
# =============================================================================

# Source experiment to analyze (must match what was run in script 20)
SOURCE_EXPERIMENT = "exp019_grid_regime"

# Colors
COLORS = {
    "bull": "#2ecc71",  # Green
    "bear": "#e74c3c",  # Red
    "neutral": "#95a5a6",  # Gray
    "best_return": "#3498db",  # Blue
    "best_risk_adjusted": "#9b59b6",  # Purple
    "best_drawdown": "#f39c12",  # Orange
}


def load_data(source_experiment: str) -> Dict[str, Any]:
    """Load robustness check results."""
    experiment_name = f"exp020_robustness_{source_experiment.replace('exp019_', '')}"
    experiment_dir = PROJECT_ROOT / "experiments" / experiment_name
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_dir}")
    
    results_df = pd.read_csv(experiment_dir / "robustness_results.csv")
    equity_df = pd.read_csv(experiment_dir / "equity_curves.csv")
    
    with open(experiment_dir / "run_metadata.json") as f:
        metadata = json.load(f)
    
    return {
        "results": results_df,
        "equity": equity_df,
        "metadata": metadata,
        "experiment_dir": experiment_dir,
    }


def create_equity_overlay(equity_df: pd.DataFrame, results_df: pd.DataFrame) -> go.Figure:
    """Create equity curve overlay plot with toggleable legend groups."""
    
    fig = go.Figure()
    
    initial_capital = 10000
    
    # Track which legend groups we've added
    legend_shown = set()
    
    # Add all individual runs grouped by config + regime
    for config_name in results_df["config_name"].unique():
        for start_regime in ["bull", "bear"]:
            # Get runs for this config + regime combination
            mask = (results_df["config_name"] == config_name) & (results_df["start_regime"] == start_regime)
            config_regime_runs = results_df[mask]["run_id"].unique()
            
            legend_group = f"{config_name}_{start_regime}"
            color = COLORS.get(start_regime, COLORS["neutral"])
            
            for i, run_id in enumerate(config_regime_runs):
                run_equity = equity_df[equity_df["run_id"] == run_id].sort_values("week")
                
                if len(run_equity) == 0:
                    continue
                    
                run_result = results_df[results_df["run_id"] == run_id].iloc[0]
                
                # Calculate return percentage
                returns_pct = ((run_equity["capital"].values - initial_capital) / initial_capital) * 100
                weeks = run_equity["week"].values
                
                # Show legend only for first trace in each group
                show_legend = legend_group not in legend_shown
                if show_legend:
                    legend_shown.add(legend_group)
                
                fig.add_trace(go.Scatter(
                    x=weeks.tolist(),
                    y=returns_pct.tolist(),
                    mode="lines",
                    name=f"{config_name} ({start_regime} starts)",
                    line=dict(color=color, width=1.5),
                    opacity=0.5,
                    legendgroup=legend_group,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{run_id}</b><br>"
                        f"Week: %{{x}}<br>"
                        f"Return: %{{y:.1f}}%<br>"
                        f"Start: {run_result['start_date']}<br>"
                        f"Regime: {start_regime}<br>"
                        "<extra></extra>"
                    ),
                ))
    
    # Add median line for each config (on top)
    for config_name in results_df["config_name"].unique():
        config_runs = results_df[results_df["config_name"] == config_name]["run_id"].unique()
        
        # Collect all weekly returns
        weeks = sorted(equity_df["week"].unique())
        median_returns = []
        
        for week in weeks:
            week_data = []
            for run_id in config_runs:
                run_equity = equity_df[(equity_df["run_id"] == run_id) & (equity_df["week"] == week)]
                if len(run_equity) > 0:
                    capital = run_equity["capital"].iloc[0]
                    week_data.append(((capital - initial_capital) / initial_capital) * 100)
            
            if week_data:
                median_returns.append(np.median(week_data))
            else:
                median_returns.append(np.nan)
        
        config_color = COLORS.get(config_name, "#000000")
        
        fig.add_trace(go.Scatter(
            x=list(weeks),
            y=median_returns,
            mode="lines",
            name=f"{config_name} (median)",
            line=dict(color=config_color, width=3),
            legendgroup=f"{config_name}_median",
        ))
    
    fig.update_layout(
        title="Equity Curves by Starting Week (Click legend to toggle)",
        xaxis_title="Week",
        yaxis_title="Cumulative Return (%)",
        height=550,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig


def create_return_distribution(results_df: pd.DataFrame) -> go.Figure:
    """Create box plot of return distribution by config and start regime."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["By Configuration", "By Starting Regime"],
        horizontal_spacing=0.15,
    )
    
    # By configuration
    for i, config_name in enumerate(results_df["config_name"].unique()):
        config_data = results_df[results_df["config_name"] == config_name]
        config_color = COLORS.get(config_name, f"hsl({i*120}, 70%, 50%)")
        
        fig.add_trace(
            go.Box(
                y=config_data["total_return_pcnt"].tolist(),
                name=config_name,
                marker_color=config_color,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ),
            row=1, col=1,
        )
    
    # By starting regime
    for regime in ["bull", "bear"]:
        regime_data = results_df[results_df["start_regime"] == regime]
        
        fig.add_trace(
            go.Box(
                y=regime_data["total_return_pcnt"].tolist(),
                name=f"{regime.title()} Start",
                marker_color=COLORS[regime],
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
            ),
            row=1, col=2,
        )
    
    fig.update_layout(
        title="Return Distribution",
        height=400,
        showlegend=False,
    )
    
    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=1, col=2)
    
    return fig


def create_regime_sensitivity(results_df: pd.DataFrame) -> go.Figure:
    """Create scatter plot: % bull weeks during test vs final return."""
    
    fig = go.Figure()
    
    # Calculate bull week percentage
    df = results_df.copy()
    df["bull_weeks_pct"] = df["bull_weeks"] / (df["bull_weeks"] + df["bear_weeks"]) * 100
    
    for config_name in df["config_name"].unique():
        config_data = df[df["config_name"] == config_name]
        config_color = COLORS.get(config_name, "#3498db")
        
        fig.add_trace(go.Scatter(
            x=config_data["bull_weeks_pct"].tolist(),
            y=config_data["total_return_pcnt"].tolist(),
            mode="markers",
            name=config_name,
            marker=dict(
                color=[COLORS.get(r, COLORS["neutral"]) for r in config_data["start_regime"].tolist()],
                size=12,
                line=dict(color=config_color, width=2),
            ),
            text=[f"{row['start_date']} ({row['start_regime']})" for _, row in config_data.iterrows()],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Config: {config_name}<br>"
                "Bull weeks: %{x:.0f}%<br>"
                "Return: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ))
    
    # Add trend line
    x = df["bull_weeks_pct"].values
    y = df["total_return_pcnt"].values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    x_line = np.linspace(x.min(), x.max(), 100)
    fig.add_trace(go.Scatter(
        x=x_line.tolist(),
        y=p(x_line).tolist(),
        mode="lines",
        name="Trend",
        line=dict(color="gray", dash="dash"),
    ))
    
    fig.update_layout(
        title="Regime Sensitivity: Bull Week Exposure vs Return",
        xaxis_title="% of Test Period in Bull Market",
        yaxis_title="Total Return (%)",
        height=450,
    )
    
    return fig


def create_summary_table(results_df: pd.DataFrame) -> go.Figure:
    """Create summary statistics table."""
    
    summary_data = []
    
    for config_name in results_df["config_name"].unique():
        cfg = results_df[results_df["config_name"] == config_name]
        bull_starts = cfg[cfg["start_regime"] == "bull"]
        bear_starts = cfg[cfg["start_regime"] == "bear"]
        
        summary_data.append({
            "Configuration": config_name,
            "N Runs": int(len(cfg)),
            "Avg Return (%)": f"{cfg['total_return_pcnt'].mean():.1f}",
            "Std Return (%)": f"{cfg['total_return_pcnt'].std():.1f}",
            "Min Return (%)": f"{cfg['total_return_pcnt'].min():.1f}",
            "Max Return (%)": f"{cfg['total_return_pcnt'].max():.1f}",
            "Avg Drawdown (%)": f"{cfg['max_drawdown_pcnt'].mean():.1f}",
            "Avg Win Rate (%)": f"{cfg['win_rate'].mean():.1f}",
            "Bull Start Avg (%)": f"{bull_starts['total_return_pcnt'].mean():.1f}" if len(bull_starts) > 0 else "N/A",
            "Bear Start Avg (%)": f"{bear_starts['total_return_pcnt'].mean():.1f}" if len(bear_starts) > 0 else "N/A",
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Convert to list of lists for plotly table
    cell_values = [summary_df[col].tolist() for col in summary_df.columns]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_df.columns),
            fill_color="#3498db",
            font=dict(color="white", size=12),
            align="center",
            height=30,
        ),
        cells=dict(
            values=cell_values,
            fill_color=[["#f8f9fa", "#ecf0f1", "#f8f9fa"]],
            font=dict(size=12),
            align="center",
            height=25,
        ),
    )])
    
    fig.update_layout(
        title="Summary Statistics by Configuration",
        height=180,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    
    return fig


def create_start_regime_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Compare performance by starting regime for each config."""
    
    configs = results_df["config_name"].unique()
    
    fig = make_subplots(
        rows=1, cols=len(configs),
        subplot_titles=configs,
        horizontal_spacing=0.1,
    )
    
    for i, config_name in enumerate(configs, 1):
        config_data = results_df[results_df["config_name"] == config_name]
        
        bull_data = config_data[config_data["start_regime"] == "bull"]["total_return_pcnt"]
        bear_data = config_data[config_data["start_regime"] == "bear"]["total_return_pcnt"]
        
        fig.add_trace(
            go.Bar(
                x=["Bull Start", "Bear Start"],
                y=[bull_data.mean(), bear_data.mean()],
                error_y=dict(
                    type="data",
                    array=[bull_data.std(), bear_data.std()],
                ),
                marker_color=[COLORS["bull"], COLORS["bear"]],
                showlegend=False,
            ),
            row=1, col=i,
        )
    
    fig.update_layout(
        title="Average Return by Starting Regime (with Std Dev)",
        height=350,
    )
    
    for i in range(1, len(configs) + 1):
        fig.update_yaxes(title_text="Avg Return (%)" if i == 1 else "", row=1, col=i)
    
    return fig


def create_dashboard(source_experiment: str) -> Path:
    """Create complete robustness analysis dashboard."""
    
    data = load_data(source_experiment)
    results_df = data["results"]
    equity_df = data["equity"]
    metadata = data["metadata"]
    experiment_dir = data["experiment_dir"]
    
    # Create figures
    equity_fig = create_equity_overlay(equity_df, results_df)
    dist_fig = create_return_distribution(results_df)
    sensitivity_fig = create_regime_sensitivity(results_df)
    summary_fig = create_summary_table(results_df)
    regime_fig = create_start_regime_comparison(results_df)
    
    # Combine into HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robustness Analysis: {metadata['experiment_name']}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #3498db, #2c3e50);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
            }}
            .header h1 {{ margin: 0 0 10px 0; }}
            .header p {{ margin: 5px 0; opacity: 0.9; }}
            .chart-container {{
                background: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .grid-2 {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .insights {{
                background: #ecf0f1;
                padding: 15px;
                border-radius: 10px;
                margin-top: 20px;
            }}
            .insights h3 {{ margin-top: 0; color: #2c3e50; }}
            .insights ul {{ margin: 0; padding-left: 20px; }}
            .insights li {{ margin: 5px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔬 Robustness Analysis</h1>
            <p><strong>Source:</strong> {metadata['source_experiment']}</p>
            <p><strong>Test Period:</strong> {metadata['test_period_weeks']} weeks per run</p>
            <p><strong>Total Runs:</strong> {metadata['total_runs']} ({metadata['samples_per_regime']} starts per regime × {len(metadata['configs_tested'])} configs)</p>
            <p><strong>Regime Scaling:</strong> {'Yes (10% larger positions in bull markets)' if metadata.get('use_regime_scaling') else 'No'}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="chart-container">
            <div id="summary"></div>
        </div>
        
        <div class="chart-container">
            <div id="equity"></div>
        </div>
        
        <div class="grid-2">
            <div class="chart-container">
                <div id="distribution"></div>
            </div>
            <div class="chart-container">
                <div id="sensitivity"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="regime"></div>
        </div>
        
        <div class="insights">
            <h3>📊 Key Insights</h3>
            <ul id="insights-list">
            </ul>
        </div>
        
        <script>
            var summaryData = {summary_fig.to_json()};
            var equityData = {equity_fig.to_json()};
            var distData = {dist_fig.to_json()};
            var sensitivityData = {sensitivity_fig.to_json()};
            var regimeData = {regime_fig.to_json()};
            
            Plotly.newPlot('summary', summaryData.data, summaryData.layout);
            Plotly.newPlot('equity', equityData.data, equityData.layout);
            Plotly.newPlot('distribution', distData.data, distData.layout);
            Plotly.newPlot('sensitivity', sensitivityData.data, sensitivityData.layout);
            Plotly.newPlot('regime', regimeData.data, regimeData.layout);
            
            // Generate insights
            var insights = [
                "All {len(results_df)} runs completed successfully across {len(results_df['config_name'].unique())} configurations.",
            ];
            
            var insightsList = document.getElementById('insights-list');
            insights.forEach(function(insight) {{
                var li = document.createElement('li');
                li.textContent = insight;
                insightsList.appendChild(li);
            }});
        </script>
    </body>
    </html>
    """
    
    output_path = experiment_dir / "robustness_dashboard.html"
    with open(output_path, "w") as f:
        f.write(html_content)
    
    print(f"\n✅ Dashboard saved to: {output_path}")
    
    # Print insights to console
    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS INSIGHTS")
    print(f"{'='*60}")
    
    # Calculate key stats
    for config_name in results_df["config_name"].unique():
        cfg = results_df[results_df["config_name"] == config_name]
        bull_starts = cfg[cfg["start_regime"] == "bull"]
        bear_starts = cfg[cfg["start_regime"] == "bear"]
        
        print(f"\n{config_name}:")
        print(f"   Mean Return: {cfg['total_return_pcnt'].mean():.1f}% ± {cfg['total_return_pcnt'].std():.1f}%")
        print(f"   Range: [{cfg['total_return_pcnt'].min():.1f}%, {cfg['total_return_pcnt'].max():.1f}%]")
        print(f"   Consistency (std/mean): {cfg['total_return_pcnt'].std() / cfg['total_return_pcnt'].mean() * 100:.0f}%")
        
        if len(bull_starts) > 0 and len(bear_starts) > 0:
            diff = bull_starts['total_return_pcnt'].mean() - bear_starts['total_return_pcnt'].mean()
            print(f"   Bull vs Bear start difference: {diff:+.1f}%")
    
    # Overall
    print(f"\n{'='*60}")
    print("OVERALL ROBUSTNESS ASSESSMENT")
    print(f"{'='*60}")
    
    # Check if all configs have positive returns in most runs
    positive_rate = (results_df["total_return_pcnt"] > 0).mean() * 100
    print(f"Positive return runs: {positive_rate:.0f}%")
    
    avg_return = results_df["total_return_pcnt"].mean()
    std_return = results_df["total_return_pcnt"].std()
    print(f"Overall mean return: {avg_return:.1f}% ± {std_return:.1f}%")
    
    # Coefficient of variation
    cv = std_return / avg_return if avg_return != 0 else float('inf')
    if cv < 0.5:
        print(f"Coefficient of variation: {cv:.2f} (LOW - consistent)")
    elif cv < 1.0:
        print(f"Coefficient of variation: {cv:.2f} (MODERATE)")
    else:
        print(f"Coefficient of variation: {cv:.2f} (HIGH - variable)")
    
    return output_path


def main(source_experiment: str):
    """Run robustness analysis."""
    print(f"\n🔬 Analyzing robustness results for: {source_experiment}\n")
    
    dashboard_path = create_dashboard(source_experiment)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main(SOURCE_EXPERIMENT)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate scatter plot dashboards for predictive relationship analysis.

Creates interactive multi-panel HTML dashboards showing:
- Feature at week t vs log_return at week t+1
- Linear regression with confidence band
- Correlation statistics (r, R², p-value)
- Visual highlighting for significant relationships

This helps identify which features might have predictive power for future returns.

Input: data/historical/{tier}/weekly/*.csv
Output: data/visualizations/{tier}/*_scatter.html, _inspector_scatter.html
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.workflow.config import (
    COMPRESS_HTML,
    DATA_TIER,
    MACRO_SYMBOL_CATEGORIES,
    MACRO_SYMBOL_LIST,
    MACRO_SYMBOLS,
    SYMBOL_PREFIX_FILTER,
    VIZ_COLORS,
)
from src.workflow.workflow_utils import (
    get_historical_dir,
    get_metadata_dir,
    get_visualizations_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()

# Scatter plot configuration
SCATTER_PANELS = [
    {
        "x_feature": "log_return",
        "title": "Return → Return",
        "x_label": "log_return(t)",
        "interpretation": "Momentum or mean reversion?",
    },
    {
        "x_feature": "momentum_4w",
        "title": "Momentum 4W → Return",
        "x_label": "momentum_4w(t)",
        "interpretation": "4-week trend continuation?",
    },
    {
        "x_feature": "momentum_12w",
        "title": "Momentum 12W → Return",
        "x_label": "momentum_12w(t)",
        "interpretation": "3-month trend continuation?",
    },
    {
        "x_feature": "intra_week_volatility",
        "title": "Volatility → Return",
        "x_label": "volatility(t)",
        "interpretation": "Risk-return relationship?",
    },
    {
        "x_feature": "log_volume_delta",
        "title": "Volume Δ → Return",
        "x_label": "volume_delta(t)",
        "interpretation": "Volume predicts returns?",
    },
    {
        "x_feature": "log_return_intraweek",
        "title": "Intra-week → Return",
        "x_label": "intraweek_return(t)",
        "interpretation": "Intra-week pattern predicts?",
    },
]

# Significance threshold
SIGNIFICANCE_ALPHA = 0.05


def load_weekly_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load weekly data from CSV.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with weekly data, or None if error
    """
    try:
        df = pd.read_csv(filepath)
        df["week_start"] = pd.to_datetime(df["week_start"])
        df = df.sort_values("week_start").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def load_all_symbol_metadata(metadata_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata for targets and macro symbols."""
    result: Dict[str, Dict[str, str]] = {}

    metadata_file = metadata_dir / "filtered_etfs.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
            for etf in data.get("etfs", []):
                result[etf["symbol"]] = {
                    "name": etf.get("name", ""),
                    "category": "target",
                }
        except Exception as e:
            logger.warning(f"Could not load ETF metadata: {e}")

    for category, symbols in MACRO_SYMBOLS.items():
        for symbol, description in symbols.items():
            result[symbol] = {
                "name": description,
                "category": category,
            }

    return result


def get_symbols_to_process(input_dir: Path) -> List[Path]:
    """Get list of CSV files to process."""
    all_csv_files = sorted(input_dir.glob("*.csv"))
    all_csv_files = [f for f in all_csv_files if not f.name.startswith("_")]

    macro_symbols: Set[str] = set(MACRO_SYMBOL_LIST)

    if SYMBOL_PREFIX_FILTER:
        csv_files = [
            f
            for f in all_csv_files
            if f.stem.startswith(SYMBOL_PREFIX_FILTER) or f.stem in macro_symbols
        ]
    else:
        csv_files = all_csv_files

    return csv_files


def compute_regression_stats(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, Any]:
    """Compute regression statistics for scatter plot.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Dict with slope, intercept, r, r_squared, p_value, std_err
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n": len(x_clean),
            "significant": False,
        }

    slope, intercept, r, p_value, std_err = stats.linregress(x_clean, y_clean)

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "r_squared": r**2,
        "p_value": p_value,
        "std_err": std_err,
        "n": len(x_clean),
        "significant": p_value < SIGNIFICANCE_ALPHA,
    }


def create_scatter_dashboard(
    df: pd.DataFrame, symbol: str, name: str = "", category: str = "target"
) -> go.Figure:
    """Create multi-panel scatter dashboard for predictive analysis.

    Args:
        df: DataFrame with weekly feature data
        symbol: Symbol ticker
        name: Full name/description of the symbol
        category: Category (target, volatility, treasury, etc.)

    Returns:
        Plotly figure object
    """
    # Create future return column (what we're trying to predict)
    df = df.copy()
    df["future_return"] = df["log_return"].shift(-1)

    # Build subplot titles dynamically based on stats
    subplot_titles = []
    panel_stats = []

    for panel in SCATTER_PANELS:
        x_feature = panel["x_feature"]
        if x_feature in df.columns:
            x_vals = df[x_feature].values
            y_vals = df["future_return"].values
            stats_dict = compute_regression_stats(x_vals, y_vals)
            panel_stats.append(stats_dict)

            # Build title with stats
            if np.isnan(stats_dict["r"]):
                title = f"<b>{panel['title']}</b><br><sup>Insufficient data</sup>"
            else:
                sig_marker = " ✓" if stats_dict["significant"] else ""
                title = (
                    f"<b>{panel['title']}</b>{sig_marker}<br>"
                    f"<sup>r={stats_dict['r']:.3f}, R²={stats_dict['r_squared']:.3f}, "
                    f"p={stats_dict['p_value']:.3f}</sup>"
                )
            subplot_titles.append(title)
        else:
            panel_stats.append(None)
            subplot_titles.append(f"<b>{panel['title']}</b><br><sup>Feature not available</sup>")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Color scheme
    point_color = "rgba(46, 134, 171, 0.5)"  # Semi-transparent blue
    sig_point_color = "rgba(40, 167, 69, 0.6)"  # Green for significant
    line_color = "#FF8C00"  # Orange for regression line
    ci_color = "rgba(255, 140, 0, 0.2)"  # Light orange for CI

    for i, (panel, stats_dict) in enumerate(zip(SCATTER_PANELS, panel_stats)):
        row = (i // 2) + 1
        col = (i % 2) + 1

        x_feature = panel["x_feature"]
        if x_feature not in df.columns or stats_dict is None:
            continue

        x_vals = df[x_feature].values
        y_vals = df["future_return"].values
        week_dates = df["week_start"].values

        # Remove NaN pairs for plotting
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_plot = x_vals[mask]
        y_plot = y_vals[mask]
        weeks_plot = week_dates[mask]

        if len(x_plot) < 3:
            continue

        # Format week dates for hover
        week_labels = [pd.Timestamp(w).strftime("%Y-%m-%d") for w in weeks_plot]

        # Choose color based on significance
        color = sig_point_color if stats_dict["significant"] else point_color

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=x_plot,
                y=y_plot,
                mode="markers",
                marker=dict(size=6, color=color, line=dict(width=0.5, color="white")),
                name=panel["title"],
                showlegend=False,
                customdata=week_labels,
                hovertemplate=(
                    f"Week: %{{customdata}}<br>"
                    f"{panel['x_label']}: %{{x:.4f}}<br>"
                    f"future_return(t+1): %{{y:.4f}}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        # Regression line
        if not np.isnan(stats_dict["slope"]):
            x_line = np.array([x_plot.min(), x_plot.max()])
            y_line = stats_dict["slope"] * x_line + stats_dict["intercept"]

            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    name="Regression",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # Confidence interval (approximate)
            # Using standard error to show uncertainty band
            n = len(x_plot)
            x_mean = x_plot.mean()
            se_fit = stats_dict["std_err"] * np.sqrt(
                1 / n + (x_line - x_mean) ** 2 / ((x_plot - x_mean) ** 2).sum()
            )
            t_crit = stats.t.ppf(0.975, n - 2)

            y_upper = y_line + t_crit * se_fit * np.sqrt(n)
            y_lower = y_line - t_crit * se_fit * np.sqrt(n)

            # CI band
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=ci_color,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        # Zero lines
        fig.add_hline(y=0, line_dash="dash", line_color="#666", opacity=0.5, row=row, col=col)
        fig.add_vline(x=0, line_dash="dash", line_color="#666", opacity=0.5, row=row, col=col)

        # Axis labels
        fig.update_xaxes(title_text=panel["x_label"], row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="log_return(t+1)", row=row, col=col)

    # Layout
    n_points = (~df["future_return"].isna()).sum()
    date_range = (
        f"{df['week_start'].min().strftime('%Y-%m-%d')} to "
        f"{df['week_start'].max().strftime('%Y-%m-%d')}"
    )

    if category != "target":
        category_badge = f"[{category.upper()}]"
        title_text = f"<b>{symbol}</b> {category_badge} - Predictive Scatter Analysis<br><sup>{name}</sup>"
    elif name:
        title_text = f"<b>{symbol}</b> - Predictive Scatter Analysis<br><sup>{name}</sup>"
    else:
        title_text = f"<b>{symbol}</b> - Predictive Scatter Analysis"

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=18)),
        width=1000,  # Square-ish: 2 cols x ~400px each
        height=1200,  # Square-ish: 3 rows x ~350px each
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=120, b=60, l=80, r=40),
    )

    # Make each subplot square by constraining the aspect ratio
    for i in range(1, 7):
        # For subplots, axes are numbered x1/y1, x2/y2, etc.
        xaxis_name = f"xaxis{i}" if i > 1 else "xaxis"
        yaxis_name = f"yaxis{i}" if i > 1 else "yaxis"
        fig.update_layout(**{
            yaxis_name: dict(scaleanchor=f"x{i}" if i > 1 else "x", scaleratio=1)
        })

    fig.add_annotation(
        text=f"{date_range} ({n_points} data points) | Y-axis: log_return at t+1 | ✓ = significant (p<0.05)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.02,
        showarrow=False,
        font=dict(size=11, color="#888"),
        xanchor="center",
    )

    return fig


def save_dashboard(fig: go.Figure, output_path: str) -> str:
    """Save Plotly figure as HTML.

    Args:
        fig: Plotly figure
        output_path: Path for output file

    Returns:
        Actual path where file was saved
    """
    html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def create_scatter_inspector_html(
    html_files: List[Dict[str, str]],
    output_path: str,
    data_tier: str,
) -> None:
    """Create HTML inspector for browsing scatter dashboards."""
    category_order = ["target", "volatility", "treasury", "dollar", "commodities"]
    html_files = sorted(
        html_files,
        key=lambda x: (
            category_order.index(x["category"])
            if x["category"] in category_order
            else 99,
            x["symbol"],
        ),
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Scatter Inspector - {data_tier.upper()}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #1a1a2e; color: #eee; min-height: 100vh; }}
        .header {{ background: #16213e; padding: 15px 20px; display: flex;
                  justify-content: space-between; align-items: center;
                  border-bottom: 2px solid #28a745; position: fixed;
                  top: 0; left: 0; right: 0; z-index: 1000; }}
        .title {{ font-size: 1.4em; font-weight: bold; color: #28a745; }}
        .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
        .nav-btn {{ background: #0f3460; color: #fff; border: none; padding: 10px 20px;
                   border-radius: 5px; cursor: pointer; font-size: 1em; }}
        .nav-btn:hover {{ background: #28a745; }}
        .nav-btn:disabled {{ background: #333; cursor: not-allowed; }}
        .search-box, .symbol-select, .category-select {{ padding: 10px; border-radius: 5px;
                                       border: 1px solid #0f3460; background: #16213e; color: #fff; }}
        .category-select {{ min-width: 100px; }}
        .progress {{ color: #888; font-size: 0.9em; }}
        .main {{ margin-top: 70px; height: calc(100vh - 70px); }}
        .iframe-container {{ width: 100%; height: 100%; }}
        .iframe-container iframe {{ width: 100%; height: 100%; border: none; }}
        .keyboard-help {{ position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
                         color: #666; font-size: 0.8em; background: rgba(0,0,0,0.7);
                         padding: 5px 15px; border-radius: 20px; }}
        .keyboard-help kbd {{ background: #333; padding: 2px 6px; border-radius: 3px; margin: 0 2px; }}
        .view-toggle {{ background: #28a745; padding: 8px 15px; border-radius: 5px; 
                       text-decoration: none; color: white; font-size: 0.9em; }}
        .view-toggle:hover {{ background: #218838; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">📊 Predictive Scatter Inspector ({data_tier.upper()})</div>
        <div class="controls">
            <a href="_inspector.html" class="view-toggle">📈 Time Series View</a>
            <select class="category-select" id="category-select">
                <option value="all">All Categories</option>
                <option value="target">Targets</option>
                <option value="volatility">Volatility</option>
                <option value="treasury">Treasury</option>
                <option value="dollar">Dollar</option>
                <option value="commodities">Commodities</option>
            </select>
            <input type="text" class="search-box" id="search" placeholder="Search symbol...">
            <select class="symbol-select" id="symbol-select"></select>
            <button class="nav-btn" id="prev-btn">← Prev</button>
            <button class="nav-btn" id="next-btn">Next →</button>
            <span class="progress" id="progress"></span>
        </div>
    </div>
    <div class="main"><div class="iframe-container"><iframe id="dashboard-frame" src=""></iframe></div></div>
    <div class="keyboard-help"><kbd>←</kbd> Prev <kbd>→</kbd> Next <kbd>/</kbd> Search</div>
    <script>
        const allFiles = {json.dumps(html_files)};
        let filteredFiles = [...allFiles];
        let currentIndex = 0;
        const frame = document.getElementById('dashboard-frame');
        const progress = document.getElementById('progress');
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');
        const searchBox = document.getElementById('search');
        const symbolSelect = document.getElementById('symbol-select');
        const categorySelect = document.getElementById('category-select');

        function updateSymbolList() {{
            symbolSelect.innerHTML = '';
            filteredFiles.forEach((f, i) => {{
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `${{f.symbol}} (${{f.category}})`;
                symbolSelect.appendChild(opt);
            }});
        }}

        function showDashboard(index) {{
            if (filteredFiles.length === 0) return;
            if (index < 0) index = 0;
            if (index >= filteredFiles.length) index = filteredFiles.length - 1;
            currentIndex = index;
            const file = filteredFiles[index];
            frame.src = file.path;
            progress.textContent = `${{file.symbol}} [${{file.category}}] (${{index + 1}}/${{filteredFiles.length}})`;
            prevBtn.disabled = index === 0;
            nextBtn.disabled = index === filteredFiles.length - 1;
            symbolSelect.value = index;
        }}

        function filterByCategory(category) {{
            if (category === 'all') {{
                filteredFiles = [...allFiles];
            }} else {{
                filteredFiles = allFiles.filter(f => f.category === category);
            }}
            updateSymbolList();
            showDashboard(0);
        }}

        categorySelect.addEventListener('change', (e) => filterByCategory(e.target.value));
        prevBtn.addEventListener('click', () => showDashboard(currentIndex - 1));
        nextBtn.addEventListener('click', () => showDashboard(currentIndex + 1));
        symbolSelect.addEventListener('change', (e) => showDashboard(parseInt(e.target.value)));
        searchBox.addEventListener('input', (e) => {{
            const query = e.target.value.toUpperCase();
            const found = filteredFiles.findIndex(f => f.symbol.toUpperCase().includes(query));
            if (found >= 0) showDashboard(found);
        }});
        document.addEventListener('keydown', (e) => {{
            if (e.target === searchBox) return;
            switch(e.key) {{
                case 'ArrowLeft': showDashboard(currentIndex - 1); break;
                case 'ArrowRight': showDashboard(currentIndex + 1); break;
                case '/': e.preventDefault(); searchBox.focus(); break;
            }}
        }});

        updateSymbolList();
        showDashboard(0);
    </script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)


def generate_significance_report(
    all_stats: List[Dict[str, Any]],
    output_dir: Path,
    top_n: int = 100,
) -> Path:
    """Generate a markdown report of the most significant feature-return relationships.

    Args:
        all_stats: List of dicts with symbol, feature, and regression stats
        output_dir: Directory to save report
        top_n: Number of top results to include

    Returns:
        Path to generated report
    """
    from datetime import datetime

    # Filter to valid stats with minimum sample size for credible rankings
    # Still compute everything, but only rank relationships with enough data
    MIN_SAMPLE_SIZE = 26  # ~6 months / 2 quarters of weekly data
    
    valid_stats = [s for s in all_stats if not np.isnan(s.get("p_value", np.nan))]
    
    # For ranking: significant + minimum sample size, sorted by R²
    rankable_stats = [
        s for s in valid_stats 
        if s["p_value"] < SIGNIFICANCE_ALPHA and s["n"] >= MIN_SAMPLE_SIZE
    ]
    sorted_stats = sorted(rankable_stats, key=lambda x: -x["r_squared"])

    # Separate significant (all) vs rankable (meeting sample size requirement)
    significant = [s for s in valid_stats if s["p_value"] < SIGNIFICANCE_ALPHA]
    top_results = sorted_stats[:top_n]  # Already filtered to significant + min sample size

    # Generate markdown
    lines = []
    lines.append("# Predictive Scatter Analysis - Significance Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total feature-symbol combinations tested:** {len(valid_stats)}")
    lines.append(f"- **Significant at p<{SIGNIFICANCE_ALPHA}:** {len(significant)}")
    lines.append(f"- **Rankable (significant + n≥{MIN_SAMPLE_SIZE}):** {len(sorted_stats)}")
    lines.append(f"- **Significance rate:** {len(significant)/len(valid_stats)*100:.1f}%")
    lines.append("")

    # Note about multiple testing
    lines.append("> ⚠️ **Multiple Testing Warning:** With many tests, some will appear ")
    lines.append(f"> significant by chance. Expected false positives at α={SIGNIFICANCE_ALPHA}: ")
    lines.append(f"> ~{int(len(valid_stats) * SIGNIFICANCE_ALPHA)} tests.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Top significant results (ranked by R² among significant relationships with enough data)
    lines.append(f"## Top {min(top_n, len(top_results))} Relationships (p<0.05, n≥{MIN_SAMPLE_SIZE}, Ranked by R²)")
    lines.append("")
    lines.append("| Rank | Symbol | Feature | R² | r | p-value | n | Direction |")
    lines.append("|------|--------|---------|----|----|---------|---|-----------|")

    for i, stat in enumerate(top_results, 1):
        direction = "📈" if stat["r"] > 0 else "📉"
        sig_marker = "✓" if stat["p_value"] < SIGNIFICANCE_ALPHA else ""
        lines.append(
            f"| {i} | [{stat['symbol']}]({stat['symbol']}_scatter.html) | "
            f"{stat['feature']} | **{stat['r_squared']:.3f}** | {stat['r']:.3f} | "
            f"{stat['p_value']:.4f}{sig_marker} | {stat['n']} | {direction} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Breakdown by feature
    lines.append("## Breakdown by Feature")
    lines.append("")
    lines.append("Which features show the most predictive signal across symbols?")
    lines.append("")

    feature_summary = {}
    for stat in valid_stats:
        feature = stat["feature"]
        if feature not in feature_summary:
            feature_summary[feature] = {"count": 0, "sig_count": 0, "avg_abs_r": []}
        feature_summary[feature]["count"] += 1
        feature_summary[feature]["avg_abs_r"].append(abs(stat["r"]))
        if stat["p_value"] < SIGNIFICANCE_ALPHA:
            feature_summary[feature]["sig_count"] += 1

    lines.append("| Feature | Symbols Tested | Significant | Sig Rate | Avg |r| |")
    lines.append("|---------|----------------|-------------|----------|--------|")

    for feature in SCATTER_PANELS:
        fname = feature["x_feature"]
        if fname in feature_summary:
            fs = feature_summary[fname]
            avg_r = np.mean(fs["avg_abs_r"])
            sig_rate = fs["sig_count"] / fs["count"] * 100
            lines.append(
                f"| {fname} | {fs['count']} | {fs['sig_count']} | "
                f"{sig_rate:.1f}% | {avg_r:.3f} |"
            )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Symbols with multiple significant features
    lines.append("## Symbols with Multiple Significant Features")
    lines.append("")

    symbol_sig_count = {}
    for stat in significant:
        sym = stat["symbol"]
        if sym not in symbol_sig_count:
            symbol_sig_count[sym] = []
        symbol_sig_count[sym].append(stat["feature"])

    multi_sig = {k: v for k, v in symbol_sig_count.items() if len(v) > 1}
    if multi_sig:
        lines.append("| Symbol | # Sig Features | Features |")
        lines.append("|--------|----------------|----------|")
        for sym, features in sorted(multi_sig.items(), key=lambda x: -len(x[1])):
            lines.append(f"| [{sym}]({sym}_scatter.html) | {len(features)} | {', '.join(features)} |")
    else:
        lines.append("*No symbols have more than one significant feature.*")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `04b-visualize-scatter.py`*")

    # Write report
    report_path = output_dir / "scatter_significance_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # Also save as JSON for programmatic access
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy types to native Python types."""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_path = output_dir / "scatter_significance_stats.json"
    with open(json_path, "w") as f:
        json.dump(convert_to_native({
            "generated_at": datetime.now().isoformat(),
            "total_tests": len(valid_stats),
            "significant_count": len(significant),
            "alpha": SIGNIFICANCE_ALPHA,
            "top_results": top_results[:top_n],
            "feature_summary": {
                k: {
                    "count": v["count"],
                    "sig_count": v["sig_count"],
                    "avg_abs_r": float(np.mean(v["avg_abs_r"])),
                }
                for k, v in feature_summary.items()
            },
        }), f, indent=2)

    return report_path


@workflow_script("04b-visualize-scatter")
def main() -> None:
    """Main workflow function."""
    metadata_dir = get_metadata_dir()
    input_dir = get_historical_dir(DATA_TIER) / "weekly"
    output_dir = get_visualizations_dir(DATA_TIER)
    os.makedirs(output_dir, exist_ok=True)

    symbol_metadata = load_all_symbol_metadata(metadata_dir)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter (targets only): {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Panels: {len(SCATTER_PANELS)}")
    print()

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run 03-generate-features.py first.")
        return

    csv_files = get_symbols_to_process(input_dir)

    target_count = sum(1 for f in csv_files if f.stem not in MACRO_SYMBOL_LIST)
    macro_count = sum(1 for f in csv_files if f.stem in MACRO_SYMBOL_LIST)

    print(f"Found {len(csv_files)} weekly CSV files to visualize")
    print(f"  Target ETFs: {target_count}")
    print(f"  Macro symbols: {macro_count}")
    print("-" * 80)

    if not csv_files:
        logger.warning("No files to process. Exiting.")
        return

    success_count = 0
    fail_count = 0
    html_files: List[Dict[str, str]] = []
    all_regression_stats: List[Dict[str, Any]] = []  # Collect stats for report

    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.stem
        meta = symbol_metadata.get(symbol, {"name": "", "category": "target"})
        name = meta.get("name", "")
        category = meta.get("category", "target")
        category_indicator = f"[{category.upper()[:3]}]"

        print(f"[{i}/{len(csv_files)}] {symbol} {category_indicator}...", end=" ")

        df = load_weekly_data(str(csv_file))
        if df is None or df.empty:
            print("✗ No data")
            fail_count += 1
            continue

        # Need at least some data points for meaningful scatter
        if len(df) < 10:
            print(f"✗ Too few points ({len(df)})")
            fail_count += 1
            continue

        try:
            # Compute stats for each panel and collect for report
            df_copy = df.copy()
            df_copy["future_return"] = df_copy["log_return"].shift(-1)

            for panel in SCATTER_PANELS:
                x_feature = panel["x_feature"]
                if x_feature in df_copy.columns:
                    x_vals = df_copy[x_feature].values
                    y_vals = df_copy["future_return"].values
                    stats_dict = compute_regression_stats(x_vals, y_vals)
                    stats_dict["symbol"] = symbol
                    stats_dict["feature"] = x_feature
                    stats_dict["category"] = category
                    all_regression_stats.append(stats_dict)

            fig = create_scatter_dashboard(df, symbol, name, category)
            output_path = str(output_dir / f"{symbol}_scatter.html")
            save_dashboard(fig, output_path)
            html_files.append({
                "symbol": symbol,
                "path": Path(output_path).name,
                "category": category,
            })
            success_count += 1
            size_kb = os.path.getsize(output_path) / 1024
            print(f"✓ ({size_kb:.0f} KB)")
        except Exception as e:
            logger.error(f"Error: {e}")
            fail_count += 1

    print()
    print("Creating scatter inspector...")
    inspector_path = output_dir / "_inspector_scatter.html"
    create_scatter_inspector_html(html_files, str(inspector_path), DATA_TIER)
    print(f"  Inspector: {inspector_path}")

    # Generate significance report
    print()
    print("Generating significance report...")
    report_path = generate_significance_report(all_regression_stats, output_dir, top_n=100)
    print(f"  Report: {report_path}")

    # Count significant results
    sig_count = sum(
        1 for s in all_regression_stats
        if not np.isnan(s.get("p_value", np.nan)) and s["p_value"] < SIGNIFICANCE_ALPHA
    )
    total_tests = sum(
        1 for s in all_regression_stats if not np.isnan(s.get("p_value", np.nan))
    )

    print()

    print_summary(
        scatter_dashboards_created=success_count,
        target_etfs=target_count,
        macro_symbols=macro_count,
        failed=fail_count,
        total_feature_tests=total_tests,
        significant_relationships=sig_count,
        significance_rate=f"{sig_count/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
        output_directory=str(output_dir),
        inspector=str(inspector_path),
        report=str(report_path),
    )


if __name__ == "__main__":
    main()

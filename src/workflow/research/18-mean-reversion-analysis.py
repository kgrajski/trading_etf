#!/usr/bin/env python3
"""
18-mean-reversion-analysis.py

Phase I: Visualization for mean-reversion signal analysis.

Generates:
1. Histogram of latest week's log_returns with hover data including all features
2. Candlestick + Bollinger Bands charts for worst 5% performers (1 year back)
3. Inspector HTML to browse the charts

This helps identify potential mean-reversion entry candidates.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp018_mean_reversion_analysis"

# Data paths
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Bollinger Bands parameters
BB_PERIOD = 20  # 20-day moving average
BB_STD = 2  # 2 standard deviations

# Worst performers threshold
WORST_PERCENTILE = 0.05  # Bottom 5%

# Lookback for candlestick charts (trading days)
CHART_LOOKBACK_DAYS = 252  # ~1 year

# Features to show in hover (in order)
HOVER_FEATURES = [
    "log_return",
    "log_return_intraweek",
    "log_range",
    "log_volume",
    "intra_week_volatility",
    "momentum_4w",
    "momentum_12w",
    "volatility_ma4",
    "volatility_ma12",
    "log_volume_delta",
]


def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    return df


def get_latest_week_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get data for the latest week, filtered to target ETFs only."""
    # Filter to target category
    target_df = df[df["category"] == "target"].copy()
    
    # Get latest week
    latest_week_idx = target_df["week_idx"].max()
    latest_week = target_df[target_df["week_idx"] == latest_week_idx].copy()
    
    # Compute actual percent return from log return
    latest_week["pct_return"] = (np.exp(latest_week["log_return"]) - 1) * 100
    
    return latest_week


def create_return_histogram(latest_week: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create histogram of latest week's returns with rich hover data.
    """
    # Sort by return for consistent ordering
    df = latest_week.sort_values("log_return").reset_index(drop=True)
    
    # Build hover text with all features
    hover_texts = []
    for _, row in df.iterrows():
        lines = [
            f"<b>{row['symbol']}</b>",
            f"{row['name'][:50]}..." if len(str(row.get('name', ''))) > 50 else f"{row.get('name', 'N/A')}",
            "",
            f"<b>Return: {row['pct_return']:.2f}%</b> (log: {row['log_return']:.4f})",
            "",
            "<b>Features:</b>",
        ]
        
        for feat in HOVER_FEATURES:
            if feat in row and pd.notna(row[feat]):
                # Format based on feature type
                if "log_" in feat and "return" not in feat:
                    lines.append(f"  {feat}: {row[feat]:.2f}")
                elif "momentum" in feat or "return" in feat:
                    lines.append(f"  {feat}: {row[feat]:.4f} ({row[feat]*100:.2f}%)")
                elif "volatility" in feat:
                    lines.append(f"  {feat}: {row[feat]:.4f}")
                else:
                    lines.append(f"  {feat}: {row[feat]:.4f}")
        
        hover_texts.append("<br>".join(lines))
    
    # Create histogram with custom hover
    fig = go.Figure()
    
    # Color by return (red for negative, green for positive)
    colors = ["#DC3545" if r < 0 else "#28A745" for r in df["pct_return"]]
    
    fig.add_trace(go.Bar(
        x=df["symbol"],
        y=df["pct_return"],
        marker_color=colors,
        hovertext=hover_texts,
        hoverinfo="text",
        name="Weekly Return",
    ))
    
    # Get week info
    week_start = df["week_start"].iloc[0]
    if isinstance(week_start, str):
        week_str = week_start
    else:
        week_str = week_start.strftime("%Y-%m-%d")
    
    # Highlight bottom 5%
    n_worst = int(len(df) * WORST_PERCENTILE)
    worst_symbols = df.head(n_worst)["symbol"].tolist()
    
    fig.update_layout(
        title=f"Weekly Returns Distribution - Week of {week_str}<br>"
              f"<sub>Bottom 5% ({n_worst} symbols) highlighted for mean-reversion analysis</sub>",
        xaxis_title="Symbol",
        yaxis_title="Return (%)",
        height=600,
        showlegend=False,
        xaxis=dict(
            tickangle=90,
            tickfont=dict(size=8),
        ),
        hovermode="closest",
    )
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add annotation for bottom 5% cutoff
    cutoff_return = df.iloc[n_worst - 1]["pct_return"] if n_worst > 0 else 0
    fig.add_hline(
        y=cutoff_return,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Bottom 5% cutoff: {cutoff_return:.2f}%",
        annotation_position="top right",
    )
    
    output_path = output_dir / "return_histogram.html"
    fig.write_html(output_path)
    logging.info(f"Histogram saved: {output_path}")
    
    return output_path


def load_daily_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load daily OHLCV data for a symbol."""
    csv_path = DAILY_DATA_DIR / f"{symbol}.csv"
    if not csv_path.exists():
        logging.warning(f"Daily data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Compute Bollinger Bands."""
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=period).mean()
    df["bb_std"] = df["close"].rolling(window=period).std()
    df["bb_upper"] = df["bb_middle"] + (std_dev * df["bb_std"])
    df["bb_lower"] = df["bb_middle"] - (std_dev * df["bb_std"])
    return df


def create_candlestick_chart(
    symbol: str,
    name: str,
    daily_df: pd.DataFrame,
    week_return: float,
    week_pct_return: float,
    features: dict,
    output_dir: Path,
) -> Path:
    """
    Create candlestick chart with Bollinger Bands and volume.
    
    Styled similar to the example: gray shaded BB area, candles, volume bars below.
    """
    # Limit to last ~1 year
    df = daily_df.tail(CHART_LOOKBACK_DAYS).copy()
    
    # Compute Bollinger Bands
    df = compute_bollinger_bands(df, BB_PERIOD, BB_STD)
    
    # Create subplots: main chart + volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    
    # Bollinger Bands - shaded area
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(128, 128, 128, 0.2)",
            line=dict(color="rgba(128, 128, 128, 0.4)", width=1),
            name="Bollinger Bands",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    
    # Bollinger middle line (20-day MA)
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bb_middle"],
            line=dict(color="rgba(128, 128, 128, 0.6)", width=1, dash="dash"),
            name="20-day MA",
            hovertemplate="MA20: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#28A745",
            decreasing_line_color="#DC3545",
            increasing_fillcolor="#28A745",
            decreasing_fillcolor="#DC3545",
        ),
        row=1, col=1,
    )
    
    # Current price line
    current_price = df["close"].iloc[-1]
    fig.add_hline(
        y=current_price,
        line_dash="solid",
        line_color="#FFD700",
        line_width=1,
        annotation_text=f"{current_price:.2f}",
        annotation_position="right",
        row=1, col=1,
    )
    
    # Volume bars
    volume_colors = [
        "#28A745" if df["close"].iloc[i] >= df["open"].iloc[i] else "#DC3545"
        for i in range(len(df))
    ]
    
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            marker_color=volume_colors,
            opacity=0.7,
            name="Volume",
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    
    # Build feature summary for subtitle
    feature_summary = []
    for feat in ["momentum_4w", "momentum_12w", "volatility_ma4"]:
        if feat in features and pd.notna(features[feat]):
            val = features[feat]
            if "momentum" in feat:
                feature_summary.append(f"{feat.replace('momentum_', 'Mom')}: {val*100:.1f}%")
            else:
                feature_summary.append(f"{feat.replace('volatility_', 'Vol')}: {val:.3f}")
    
    # Layout
    last_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    range_52w = f"{df['low'].min():.2f} - {df['high'].max():.2f}"
    avg_volume = df["volume"].mean()
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{symbol}</b> - {name[:50]}<br>"
                f"<sub>Last: {current_price:.2f} | "
                f"Week Return: {week_pct_return:.2f}% | "
                f"52w Range: {range_52w} | "
                f"Avg Vol: {avg_volume/1e6:.1f}M</sub><br>"
                f"<sub>{' | '.join(feature_summary)}</sub>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        hovermode="x unified",
        margin=dict(t=120, b=40, l=60, r=60),
    )
    
    # Hide weekend gaps
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    output_path = output_dir / f"{symbol}.html"
    fig.write_html(output_path)
    
    return output_path


def create_inspector_html(
    worst_performers: pd.DataFrame,
    chart_files: list[tuple[str, Path]],
    output_dir: Path,
) -> Path:
    """
    Create an inspector HTML page to browse the candlestick charts.
    """
    week_start = worst_performers["week_start"].iloc[0]
    if isinstance(week_start, str):
        week_str = week_start
    else:
        week_str = week_start.strftime("%Y-%m-%d")
    
    # Build table rows
    table_rows = []
    for _, row in worst_performers.iterrows():
        symbol = row["symbol"]
        name = str(row.get("name", ""))[:40]
        pct_return = row["pct_return"]
        mom_4w = row.get("momentum_4w", 0) * 100 if pd.notna(row.get("momentum_4w")) else 0
        mom_12w = row.get("momentum_12w", 0) * 100 if pd.notna(row.get("momentum_12w")) else 0
        vol_4w = row.get("volatility_ma4", 0) if pd.notna(row.get("volatility_ma4")) else 0
        
        table_rows.append(f"""
        <tr onclick="loadChart('{symbol}')" class="clickable">
            <td><strong>{symbol}</strong></td>
            <td>{name}</td>
            <td class="{'negative' if pct_return < 0 else 'positive'}">{pct_return:.2f}%</td>
            <td class="{'negative' if mom_4w < 0 else 'positive'}">{mom_4w:.1f}%</td>
            <td class="{'negative' if mom_12w < 0 else 'positive'}">{mom_12w:.1f}%</td>
            <td>{vol_4w:.3f}</td>
        </tr>
        """)
    
    # First symbol for initial load
    first_symbol = worst_performers.iloc[0]["symbol"] if len(worst_performers) > 0 else ""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mean-Reversion Candidates - Week of {week_str}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}
        .container {{
            display: flex;
            gap: 20px;
            height: calc(100vh - 120px);
        }}
        .sidebar {{
            width: 500px;
            flex-shrink: 0;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .sidebar-header {{
            padding: 15px;
            background: #2E86AB;
            color: white;
            font-weight: bold;
        }}
        .table-container {{
            overflow-y: auto;
            flex: 1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th {{
            background: #f0f0f0;
            padding: 10px 8px;
            text-align: left;
            font-size: 12px;
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        td {{
            padding: 8px;
            border-bottom: 1px solid #eee;
            font-size: 13px;
        }}
        tr.clickable {{
            cursor: pointer;
        }}
        tr.clickable:hover {{
            background: #e8f4f8;
        }}
        tr.active {{
            background: #d0e8f0 !important;
        }}
        .negative {{
            color: #DC3545;
        }}
        .positive {{
            color: #28A745;
        }}
        .chart-container {{
            flex: 1;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        iframe {{
            width: 100%;
            height: 100%;
            border: none;
        }}
        .histogram-link {{
            display: inline-block;
            margin-bottom: 15px;
            padding: 8px 16px;
            background: #2E86AB;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }}
        .histogram-link:hover {{
            background: #236b8e;
        }}
    </style>
</head>
<body>
    <h1>Mean-Reversion Candidates</h1>
    <div class="subtitle">Week of {week_str} | Bottom 5% Performers ({len(worst_performers)} symbols)</div>
    
    <a href="return_histogram.html" class="histogram-link" target="_blank">📊 View Full Return Histogram</a>
    
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                Click a symbol to view chart
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Name</th>
                            <th>Return</th>
                            <th>Mom 4w</th>
                            <th>Mom 12w</th>
                            <th>Vol 4w</th>
                        </tr>
                    </thead>
                    <tbody id="symbolTable">
                        {"".join(table_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        <div class="chart-container">
            <iframe id="chartFrame" src="charts/{first_symbol}.html"></iframe>
        </div>
    </div>
    
    <script>
        let currentSymbol = '{first_symbol}';
        
        function loadChart(symbol) {{
            document.getElementById('chartFrame').src = 'charts/' + symbol + '.html';
            
            // Update active state
            document.querySelectorAll('#symbolTable tr').forEach(row => {{
                row.classList.remove('active');
            }});
            event.currentTarget.classList.add('active');
            currentSymbol = symbol;
        }}
        
        // Set initial active
        document.querySelector('#symbolTable tr')?.classList.add('active');
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            const rows = Array.from(document.querySelectorAll('#symbolTable tr'));
            const currentIdx = rows.findIndex(r => r.classList.contains('active'));
            
            if (e.key === 'ArrowDown' && currentIdx < rows.length - 1) {{
                e.preventDefault();
                rows[currentIdx + 1].click();
                rows[currentIdx + 1].scrollIntoView({{ block: 'nearest' }});
            }} else if (e.key === 'ArrowUp' && currentIdx > 0) {{
                e.preventDefault();
                rows[currentIdx - 1].click();
                rows[currentIdx - 1].scrollIntoView({{ block: 'nearest' }});
            }}
        }});
    </script>
</body>
</html>
"""
    
    output_path = output_dir / "_inspector.html"
    output_path.write_text(html_content)
    logging.info(f"Inspector saved: {output_path}")
    
    return output_path


@workflow_script("18-mean-reversion-analysis")
def main() -> None:
    """Run mean-reversion analysis visualization."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charts_dir = OUTPUT_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Load feature matrix
    logging.info("Loading feature matrix...")
    df = load_feature_matrix()
    logging.info(f"Loaded {len(df)} rows, {df['week_idx'].nunique()} weeks")
    
    # Get latest week data
    logging.info("Extracting latest week data...")
    latest_week = get_latest_week_data(df)
    logging.info(f"Latest week: {latest_week['week_start'].iloc[0]} with {len(latest_week)} symbols")
    
    # Create return histogram
    logging.info("Creating return histogram...")
    histogram_path = create_return_histogram(latest_week, OUTPUT_DIR)
    
    # Get worst 5% performers
    n_worst = int(len(latest_week) * WORST_PERCENTILE)
    worst_performers = latest_week.nsmallest(n_worst, "log_return").reset_index(drop=True)
    logging.info(f"Identified {len(worst_performers)} symbols in bottom {WORST_PERCENTILE*100:.0f}%")
    
    # Create candlestick charts for worst performers
    logging.info("Creating candlestick charts...")
    chart_files = []
    
    for idx, row in worst_performers.iterrows():
        symbol = row["symbol"]
        name = row.get("name", symbol)
        
        # Load daily data
        daily_df = load_daily_data(symbol)
        if daily_df is None or len(daily_df) < BB_PERIOD:
            logging.warning(f"Skipping {symbol}: insufficient daily data")
            continue
        
        # Extract features for this symbol
        features = {feat: row.get(feat) for feat in HOVER_FEATURES}
        
        # Create chart
        chart_path = create_candlestick_chart(
            symbol=symbol,
            name=name,
            daily_df=daily_df,
            week_return=row["log_return"],
            week_pct_return=row["pct_return"],
            features=features,
            output_dir=charts_dir,
        )
        
        chart_files.append((symbol, chart_path))
        
        if (idx + 1) % 10 == 0:
            logging.info(f"  Created {idx + 1}/{len(worst_performers)} charts")
    
    logging.info(f"Created {len(chart_files)} candlestick charts")
    
    # Create inspector HTML
    logging.info("Creating inspector...")
    inspector_path = create_inspector_html(worst_performers, chart_files, OUTPUT_DIR)
    
    # Save summary
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "generated_at": datetime.now().isoformat(),
        "week_start": str(latest_week["week_start"].iloc[0]),
        "total_symbols": len(latest_week),
        "worst_percentile": WORST_PERCENTILE,
        "n_worst_performers": len(worst_performers),
        "n_charts_generated": len(chart_files),
        "files": {
            "histogram": str(histogram_path),
            "inspector": str(inspector_path),
            "charts_dir": str(charts_dir),
        },
    }
    
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save worst performers data
    worst_performers.to_csv(OUTPUT_DIR / "worst_performers.csv", index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"MEAN-REVERSION ANALYSIS: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    print(f"\nWeek of: {latest_week['week_start'].iloc[0]}")
    print(f"Total symbols: {len(latest_week)}")
    print(f"Bottom 5% ({len(worst_performers)} symbols):")
    print("-" * 50)
    
    for idx, row in worst_performers.head(10).iterrows():
        print(f"  {row['symbol']:<6} {row['pct_return']:>7.2f}%  {str(row.get('name', ''))[:30]}")
    
    if len(worst_performers) > 10:
        print(f"  ... and {len(worst_performers) - 10} more")
    
    print("\n" + "-" * 50)
    print("Output files:")
    print(f"  Histogram:  {histogram_path}")
    print(f"  Inspector:  {inspector_path}")
    print(f"  Charts:     {charts_dir}")
    print("=" * 70)
    
    logging.info("Analysis complete")


if __name__ == "__main__":
    main()

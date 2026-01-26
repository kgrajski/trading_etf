#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate feature visualization dashboards for each symbol using Plotly.

Creates interactive multi-panel HTML dashboards showing:
- Panel 1: Price (close + high/low range)
- Panel 2: Volume (weekly volume)
- Panel 3: Returns (log_return + moving averages)
- Panel 4: Momentum (4-week and 12-week momentum)
- Panel 5: Volatility (intra-week + moving averages)
- Panel 6: Volume Dynamics (log_volume_delta + trend)

Processes both target ETFs and macro symbols.
Also generates an HTML inspector for browsing all dashboards.

Input: data/historical/{tier}/weekly/*.csv
Output: data/visualizations/{tier}/*.html, _inspector.html
"""

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Load metadata for targets and macro symbols.

    Combines:
    1. Target ETFs from filtered_etfs.json
    2. Macro symbols from config

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        Dict mapping symbol to {"name": ..., "category": ...}
    """
    result: Dict[str, Dict[str, str]] = {}

    # Load target ETFs from filtered_etfs.json
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

    # Add macro symbols from config
    for category, symbols in MACRO_SYMBOLS.items():
        for symbol, description in symbols.items():
            result[symbol] = {
                "name": description,
                "category": category,
            }

    return result


def get_symbols_to_process(input_dir: Path) -> List[Path]:
    """Get list of CSV files to process.

    Applies SYMBOL_PREFIX_FILTER to target ETFs only.
    Macro symbols are always processed.

    Args:
        input_dir: Directory containing weekly CSV files

    Returns:
        List of CSV file paths to process
    """
    all_csv_files = sorted(input_dir.glob("*.csv"))
    # Exclude manifest and other metadata files
    all_csv_files = [f for f in all_csv_files if not f.name.startswith("_")]

    # Get list of macro symbols
    macro_symbols: Set[str] = set(MACRO_SYMBOL_LIST)

    if SYMBOL_PREFIX_FILTER:
        # Apply filter to targets only, always include macros
        csv_files = [
            f
            for f in all_csv_files
            if f.stem.startswith(SYMBOL_PREFIX_FILTER) or f.stem in macro_symbols
        ]
    else:
        csv_files = all_csv_files

    return csv_files


def create_dashboard(
    df: pd.DataFrame, symbol: str, name: str = "", category: str = "target"
) -> go.Figure:
    """Create multi-panel Plotly dashboard for a symbol.

    Args:
        df: DataFrame with weekly feature data
        symbol: Symbol ticker
        name: Full name/description of the symbol
        category: Category (target, volatility, treasury, etc.)

    Returns:
        Plotly figure object
    """
    df["date_str"] = df["week_start"].dt.strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "<b>Price</b><br><sup>Weekly close with high/low range</sup>",
            "<b>Volume</b><br><sup>Weekly trading volume (millions)</sup>",
            "<b>Returns</b><br><sup>Week-over-week: log(close_t / close_{t-1})</sup>",
            "<b>Momentum</b><br><sup>log(close_t / close_{t-n})</sup>",
            "<b>Volatility</b><br><sup>Std of daily log returns</sup>",
            "<b>Volume Change</b><br><sup>Week-over-week log volume delta</sup>",
        ),
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        shared_xaxes=True,
    )

    dates = df["week_start"]

    # Panel 1: Price
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["high"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["low"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=VIZ_COLORS["price_fill"],
            name="High/Low Range",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(color=VIZ_COLORS["price"], width=2),
            customdata=df["date_str"],
            hovertemplate="%{customdata}<br>Close: $%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Volume
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["volume"] / 1e6,
            name="Volume (M)",
            marker_color=VIZ_COLORS["volume"],
            opacity=0.7,
            customdata=df["date_str"],
            hovertemplate="%{customdata}<br>Volume: %{y:.1f}M<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Panel 3: Returns
    colors = [
        VIZ_COLORS["positive"] if r >= 0 else VIZ_COLORS["negative"]
        for r in df["log_return"].fillna(0)
    ]
    fig.add_trace(
        go.Bar(
            x=dates,
            y=df["log_return"],
            name="Log Return",
            marker_color=colors,
            opacity=0.7,
            customdata=df["date_str"],
            hovertemplate="%{customdata}<br>Return: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if "log_return_ma4" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["log_return_ma4"],
                mode="lines",
                name="Return MA4",
                line=dict(color=VIZ_COLORS["ma4"], width=2),
                hovertemplate="MA4: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if "log_return_ma12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["log_return_ma12"],
                mode="lines",
                name="Return MA12",
                line=dict(color=VIZ_COLORS["ma12"], width=2),
                hovertemplate="MA12: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 4: Momentum
    if "momentum_4w" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["momentum_4w"],
                mode="lines",
                name="Momentum 4W",
                line=dict(color=VIZ_COLORS["ma4"], width=2),
                hovertemplate="4W: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    if "momentum_12w" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["momentum_12w"],
                mode="lines",
                name="Momentum 12W",
                line=dict(color=VIZ_COLORS["ma12"], width=2),
                hovertemplate="12W: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    # Panel 5: Volatility
    if "intra_week_volatility" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["intra_week_volatility"],
                mode="lines",
                name="Weekly Vol",
                line=dict(color=VIZ_COLORS["price"], width=1),
                opacity=0.7,
                hovertemplate="Vol: %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    if "volatility_ma4" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["volatility_ma4"],
                mode="lines",
                name="Vol MA4",
                line=dict(color=VIZ_COLORS["ma4"], width=2),
                hovertemplate="MA4: %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    if "volatility_ma12" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["volatility_ma12"],
                mode="lines",
                name="Vol MA12",
                line=dict(color=VIZ_COLORS["ma12"], width=2),
                hovertemplate="MA12: %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Panel 6: Volume Change
    if "log_volume_delta" in df.columns:
        delta_values = df["log_volume_delta"].fillna(0)
        colors = [
            VIZ_COLORS["positive"] if v >= 0 else VIZ_COLORS["negative"]
            for v in delta_values
        ]
        fig.add_trace(
            go.Bar(
                x=dates,
                y=delta_values,
                name="Vol Delta",
                marker_color=colors,
                opacity=0.7,
                customdata=df["date_str"],
                hovertemplate="%{customdata}<br>Δ Volume: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=2,
        )
        delta_ma4 = delta_values.rolling(window=4, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=delta_ma4,
                mode="lines",
                name="Delta MA4",
                line=dict(color=VIZ_COLORS["ma4"], width=2),
                hovertemplate="Trend: %{y:.3f}<extra></extra>",
            ),
            row=3,
            col=2,
        )

    # Zero lines
    for row, col in [(2, 1), (2, 2), (3, 2)]:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=VIZ_COLORS["zero"],
            opacity=0.5,
            row=row,
            col=col,
        )

    # Layout
    date_range = (
        f"{df['week_start'].min().strftime('%Y-%m-%d')} to "
        f"{df['week_start'].max().strftime('%Y-%m-%d')}"
    )

    # Build title with category badge for macro symbols
    if category != "target":
        category_badge = f"[{category.upper()}]"
        title_text = f"<b>{symbol}</b> {category_badge}<br><sup>{name}</sup>"
    elif name:
        title_text = f"<b>{symbol}</b><br><sup>{name}</sup>"
    else:
        title_text = f"<b>{symbol}</b>"

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center", font=dict(size=18)),
        height=950,
        showlegend=False,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(t=100, b=40, l=60, r=40),
    )

    fig.add_annotation(
        text=f"{date_range} ({len(df)} weeks)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.02,
        showarrow=False,
        font=dict(size=11, color="#888"),
        xanchor="center",
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume (M)", row=1, col=2)
    fig.update_yaxes(title_text="Log Return", row=2, col=1)
    fig.update_yaxes(title_text="Log Momentum", row=2, col=2)
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_yaxes(title_text="Δ Log Volume", row=3, col=2)

    fig.update_xaxes(matches="x5")

    return fig


def save_dashboard(fig: go.Figure, output_path: str, compress: bool = True) -> str:
    """Save Plotly figure as HTML.

    Args:
        fig: Plotly figure
        output_path: Path for output file
        compress: Whether to gzip compress

    Returns:
        Actual path where file was saved
    """
    html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)

    if compress:
        output_path = output_path + ".gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            f.write(html_content)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    return output_path


def create_inspector_html(
    html_files: List[Dict[str, str]],
    output_path: str,
    data_tier: str,
    compressed: bool,
) -> None:
    """Create HTML inspector for browsing all dashboards.

    Args:
        html_files: List of dicts with symbol, path, category
        output_path: Path for inspector HTML
        data_tier: Data tier name
        compressed: Whether dashboards are compressed
    """
    # Sort by category then symbol
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
    <title>ETF Feature Inspector - {data_tier.upper()}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               background: #1a1a2e; color: #eee; min-height: 100vh; }}
        .header {{ background: #16213e; padding: 15px 20px; display: flex;
                  justify-content: space-between; align-items: center;
                  border-bottom: 2px solid #0f3460; position: fixed;
                  top: 0; left: 0; right: 0; z-index: 1000; }}
        .title {{ font-size: 1.4em; font-weight: bold; color: #e94560; }}
        .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
        .nav-btn {{ background: #0f3460; color: #fff; border: none; padding: 10px 20px;
                   border-radius: 5px; cursor: pointer; font-size: 1em; }}
        .nav-btn:hover {{ background: #e94560; }}
        .nav-btn:disabled {{ background: #333; cursor: not-allowed; }}
        .search-box, .symbol-select, .category-select {{ padding: 10px; border-radius: 5px;
                                       border: 1px solid #0f3460; background: #16213e; color: #fff; }}
        .category-select {{ min-width: 100px; }}
        .progress {{ color: #888; font-size: 0.9em; }}
        .category-badge {{ padding: 2px 8px; border-radius: 3px; font-size: 0.8em; margin-left: 5px; }}
        .cat-target {{ background: #0f3460; }}
        .cat-volatility {{ background: #e94560; }}
        .cat-treasury {{ background: #28a745; }}
        .cat-dollar {{ background: #ffc107; color: #000; }}
        .cat-commodities {{ background: #17a2b8; }}
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
        <div class="title">📈 ETF Feature Inspector ({data_tier.upper()})</div>
        <div class="controls">
            <a href="_inspector_scatter.html" class="view-toggle">📊 Scatter View</a>
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


@workflow_script("04-visualize-features")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    input_dir = get_historical_dir(DATA_TIER) / "weekly"
    output_dir = get_visualizations_dir(DATA_TIER)
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata for all symbols (targets + macro)
    symbol_metadata = load_all_symbol_metadata(metadata_dir)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter (targets only): {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Compress HTML: {COMPRESS_HTML}")
    print()

    # Get input files
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run 03-generate-features.py first.")
        return

    csv_files = get_symbols_to_process(input_dir)

    # Count by category
    target_count = sum(
        1 for f in csv_files if f.stem not in MACRO_SYMBOL_LIST
    )
    macro_count = sum(
        1 for f in csv_files if f.stem in MACRO_SYMBOL_LIST
    )

    print(f"Found {len(csv_files)} weekly CSV files to visualize")
    print(f"  Target ETFs: {target_count}")
    print(f"  Macro symbols: {macro_count}")
    print("-" * 80)

    if not csv_files:
        logger.warning("No files to process. Exiting.")
        return

    # Clean up old files
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()

    # Generate dashboards
    success_count = 0
    fail_count = 0
    html_files: List[Dict[str, str]] = []

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

        try:
            fig = create_dashboard(df, symbol, name, category)
            output_path = str(output_dir / f"{symbol}.html")
            actual_path = save_dashboard(fig, output_path, compress=COMPRESS_HTML)
            html_files.append({
                "symbol": symbol,
                "path": Path(actual_path).name,
                "category": category,
            })
            success_count += 1
            size_kb = os.path.getsize(actual_path) / 1024
            print(f"✓ ({size_kb:.0f} KB)")
        except Exception as e:
            logger.error(f"Error: {e}")
            fail_count += 1

    # Create inspector
    print()
    print("Creating HTML inspector...")
    inspector_path = output_dir / "_inspector.html"
    create_inspector_html(html_files, str(inspector_path), DATA_TIER, COMPRESS_HTML)
    print(f"  Inspector: {inspector_path}")

    # Summary
    print_summary(
        dashboards_created=success_count,
        target_etfs=target_count,
        macro_symbols=macro_count,
        failed=fail_count,
        output_directory=str(output_dir),
        inspector=str(inspector_path),
    )


if __name__ == "__main__":
    main()

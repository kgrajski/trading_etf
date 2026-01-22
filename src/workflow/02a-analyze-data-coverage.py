#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze data coverage across the ETF universe.

This script analyzes the temporal coverage of daily data to help find the
"sweet spot" for lookback period vs. symbol count trade-off.

Generates:
1. Histogram: Number of symbols by trading duration (weeks/years)
2. Time series: Number of active symbols per week over time
3. Sweet spot table: Symbols with complete data for various lookback periods
4. Retrospective: New ETFs by quarter/year with description analysis

Input: data/historical/{tier}/daily/*.csv
Output: data/visualizations/{tier}/coverage_analysis.html
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import (
    DATA_TIER,
    MACRO_SYMBOL_LIST,
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


def load_symbol_metadata(metadata_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load symbol metadata from filtered_etfs.json.

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        Dict mapping symbol to metadata (name, sponsor, etc.)
    """
    result: Dict[str, Dict[str, str]] = {}

    # Load filtered ETFs
    filtered_file = metadata_dir / "filtered_etfs.json"
    if filtered_file.exists():
        with open(filtered_file) as f:
            data = json.load(f)
        for etf in data.get("etfs", []):
            result[etf["symbol"]] = {
                "name": etf.get("name", ""),
                "sponsor": etf.get("sponsor", "Unknown"),
                "category": "target",
            }

    return result


def scan_daily_data(daily_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan all daily CSV files and extract date ranges.

    Args:
        daily_dir: Directory containing daily CSV files

    Returns:
        Dict mapping symbol to {first_date, last_date, trading_days, weeks}
    """
    coverage: Dict[str, Dict[str, Any]] = {}

    csv_files = sorted(daily_dir.glob("*.csv"))
    print(f"Scanning {len(csv_files)} daily CSV files...")

    for csv_file in csv_files:
        symbol = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            if df.empty or "date" not in df.columns:
                continue

            df["date"] = pd.to_datetime(df["date"])
            first_date = df["date"].min()
            last_date = df["date"].max()
            trading_days = len(df)
            weeks = (last_date - first_date).days // 7

            coverage[symbol] = {
                "first_date": first_date,
                "last_date": last_date,
                "trading_days": trading_days,
                "weeks": weeks,
                "is_macro": symbol in MACRO_SYMBOL_LIST,
            }
        except Exception as e:
            logger.warning(f"Error reading {symbol}: {e}")

    return coverage


def compute_weekly_symbol_counts(
    coverage: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Compute number of active symbols per week.

    Args:
        coverage: Coverage data from scan_daily_data

    Returns:
        DataFrame with week_start as index and symbol count as column
    """
    # Find overall date range
    all_first = min(c["first_date"] for c in coverage.values())
    all_last = max(c["last_date"] for c in coverage.values())

    # Generate weekly date range
    weeks = pd.date_range(start=all_first, end=all_last, freq="W-MON")

    # Count symbols active each week
    counts = []
    for week in weeks:
        week_end = week + pd.Timedelta(days=6)
        active = sum(
            1
            for c in coverage.values()
            if c["first_date"] <= week_end and c["last_date"] >= week
        )
        active_targets = sum(
            1
            for c in coverage.values()
            if c["first_date"] <= week_end
            and c["last_date"] >= week
            and not c["is_macro"]
        )
        counts.append(
            {"week_start": week, "total_symbols": active, "target_symbols": active_targets}
        )

    return pd.DataFrame(counts).set_index("week_start")


def compute_lookback_analysis(
    coverage: Dict[str, Dict[str, Any]], reference_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Compute symbols with complete data for various lookback periods.

    Args:
        coverage: Coverage data from scan_daily_data
        reference_date: End date for lookback (default: latest date in data)

    Returns:
        DataFrame with lookback periods and symbol counts
    """
    if reference_date is None:
        reference_date = max(c["last_date"] for c in coverage.values())

    lookback_weeks = [13, 26, 52, 78, 104, 156, 208, 260]  # 3mo to 5yr
    lookback_labels = ["3 months", "6 months", "1 year", "1.5 years", "2 years", "3 years", "4 years", "5 years"]

    results = []
    for weeks, label in zip(lookback_weeks, lookback_labels):
        start_date = reference_date - pd.Timedelta(weeks=weeks)

        # Count symbols with data spanning the full lookback
        complete_all = sum(
            1
            for c in coverage.values()
            if c["first_date"] <= start_date and c["last_date"] >= reference_date
        )
        complete_targets = sum(
            1
            for c in coverage.values()
            if c["first_date"] <= start_date
            and c["last_date"] >= reference_date
            and not c["is_macro"]
        )

        results.append(
            {
                "lookback_weeks": weeks,
                "lookback_label": label,
                "total_symbols": complete_all,
                "target_symbols": complete_targets,
            }
        )

    return pd.DataFrame(results)


def compute_etf_launches_by_period(
    coverage: Dict[str, Dict[str, Any]], symbol_metadata: Dict[str, Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
    """Group new ETFs by quarter/year.

    Args:
        coverage: Coverage data from scan_daily_data
        symbol_metadata: Symbol metadata with names and sponsors

    Returns:
        Dict mapping period to list of ETFs launched
    """
    # Group by quarter
    launches: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for symbol, cov in coverage.items():
        if cov["is_macro"]:
            continue

        first_date = cov["first_date"]
        quarter = f"{first_date.year}-Q{(first_date.month - 1) // 3 + 1}"

        meta = symbol_metadata.get(symbol, {})
        launches[quarter].append(
            {
                "symbol": symbol,
                "name": meta.get("name", ""),
                "sponsor": meta.get("sponsor", "Unknown"),
                "first_date": first_date.strftime("%Y-%m-%d"),
            }
        )

    return dict(launches)


def create_coverage_dashboard(
    coverage: Dict[str, Dict[str, Any]],
    weekly_counts: pd.DataFrame,
    lookback_df: pd.DataFrame,
    launches: Dict[str, List[Dict[str, str]]],
) -> go.Figure:
    """Create multi-panel coverage analysis dashboard.

    Args:
        coverage: Coverage data
        weekly_counts: Weekly symbol counts
        lookback_df: Lookback analysis
        launches: ETF launches by period

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>Symbols by Trading Duration</b>",
            "<b>Active Symbols Over Time</b>",
            "<b>Sweet Spot Analysis</b>",
            "<b>New ETFs by Quarter</b>",
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    colors = {
        "primary": "#2E86AB",
        "secondary": "#A23B72",
        "accent": "#F18F01",
    }

    # Panel 1: Duration histogram
    target_weeks = [c["weeks"] for c in coverage.values() if not c["is_macro"]]
    year_bins = [0, 26, 52, 104, 156, 208, 260, 1000]
    year_labels = ["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-4yr", "4-5yr", "5yr+"]
    hist_counts = []
    for i in range(len(year_bins) - 1):
        count = sum(1 for w in target_weeks if year_bins[i] <= w < year_bins[i + 1])
        hist_counts.append(count)

    fig.add_trace(
        go.Bar(
            x=year_labels,
            y=hist_counts,
            marker_color=colors["primary"],
            name="Target ETFs",
            hovertemplate="%{x}: %{y} ETFs<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Symbols over time
    fig.add_trace(
        go.Scatter(
            x=weekly_counts.index,
            y=weekly_counts["target_symbols"],
            mode="lines",
            name="Target Symbols",
            line=dict(color=colors["primary"], width=2),
            hovertemplate="%{x|%Y-%m-%d}: %{y} symbols<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Panel 3: Sweet spot analysis
    fig.add_trace(
        go.Bar(
            x=lookback_df["lookback_label"],
            y=lookback_df["target_symbols"],
            marker_color=colors["secondary"],
            name="Complete Data",
            hovertemplate="%{x}: %{y} symbols<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Panel 4: New ETFs by quarter
    quarters = sorted(launches.keys())[-12:]  # Last 12 quarters
    quarter_counts = [len(launches.get(q, [])) for q in quarters]

    fig.add_trace(
        go.Bar(
            x=quarters,
            y=quarter_counts,
            marker_color=colors["accent"],
            name="New ETFs",
            hovertemplate="%{x}: %{y} new ETFs<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>ETF Data Coverage Analysis</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        height=800,
        showlegend=False,
        template="plotly_dark",
        margin=dict(t=80, b=40, l=60, r=40),
    )

    fig.update_yaxes(title_text="Number of ETFs", row=1, col=1)
    fig.update_yaxes(title_text="Active Symbols", row=1, col=2)
    fig.update_yaxes(title_text="Symbols with Complete Data", row=2, col=1)
    fig.update_yaxes(title_text="New ETFs Launched", row=2, col=2)

    fig.update_xaxes(title_text="Trading Duration", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Lookback Period", row=2, col=1, tickangle=45)
    fig.update_xaxes(title_text="Quarter", row=2, col=2, tickangle=45)

    return fig


@workflow_script("02a-analyze-data-coverage")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    daily_dir = get_historical_dir(DATA_TIER) / "daily"
    output_dir = get_visualizations_dir(DATA_TIER)
    os.makedirs(output_dir, exist_ok=True)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Daily data: {daily_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Check if data exists
    if not daily_dir.exists():
        logger.error(f"Daily data directory not found: {daily_dir}")
        logger.error("Please run 02-fetch-daily-data.py first.")
        return

    # Load metadata
    print("Loading symbol metadata...")
    symbol_metadata = load_symbol_metadata(metadata_dir)
    print(f"  Loaded metadata for {len(symbol_metadata)} symbols")
    print()

    # Scan daily data
    coverage = scan_daily_data(daily_dir)
    target_coverage = {k: v for k, v in coverage.items() if not v["is_macro"]}
    macro_coverage = {k: v for k, v in coverage.items() if v["is_macro"]}

    print()
    print(f"Coverage summary:")
    print(f"  Total symbols: {len(coverage)}")
    print(f"  Target ETFs: {len(target_coverage)}")
    print(f"  Macro symbols: {len(macro_coverage)}")
    print()

    if not coverage:
        logger.error("No data found. Please run 02-fetch-daily-data.py first.")
        return

    # Date range
    all_first = min(c["first_date"] for c in coverage.values())
    all_last = max(c["last_date"] for c in coverage.values())
    print(f"Date range: {all_first.date()} to {all_last.date()}")
    print()

    # Compute analyses
    print("Computing weekly symbol counts...")
    weekly_counts = compute_weekly_symbol_counts(coverage)

    print("Computing lookback analysis...")
    lookback_df = compute_lookback_analysis(coverage)

    print("Grouping ETF launches by period...")
    launches = compute_etf_launches_by_period(coverage, symbol_metadata)

    # Print sweet spot table
    print()
    print("=" * 60)
    print("SWEET SPOT ANALYSIS")
    print("=" * 60)
    print(f"{'Lookback':<15} {'Target ETFs':>15} {'Total Symbols':>15}")
    print("-" * 60)
    for _, row in lookback_df.iterrows():
        print(
            f"{row['lookback_label']:<15} "
            f"{row['target_symbols']:>15} "
            f"{row['total_symbols']:>15}"
        )
    print()

    # Print duration breakdown
    print("=" * 60)
    print("DURATION BREAKDOWN (Target ETFs Only)")
    print("=" * 60)
    target_weeks = [c["weeks"] for c in target_coverage.values()]
    print(f"  Median duration: {pd.Series(target_weeks).median():.0f} weeks")
    print(f"  Mean duration: {pd.Series(target_weeks).mean():.0f} weeks")
    print(f"  Min duration: {min(target_weeks)} weeks")
    print(f"  Max duration: {max(target_weeks)} weeks")
    print()

    # Create dashboard
    print("Creating coverage dashboard...")
    fig = create_coverage_dashboard(coverage, weekly_counts, lookback_df, launches)

    output_path = output_dir / "coverage_analysis.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"  Dashboard: {output_path}")

    # Save detailed analysis
    analysis_data = {
        "generated_at": datetime.now().isoformat(),
        "data_tier": DATA_TIER,
        "date_range": {
            "start": str(all_first.date()),
            "end": str(all_last.date()),
        },
        "symbol_counts": {
            "total": len(coverage),
            "targets": len(target_coverage),
            "macros": len(macro_coverage),
        },
        "lookback_analysis": lookback_df.to_dict(orient="records"),
        "launches_by_quarter": {
            q: len(etfs) for q, etfs in launches.items()
        },
    }

    analysis_path = output_dir / "coverage_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  Analysis data: {analysis_path}")
    print()

    # Summary
    print_summary(
        total_symbols=len(coverage),
        target_etfs=len(target_coverage),
        macro_symbols=len(macro_coverage),
        date_range=f"{all_first.date()} to {all_last.date()}",
        dashboard=str(output_path),
    )


if __name__ == "__main__":
    main()

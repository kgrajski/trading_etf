#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF Universe Retrospective Analysis.

Generates a retrospective report characterizing the ETF universe:
- Quarterly breakdown of new ETFs
- Category/theme analysis from descriptions
- Sponsor market share trends
- Asset class and sector classification

This script analyzes ETF metadata and trading history to provide
insights into the composition and evolution of the ETF universe.

Input:
  - data/metadata/filtered_etfs.json (ETF metadata)
  - data/historical/{tier}/daily/*.csv (trading history)

Output:
  - reports/etf_retrospective_{date}.md (Markdown report)
  - data/visualizations/{tier}/etf_retrospective.html (Dashboard)
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import DATA_TIER, MACRO_SYMBOL_LIST
from src.workflow.workflow_utils import (
    get_historical_dir,
    get_metadata_dir,
    get_visualizations_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()

# =============================================================================
# CATEGORY DEFINITIONS
# =============================================================================

# Keywords for categorizing ETFs by sector/theme
SECTOR_KEYWORDS = {
    "Technology": [
        "tech", "technology", "software", "semiconductor", "internet",
        "cloud", "cyber", "digital", "ai", "artificial intelligence",
    ],
    "Healthcare": [
        "health", "healthcare", "biotech", "pharmaceutical", "medical",
        "genomic", "drug", "therapeutics",
    ],
    "Financials": [
        "financial", "bank", "insurance", "reit", "real estate",
        "mortgage", "capital markets",
    ],
    "Energy": [
        "energy", "oil", "gas", "petroleum", "crude", "natural gas",
        "clean energy", "solar", "wind", "renewable",
    ],
    "Consumer": [
        "consumer", "retail", "discretionary", "staples", "food",
        "beverage", "apparel",
    ],
    "Industrials": [
        "industrial", "aerospace", "defense", "manufacturing",
        "transportation", "infrastructure",
    ],
    "Materials": [
        "materials", "metals", "mining", "gold", "silver", "copper",
        "steel", "chemicals",
    ],
    "Utilities": [
        "utility", "utilities", "electric", "water", "power",
    ],
    "Communications": [
        "communication", "telecom", "media", "entertainment",
    ],
}

STYLE_KEYWORDS = {
    "Growth": ["growth"],
    "Value": ["value"],
    "Dividend": ["dividend", "income", "yield"],
    "Momentum": ["momentum"],
    "Quality": ["quality"],
    "Low Volatility": ["low volatility", "minimum volatility", "low vol"],
    "ESG": ["esg", "sustainable", "green", "clean", "climate", "carbon"],
}

SIZE_KEYWORDS = {
    "Large Cap": ["large cap", "large-cap", "mega cap", "s&p 500", "s&p500"],
    "Mid Cap": ["mid cap", "mid-cap", "midcap", "s&p 400"],
    "Small Cap": ["small cap", "small-cap", "smallcap", "s&p 600", "russell 2000"],
    "Micro Cap": ["micro cap", "micro-cap"],
}

ASSET_CLASS_KEYWORDS = {
    "Equity": ["equity", "stock", "shares"],
    "Fixed Income": [
        "bond", "treasury", "corporate bond", "municipal", "fixed income",
        "credit", "debt", "tips", "t-bill",
    ],
    "Commodity": ["commodity", "gold", "silver", "oil", "agriculture", "metals"],
    "Real Estate": ["real estate", "reit", "property"],
    "Multi-Asset": ["multi-asset", "balanced", "allocation"],
}

GEOGRAPHY_KEYWORDS = {
    "US": ["u.s.", "us ", "american", "domestic"],
    "International": ["international", "global", "world", "developed"],
    "Emerging Markets": ["emerging", "em ", "frontier"],
    "Europe": ["europe", "european", "euro"],
    "Asia Pacific": ["asia", "pacific", "japan", "china", "india"],
    "Latin America": ["latin", "brazil", "mexico"],
}


def categorize_etf(name: str) -> Dict[str, List[str]]:
    """Categorize an ETF based on its name/description.

    Args:
        name: ETF name/description

    Returns:
        Dict with category types and matched categories
    """
    name_lower = name.lower()
    categories: Dict[str, List[str]] = {
        "sector": [],
        "style": [],
        "size": [],
        "asset_class": [],
        "geography": [],
    }

    # Check each category type
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            categories["sector"].append(sector)

    for style, keywords in STYLE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            categories["style"].append(style)

    for size, keywords in SIZE_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            categories["size"].append(size)

    for asset_class, keywords in ASSET_CLASS_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            categories["asset_class"].append(asset_class)

    for geo, keywords in GEOGRAPHY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            categories["geography"].append(geo)

    return categories


def load_etf_metadata(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load ETF metadata from filtered_etfs.json.

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        List of ETF dictionaries
    """
    filtered_file = metadata_dir / "filtered_etfs.json"
    if not filtered_file.exists():
        logger.error(f"Filtered ETFs file not found: {filtered_file}")
        return []

    with open(filtered_file) as f:
        data = json.load(f)

    return data.get("etfs", [])


def get_etf_date_ranges(daily_dir: Path) -> Tuple[Dict[str, datetime], datetime, datetime]:
    """Get first trading date for each ETF and overall data coverage.

    Args:
        daily_dir: Directory containing daily CSV files

    Returns:
        Tuple of (first_dates dict, overall_min_date, overall_max_date)
    """
    first_dates: Dict[str, datetime] = {}
    overall_max_date: Optional[datetime] = None

    for csv_file in daily_dir.glob("*.csv"):
        symbol = csv_file.stem
        if symbol in MACRO_SYMBOL_LIST:
            continue

        try:
            df = pd.read_csv(csv_file, usecols=["date"])
            if not df.empty:
                dates = pd.to_datetime(df["date"])
                first_dates[symbol] = dates.min()
                file_max = dates.max()
                if overall_max_date is None or file_max > overall_max_date:
                    overall_max_date = file_max
        except Exception as e:
            logger.warning(f"Error reading {symbol}: {e}")

    overall_min_date = min(first_dates.values()) if first_dates else datetime.now()
    if overall_max_date is None:
        overall_max_date = datetime.now()

    return first_dates, overall_min_date, overall_max_date


def build_quarterly_report(
    etfs: List[Dict[str, Any]], first_dates: Dict[str, datetime]
) -> Dict[str, List[Dict[str, Any]]]:
    """Build quarterly breakdown of ETFs.

    Args:
        etfs: List of ETF metadata
        first_dates: Dict mapping symbol to first date

    Returns:
        Dict mapping quarter to list of ETFs
    """
    quarterly: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for etf in etfs:
        symbol = etf.get("symbol", "")
        if symbol not in first_dates:
            continue

        first_date = first_dates[symbol]
        quarter = f"{first_date.year}-Q{(first_date.month - 1) // 3 + 1}"

        categories = categorize_etf(etf.get("name", ""))

        quarterly[quarter].append({
            "symbol": symbol,
            "name": etf.get("name", ""),
            "sponsor": etf.get("sponsor", "Unknown"),
            "first_date": first_date.strftime("%Y-%m-%d"),
            "categories": categories,
        })

    return dict(quarterly)


def compute_category_trends(
    quarterly: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, int]]:
    """Compute category trends by year.

    Args:
        quarterly: Quarterly ETF breakdown

    Returns:
        Dict mapping category to {year: count}
    """
    # Aggregate by year
    yearly_sectors: Dict[str, Counter] = defaultdict(Counter)

    for quarter, etfs in quarterly.items():
        year = quarter.split("-")[0]
        for etf in etfs:
            for sector in etf["categories"].get("sector", []):
                yearly_sectors[sector][year] += 1

    return {sector: dict(counts) for sector, counts in yearly_sectors.items()}


def compute_sponsor_trends(
    quarterly: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Dict[str, int]]:
    """Compute sponsor trends by year.

    Args:
        quarterly: Quarterly ETF breakdown

    Returns:
        Dict mapping sponsor to {year: count}
    """
    yearly_sponsors: Dict[str, Counter] = defaultdict(Counter)

    for quarter, etfs in quarterly.items():
        year = quarter.split("-")[0]
        for etf in etfs:
            sponsor = etf["sponsor"]
            yearly_sponsors[sponsor][year] += 1

    return {sponsor: dict(counts) for sponsor, counts in yearly_sponsors.items()}


def generate_markdown_report(
    quarterly: Dict[str, List[Dict[str, Any]]],
    category_trends: Dict[str, Dict[str, int]],
    sponsor_trends: Dict[str, Dict[str, int]],
    total_etfs: int,
    launch_range: Tuple[str, str],
    data_range: Tuple[str, str],
) -> str:
    """Generate markdown retrospective report.

    Args:
        quarterly: Quarterly ETF breakdown
        category_trends: Category trends by year
        sponsor_trends: Sponsor trends by year
        total_etfs: Total number of ETFs
        launch_range: (earliest_launch, latest_launch) tuple
        data_range: (data_start, data_end) tuple

    Returns:
        Markdown report as string
    """
    lines = []

    # Header
    lines.append("# ETF Universe Retrospective Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Total ETFs Tracked:** {total_etfs}")
    lines.append(f"- **Data Coverage:** {data_range[0]} to {data_range[1]}")
    lines.append(f"- **ETF Launch Range:** {launch_range[0]} to {launch_range[1]} (first trading dates)")
    lines.append(f"- **Quarters Analyzed:** {len(quarterly)}")
    lines.append("")

    # Sponsor breakdown
    all_sponsors = Counter()
    for quarter_etfs in quarterly.values():
        for etf in quarter_etfs:
            all_sponsors[etf["sponsor"]] += 1

    lines.append("### Sponsor Breakdown")
    lines.append("")
    lines.append("| Sponsor | Count | % |")
    lines.append("|---------|-------|---|")
    total = sum(all_sponsors.values())
    for sponsor, count in all_sponsors.most_common():
        pct = count / total * 100 if total > 0 else 0
        lines.append(f"| {sponsor} | {count} | {pct:.1f}% |")
    lines.append("")

    # Category breakdown
    all_sectors = Counter()
    for quarter_etfs in quarterly.values():
        for etf in quarter_etfs:
            for sector in etf["categories"].get("sector", []):
                all_sectors[sector] += 1

    if all_sectors:
        lines.append("### Sector Breakdown")
        lines.append("")
        lines.append("| Sector | Count |")
        lines.append("|--------|-------|")
        for sector, count in all_sectors.most_common(10):
            lines.append(f"| {sector} | {count} |")
        lines.append("")

    lines.append("---")
    lines.append("")

    # Quarterly Details
    lines.append("## Quarterly Breakdown")
    lines.append("")

    for quarter in sorted(quarterly.keys(), reverse=True):
        etfs = quarterly[quarter]
        lines.append(f"### {quarter} ({len(etfs)} ETFs)")
        lines.append("")

        if etfs:
            lines.append("| Symbol | Name | Sponsor | Sectors |")
            lines.append("|--------|------|---------|---------|")

            for etf in sorted(etfs, key=lambda x: x["symbol"]):
                sectors = ", ".join(etf["categories"].get("sector", ["-"]))
                if not sectors:
                    sectors = "-"
                name = etf["name"][:50] + "..." if len(etf["name"]) > 50 else etf["name"]
                lines.append(f"| {etf['symbol']} | {name} | {etf['sponsor']} | {sectors} |")

            lines.append("")

    lines.append("---")
    lines.append("")

    # Sponsor Trends
    if sponsor_trends:
        lines.append("## Sponsor Trends by Year")
        lines.append("")

        years = sorted(
            set(y for counts in sponsor_trends.values() for y in counts.keys())
        )

        header = "| Sponsor | " + " | ".join(years) + " | Total |"
        lines.append(header)
        lines.append("|" + "|".join(["---"] * (len(years) + 2)) + "|")

        for sponsor in sorted(sponsor_trends.keys()):
            counts = sponsor_trends[sponsor]
            row_counts = [str(counts.get(y, 0)) for y in years]
            total = sum(counts.values())
            lines.append(f"| {sponsor} | " + " | ".join(row_counts) + f" | {total} |")

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `02b-etf-retrospective.py`*")

    return "\n".join(lines)


def create_retrospective_dashboard(
    quarterly: Dict[str, List[Dict[str, Any]]],
    category_trends: Dict[str, Dict[str, int]],
    sponsor_trends: Dict[str, Dict[str, int]],
) -> go.Figure:
    """Create retrospective analysis dashboard.

    Args:
        quarterly: Quarterly ETF breakdown
        category_trends: Category trends by year
        sponsor_trends: Sponsor trends by year

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "<b>New ETFs by Quarter</b>",
            "<b>Sponsor Market Share</b>",
            "<b>Sector Breakdown</b>",
            "<b>Sponsor Trends by Year</b>",
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
              "#95C623", "#4ECDC4", "#FF6B6B", "#45B7D1", "#96CEB4"]

    # Panel 1: New ETFs by quarter
    quarters = sorted(quarterly.keys())
    quarter_counts = [len(quarterly[q]) for q in quarters]

    fig.add_trace(
        go.Bar(
            x=quarters,
            y=quarter_counts,
            marker_color=colors[0],
            name="New ETFs",
            hovertemplate="%{x}: %{y} ETFs<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Sponsor pie chart
    sponsor_totals = Counter()
    for quarter_etfs in quarterly.values():
        for etf in quarter_etfs:
            sponsor_totals[etf["sponsor"]] += 1

    fig.add_trace(
        go.Pie(
            labels=list(sponsor_totals.keys()),
            values=list(sponsor_totals.values()),
            marker_colors=colors[:len(sponsor_totals)],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} ETFs<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Panel 3: Sector breakdown
    sector_totals = Counter()
    for quarter_etfs in quarterly.values():
        for etf in quarter_etfs:
            for sector in etf["categories"].get("sector", []):
                sector_totals[sector] += 1

    if sector_totals:
        sectors = [s for s, _ in sector_totals.most_common(8)]
        sector_counts = [sector_totals[s] for s in sectors]

        fig.add_trace(
            go.Bar(
                x=sectors,
                y=sector_counts,
                marker_color=colors[1],
                name="Sectors",
                hovertemplate="%{x}: %{y} ETFs<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Panel 4: Sponsor trends by year
    if sponsor_trends:
        years = sorted(
            set(y for counts in sponsor_trends.values() for y in counts.keys())
        )

        for i, (sponsor, counts) in enumerate(sponsor_trends.items()):
            year_counts = [counts.get(y, 0) for y in years]
            fig.add_trace(
                go.Bar(
                    x=years,
                    y=year_counts,
                    name=sponsor,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"{sponsor}<br>%{{x}}: %{{y}} ETFs<extra></extra>",
                ),
                row=2,
                col=2,
            )

    # Layout
    fig.update_layout(
        title=dict(
            text="<b>ETF Universe Retrospective</b>",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        height=800,
        template="plotly_dark",
        barmode="stack",
        margin=dict(t=80, b=40, l=60, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)

    return fig


@workflow_script("02b-etf-retrospective")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    daily_dir = get_historical_dir(DATA_TIER) / "daily"
    viz_dir = get_visualizations_dir(DATA_TIER)
    reports_dir = Path(__file__).parent.parent.parent / "reports"

    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Metadata: {metadata_dir}")
    print(f"  Daily data: {daily_dir}")
    print(f"  Reports: {reports_dir}")
    print()

    # Load ETF metadata
    print("Loading ETF metadata...")
    etfs = load_etf_metadata(metadata_dir)
    print(f"  Loaded {len(etfs)} ETFs")

    if not etfs:
        logger.error("No ETF metadata found. Run 00 and 01 first.")
        return

    # Get first trading dates and data coverage
    print("Scanning daily data for first trading dates...")
    first_dates, data_start, data_end = get_etf_date_ranges(daily_dir)
    print(f"  Found dates for {len(first_dates)} symbols")

    if not first_dates:
        logger.error("No daily data found. Run 02 first.")
        return

    # Date ranges
    launch_min = min(first_dates.values())
    launch_max = max(first_dates.values())
    launch_range = (launch_min.strftime("%Y-%m-%d"), launch_max.strftime("%Y-%m-%d"))
    data_range = (data_start.strftime("%Y-%m-%d"), data_end.strftime("%Y-%m-%d"))
    print(f"  ETF launch range: {launch_range[0]} to {launch_range[1]}")
    print(f"  Data coverage: {data_range[0]} to {data_range[1]}")
    print()

    # Build quarterly report
    print("Building quarterly breakdown...")
    quarterly = build_quarterly_report(etfs, first_dates)
    print(f"  Quarters with data: {len(quarterly)}")

    # Compute trends
    print("Computing category and sponsor trends...")
    category_trends = compute_category_trends(quarterly)
    sponsor_trends = compute_sponsor_trends(quarterly)

    # Category summary
    print()
    print("=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)

    all_sectors = Counter()
    all_styles = Counter()
    for quarter_etfs in quarterly.values():
        for etf in quarter_etfs:
            for sector in etf["categories"].get("sector", []):
                all_sectors[sector] += 1
            for style in etf["categories"].get("style", []):
                all_styles[style] += 1

    print("\nTop Sectors:")
    for sector, count in all_sectors.most_common(10):
        print(f"  {sector}: {count}")

    if all_styles:
        print("\nTop Styles:")
        for style, count in all_styles.most_common(5):
            print(f"  {style}: {count}")

    print()

    # Generate markdown report
    print("Generating markdown report...")
    report_content = generate_markdown_report(
        quarterly, category_trends, sponsor_trends, len(etfs), launch_range, data_range
    )

    report_filename = f"etf_retrospective_{datetime.now().strftime('%Y%m%d')}.md"
    report_path = reports_dir / report_filename
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"  Report: {report_path}")

    # Create dashboard
    print("Creating dashboard...")
    fig = create_retrospective_dashboard(quarterly, category_trends, sponsor_trends)

    dashboard_path = viz_dir / "etf_retrospective.html"
    fig.write_html(str(dashboard_path), include_plotlyjs="cdn")
    print(f"  Dashboard: {dashboard_path}")

    # Save JSON analysis
    analysis_data = {
        "generated_at": datetime.now().isoformat(),
        "data_coverage": {"start": data_range[0], "end": data_range[1]},
        "etf_launch_range": {"start": launch_range[0], "end": launch_range[1]},
        "total_etfs": len(etfs),
        "symbols_with_data": len(first_dates),
        "quarters": len(quarterly),
        "category_trends": category_trends,
        "sponsor_trends": sponsor_trends,
        "quarterly_counts": {q: len(e) for q, e in quarterly.items()},
    }

    analysis_path = viz_dir / "etf_retrospective.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  Analysis data: {analysis_path}")
    print()

    # Summary
    print_summary(
        total_etfs=len(etfs),
        symbols_with_data=len(first_dates),
        quarters_analyzed=len(quarterly),
        top_sectors=", ".join(s for s, _ in all_sectors.most_common(3)),
        report=str(report_path),
        dashboard=str(dashboard_path),
    )


if __name__ == "__main__":
    main()

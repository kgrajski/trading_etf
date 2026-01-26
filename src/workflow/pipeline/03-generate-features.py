#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate weekly features from daily bars.

Aggregates daily OHLCV data into weekly summaries with derived metrics:
- L0 (Base): Core OHLCV aggregation (open, high, low, close, volume, etc.)
- L1 (Derived): Log-transformed single-week metrics (log_return, log_range, etc.)
- L2 (Temporal): Multi-week rolling/smoothed metrics (momentum, MAs, etc.)

Processes both target ETFs and macro symbols uniformly.

For complete feature documentation including formulas and interpretations,
see: docs/FEATURES.md

Input: data/historical/{tier}/daily/*.csv
Output: data/historical/{tier}/weekly/*.csv, _manifest.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.workflow.config import (
    DATA_TIER,
    FEATURES_ENABLED,
    LOOKBACK_PERIODS,
    MACRO_SYMBOL_CATEGORIES,
    MACRO_SYMBOL_LIST,
    SYMBOL_PREFIX_FILTER,
)
from src.workflow.workflow_utils import (
    get_historical_dir,
    get_metadata_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


def load_fetch_manifest(metadata_dir: Path) -> Dict[str, Any]:
    """Load fetch manifest to get symbol categories.

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        Fetch manifest dictionary, or empty dict if not found
    """
    manifest_path = metadata_dir / "fetch_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def get_symbol_category(symbol: str, fetch_manifest: Dict[str, Any]) -> str:
    """Get category for a symbol.

    Priority:
    1. From fetch_manifest categories
    2. From MACRO_SYMBOL_CATEGORIES config
    3. Default to "target"

    Args:
        symbol: Symbol to look up
        fetch_manifest: Loaded fetch manifest

    Returns:
        Category string (e.g., "target", "volatility", "treasury")
    """
    # Check fetch manifest first
    categories = fetch_manifest.get("categories", {})
    if symbol in categories:
        return categories[symbol]

    # Check macro symbols from config
    if symbol in MACRO_SYMBOL_CATEGORIES:
        return MACRO_SYMBOL_CATEGORIES[symbol]

    # Default to target
    return "target"


def get_symbols_to_process(
    input_dir: Path, fetch_manifest: Dict[str, Any]
) -> List[Path]:
    """Get list of CSV files to process.

    Applies SYMBOL_PREFIX_FILTER to target ETFs only.
    Macro symbols are always processed.

    Args:
        input_dir: Directory containing daily CSV files
        fetch_manifest: Loaded fetch manifest

    Returns:
        List of CSV file paths to process
    """
    all_csv_files = sorted(input_dir.glob("*.csv"))

    # Get list of macro symbols
    macro_symbols: Set[str] = set(MACRO_SYMBOL_LIST)

    # Also check fetch manifest
    if "macro_symbols" in fetch_manifest:
        macro_symbols.update(fetch_manifest["macro_symbols"])

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


def load_daily_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load daily bar data from CSV.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with daily data, or None if error
    """
    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def compute_L0_base(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute L0 base weekly aggregates.

    Aggregates daily OHLCV data into weekly bars using ISO week calendar.

    Args:
        daily_df: DataFrame with daily bars

    Returns:
        DataFrame with weekly aggregates
    """
    df = daily_df.copy()
    df["iso_year"] = df["date"].dt.isocalendar().year
    df["iso_week"] = df["date"].dt.isocalendar().week

    weekly = df.groupby(["symbol", "iso_year", "iso_week"]).agg(
        week_start=("date", "min"),
        week_end=("date", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        trading_days=("date", "count"),
    )

    weekly = weekly.reset_index()

    # Convert to Monday of ISO week
    weekly["week_start"] = pd.to_datetime(
        weekly["iso_year"].astype(str) + "-W" + weekly["iso_week"].astype(str) + "-1",
        format="%G-W%V-%u",
    )

    weekly = weekly.sort_values("week_start").reset_index(drop=True)

    columns = [
        "symbol",
        "week_start",
        "week_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "trading_days",
    ]
    return weekly[columns]


def compute_L1_derived(weekly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute L1 derived single-week metrics.

    These are log-transformed features for better statistical properties.

    Args:
        weekly_df: DataFrame with weekly aggregates
        daily_df: Original daily data for volatility calculation

    Returns:
        DataFrame with L1 features added
    """
    df = weekly_df.copy()

    if FEATURES_ENABLED.get("L1_log_return"):
        # Week-over-week close-to-close return (primary return measure)
        # This is what we predict: log(close_t / close_{t-1})
        df["log_return"] = np.log(df["close"] / df["close"].shift(1).replace(0, np.nan))

    if FEATURES_ENABLED.get("L1_log_return_intraweek"):
        # Intra-week return: open to close within the same week
        # May have predictive value for momentum/reversal patterns
        df["log_return_intraweek"] = np.log(df["close"] / df["open"].replace(0, np.nan))

    if FEATURES_ENABLED.get("L1_log_range"):
        df["log_range"] = np.log(df["high"] / df["low"].replace(0, np.nan))

    if FEATURES_ENABLED.get("L1_log_volume"):
        df["log_volume"] = np.log(df["volume"].replace(0, np.nan))

    if FEATURES_ENABLED.get("L1_log_avg_daily_volume"):
        avg_daily_vol = df["volume"] / df["trading_days"]
        df["log_avg_daily_volume"] = np.log(avg_daily_vol.replace(0, np.nan))

    if FEATURES_ENABLED.get("L1_intra_week_volatility"):
        daily = daily_df.copy()
        daily["iso_year"] = daily["date"].dt.isocalendar().year
        daily["iso_week"] = daily["date"].dt.isocalendar().week
        daily["daily_log_return"] = np.log(daily["close"] / daily["close"].shift(1))

        vol_by_week = (
            daily.groupby(["iso_year", "iso_week"])["daily_log_return"]
            .std()
            .reset_index()
        )
        vol_by_week.columns = ["iso_year", "iso_week", "intra_week_volatility"]

        df["iso_year"] = df["week_start"].dt.isocalendar().year
        df["iso_week"] = df["week_start"].dt.isocalendar().week
        df = df.merge(vol_by_week, on=["iso_year", "iso_week"], how="left")
        df = df.drop(columns=["iso_year", "iso_week"])

    return df


def compute_L2_temporal(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute L2 temporal multi-week metrics.

    These are rolling/smoothed features using lookback windows.

    Args:
        weekly_df: DataFrame with L0 and L1 features

    Returns:
        DataFrame with L2 features added
    """
    df = weekly_df.copy()
    df = df.sort_values("week_start").reset_index(drop=True)

    short = LOOKBACK_PERIODS["short"]
    medium = LOOKBACK_PERIODS["medium"]
    long_ = LOOKBACK_PERIODS["long"]

    if FEATURES_ENABLED.get("L2_log_volume_delta"):
        df["log_volume_delta"] = df["log_volume"] - df["log_volume"].shift(short)

    if FEATURES_ENABLED.get("L2_log_return_ma4"):
        df["log_return_ma4"] = (
            df["log_return"].rolling(window=medium, min_periods=1).mean()
        )

    if FEATURES_ENABLED.get("L2_log_volume_ma4"):
        df["log_volume_ma4"] = (
            df["log_volume"].rolling(window=medium, min_periods=1).mean()
        )

    if FEATURES_ENABLED.get("L2_momentum_4w"):
        df["momentum_4w"] = np.log(
            df["close"] / df["close"].shift(medium).replace(0, np.nan)
        )

    if FEATURES_ENABLED.get("L2_volatility_ma4"):
        df["volatility_ma4"] = (
            df["intra_week_volatility"].rolling(window=medium, min_periods=1).mean()
        )

    if FEATURES_ENABLED.get("L2_log_return_ma12"):
        df["log_return_ma12"] = (
            df["log_return"].rolling(window=long_, min_periods=1).mean()
        )

    if FEATURES_ENABLED.get("L2_log_volume_ma12"):
        df["log_volume_ma12"] = (
            df["log_volume"].rolling(window=long_, min_periods=1).mean()
        )

    if FEATURES_ENABLED.get("L2_momentum_12w"):
        df["momentum_12w"] = np.log(
            df["close"] / df["close"].shift(long_).replace(0, np.nan)
        )

    if FEATURES_ENABLED.get("L2_volatility_ma12"):
        df["volatility_ma12"] = (
            df["intra_week_volatility"].rolling(window=long_, min_periods=1).mean()
        )

    return df


def save_weekly_data(df: pd.DataFrame, output_path: str) -> None:
    """Save weekly data to CSV.

    Args:
        df: DataFrame with weekly features
        output_path: Path to output CSV
    """
    df_out = df.copy()
    df_out["week_start"] = df_out["week_start"].dt.strftime("%Y-%m-%d")
    df_out["week_end"] = df_out["week_end"].dt.strftime("%Y-%m-%d")
    df_out.to_csv(output_path, index=False)


@workflow_script("03-generate-features")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    input_dir = get_historical_dir(DATA_TIER) / "daily"
    output_dir = get_historical_dir(DATA_TIER) / "weekly"
    os.makedirs(output_dir, exist_ok=True)

    l1_features = [k for k, v in FEATURES_ENABLED.items() if k.startswith("L1") and v]
    l2_features = [k for k, v in FEATURES_ENABLED.items() if k.startswith("L2") and v]

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter (targets only): {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  L1 features: {len(l1_features)}")
    print(f"  L2 features: {len(l2_features)}")
    print(f"  Lookback periods: {LOOKBACK_PERIODS}")
    print()

    # Get input files
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run 02-fetch-daily-data.py first.")
        return

    # Load fetch manifest for category info
    fetch_manifest = load_fetch_manifest(metadata_dir)

    # Get symbols to process (applies filter to targets only)
    csv_files = get_symbols_to_process(input_dir, fetch_manifest)

    # Count by category
    target_count = 0
    macro_count = 0
    for f in csv_files:
        cat = get_symbol_category(f.stem, fetch_manifest)
        if cat == "target":
            target_count += 1
        else:
            macro_count += 1

    print(f"Found {len(csv_files)} daily CSV files to process")
    print(f"  Target ETFs: {target_count}")
    print(f"  Macro symbols: {macro_count}")
    print("-" * 80)

    if not csv_files:
        logger.warning("No files to process. Exiting.")
        return

    # Process each symbol
    manifest = {
        "created_at": datetime.now().isoformat(),
        "data_tier": DATA_TIER,
        "source_dir": str(input_dir),
        "features_enabled": {"L1": l1_features, "L2": l2_features},
        "lookback_periods": LOOKBACK_PERIODS,
        "symbols": {},
        "categories": {
            "target": [],
            "volatility": [],
            "treasury": [],
            "dollar": [],
            "commodities": [],
        },
    }

    success_count = 0
    fail_count = 0
    total_weeks = 0

    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.stem
        category = get_symbol_category(symbol, fetch_manifest)
        category_indicator = f"[{category.upper()[:3]}]"
        print(f"[{i}/{len(csv_files)}] {symbol} {category_indicator}...", end=" ")

        daily_df = load_daily_data(str(csv_file))
        if daily_df is None or daily_df.empty:
            print("✗ No data")
            fail_count += 1
            continue

        weekly_df = compute_L0_base(daily_df)
        weekly_df = compute_L1_derived(weekly_df, daily_df)
        weekly_df = compute_L2_temporal(weekly_df)

        output_path = output_dir / f"{symbol}.csv"
        save_weekly_data(weekly_df, str(output_path))

        weeks_count = len(weekly_df)
        total_weeks += weeks_count
        manifest["symbols"][symbol] = {
            "weeks": weeks_count,
            "first_week": weekly_df["week_start"].min().strftime("%Y-%m-%d"),
            "last_week": weekly_df["week_start"].max().strftime("%Y-%m-%d"),
            "category": category,
        }

        # Track by category
        if category in manifest["categories"]:
            manifest["categories"][category].append(symbol)
        else:
            manifest["categories"][category] = [symbol]

        print(f"✓ {weeks_count} weeks")
        success_count += 1

    # Save manifest
    manifest_path = output_dir / "_manifest.json"
    manifest["symbols_processed"] = success_count
    manifest["total_weeks"] = total_weeks
    manifest["target_count"] = len(manifest["categories"].get("target", []))
    manifest["macro_count"] = (
        len(manifest["categories"].get("volatility", []))
        + len(manifest["categories"].get("treasury", []))
        + len(manifest["categories"].get("dollar", []))
        + len(manifest["categories"].get("commodities", []))
    )

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print_summary(
        symbols_processed=success_count,
        target_etfs=manifest["target_count"],
        macro_symbols=manifest["macro_count"],
        failed=fail_count,
        total_weeks_derived=total_weeks,
        output_directory=str(output_dir),
        manifest=str(manifest_path),
    )


if __name__ == "__main__":
    main()

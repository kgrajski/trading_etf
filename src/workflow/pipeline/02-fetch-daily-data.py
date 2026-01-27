#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch daily ETF data from Alpaca.

Fetches daily OHLCV bars for:
1. Target ETFs from filtered_etfs.json (prediction targets)
2. Macro symbol proxies from config (feature-only symbols)

Supports incremental mode: checks existing data and fetches only new bars.

Input:
  - data/metadata/filtered_etfs.json (from 01-filter-etf-universe.py)
  - MACRO_SYMBOLS from config.py

Output:
  - data/historical/{tier}/daily/{symbol}.csv
  - data/metadata/fetch_manifest.json
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.workflow.config import (
    API_DELAY_SECONDS,
    DATA_TIER,
    DATE_RANGE_END,
    DATE_RANGE_START,
    INCREMENTAL_MODE,
    LOOKBACK_BUFFER_DAYS,
    MACRO_SYMBOL_CATEGORIES,
    MACRO_SYMBOLS,
    SYMBOL_PREFIX_FILTER,
    YEARS_OF_HISTORY,
)
from src.workflow.workflow_utils import (
    get_historical_dir,
    get_metadata_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


def load_filtered_etfs(input_file: Path) -> List[Dict[str, Any]]:
    """Load filtered ETF list from script 01's output.

    Args:
        input_file: Path to filtered_etfs.json

    Returns:
        List of ETF dictionaries with symbol, name, etc.
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    return data.get("etfs", [])


def build_macro_symbol_list() -> List[Dict[str, Any]]:
    """Build list of macro symbols from config.

    Returns:
        List of macro symbol dictionaries with symbol, name, category
    """
    macros = []
    for category, symbols in MACRO_SYMBOLS.items():
        for symbol, description in symbols.items():
            macros.append(
                {
                    "symbol": symbol,
                    "name": description,
                    "category": category,
                    "is_macro": True,
                }
            )
    return macros


def get_fetch_list(
    metadata_dir: Path, apply_prefix_filter: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Get combined list of target ETFs and macro symbols.

    Args:
        metadata_dir: Path to metadata directory
        apply_prefix_filter: Whether to apply SYMBOL_PREFIX_FILTER to targets

    Returns:
        Tuple of (target_etfs, macro_symbols)
    """
    # Load target ETFs
    input_file = metadata_dir / "filtered_etfs.json"
    if input_file.exists():
        targets = load_filtered_etfs(input_file)
        # Add category and is_macro flag
        for etf in targets:
            etf["category"] = "target"
            etf["is_macro"] = False
    else:
        logger.warning(f"No filtered_etfs.json found at {input_file}")
        targets = []

    # Apply prefix filter to targets only (not macro symbols)
    if apply_prefix_filter and SYMBOL_PREFIX_FILTER:
        targets = [
            etf for etf in targets if etf["symbol"].startswith(SYMBOL_PREFIX_FILTER)
        ]

    # Build macro symbol list (never filtered by prefix)
    macros = build_macro_symbol_list()

    return targets, macros


def get_alpaca_client() -> StockHistoricalDataClient:
    """Create Alpaca historical data client.

    Returns:
        Configured StockHistoricalDataClient

    Raises:
        ValueError: If API credentials are not found
    """
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. "
            "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env file."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def calculate_full_history_dates() -> Tuple[datetime, datetime]:
    """Calculate date range for full history fetch.

    Returns:
        Tuple of (start_date, end_date)
    """
    if DATE_RANGE_END:
        end_date = datetime.strptime(DATE_RANGE_END, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if DATE_RANGE_START:
        start_date = datetime.strptime(DATE_RANGE_START, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=YEARS_OF_HISTORY * 365)

    return start_date, end_date


def get_fetch_range(
    symbol: str, output_dir: Path, default_start: datetime, default_end: datetime
) -> Tuple[datetime, datetime, bool]:
    """Determine date range based on existing data (incremental mode).

    Args:
        symbol: Symbol to check
        output_dir: Directory where CSVs are stored
        default_start: Default start date for full fetch
        default_end: Default end date

    Returns:
        Tuple of (start_date, end_date, is_incremental)
    """
    csv_path = output_dir / f"{symbol}.csv"

    if INCREMENTAL_MODE and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and "date" in df.columns:
                last_date = pd.to_datetime(df["date"]).max()
                # Go back a few days to catch any corrections
                start = last_date - timedelta(days=LOOKBACK_BUFFER_DAYS)
                return start, default_end, True
        except Exception as e:
            logger.warning(f"Could not read existing data for {symbol}: {e}")

    return default_start, default_end, False


def fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.DataFrame]:
    """Fetch daily bars for a single symbol.

    Args:
        client: Alpaca client
        symbol: Symbol to fetch
        start_date: Start of date range
        end_date: End of date range

    Returns:
        DataFrame with daily bars, or None if no data
    """
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        adjustment=Adjustment.ALL,  # Adjust for splits AND dividends
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return None

        df = df.reset_index()
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

        column_order = [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
        df = df[[col for col in column_order if col in df.columns]]

        return df

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None


def merge_and_save(
    new_df: pd.DataFrame, output_path: Path, is_incremental: bool
) -> int:
    """Merge new data with existing (if incremental) and save.

    Args:
        new_df: New data to save
        output_path: Path to output CSV
        is_incremental: Whether this is an incremental update

    Returns:
        Number of rows in final dataset
    """
    if is_incremental and output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            # Keep latest data for each date (in case of corrections)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Could not merge with existing data: {e}")
            combined = new_df.sort_values("date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return len(combined)


def save_fetch_manifest(
    metadata_dir: Path,
    target_symbols: List[str],
    macro_symbols: List[str],
    categories: Dict[str, str],
    stats: Dict[str, Any],
) -> Path:
    """Save fetch manifest with metadata about what was fetched.

    Args:
        metadata_dir: Path to metadata directory
        target_symbols: List of target symbol names
        macro_symbols: List of macro symbol names
        categories: Map of symbol to category
        stats: Fetch statistics

    Returns:
        Path to saved manifest
    """
    manifest = {
        "last_updated": datetime.now().isoformat(),
        "data_tier": DATA_TIER,
        "incremental_mode": INCREMENTAL_MODE,
        "target_symbols": sorted(target_symbols),
        "macro_symbols": sorted(macro_symbols),
        "categories": categories,
        "stats": stats,
    }

    manifest_path = metadata_dir / "fetch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


@workflow_script("02-fetch-daily-data")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    output_dir = get_historical_dir(DATA_TIER) / "daily"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate default date range
    default_start, default_end = calculate_full_history_dates()

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Incremental mode: {INCREMENTAL_MODE}")
    if INCREMENTAL_MODE:
        print(f"  Lookback buffer: {LOOKBACK_BUFFER_DAYS} days")
    print(
        f"  Default date range: {default_start.strftime('%Y-%m-%d')} "
        f"to {default_end.strftime('%Y-%m-%d')}"
    )
    print(f"  Symbol filter (targets only): {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Output: {output_dir}")
    print()

    # Get fetch lists
    print("Building fetch list...")
    targets, macros = get_fetch_list(metadata_dir)

    print(f"  Target ETFs: {len(targets)}")
    if targets:
        target_symbols = [t["symbol"] for t in targets]
        print(f"    Symbols: {', '.join(target_symbols[:10])}"
              f"{'...' if len(target_symbols) > 10 else ''}")

    print(f"  Macro symbols: {len(macros)}")
    if macros:
        macro_symbols = [m["symbol"] for m in macros]
        print(f"    Symbols: {', '.join(macro_symbols)}")
    print()

    # Combine for fetching
    all_symbols = targets + macros

    if not all_symbols:
        logger.error("No symbols to fetch. Please run 00 and 01 first.")
        return

    # Initialize Alpaca client
    print("Initializing Alpaca client...")
    try:
        client = get_alpaca_client()
        print("  Client initialized successfully")
    except ValueError as e:
        logger.error(str(e))
        return

    print()
    print(f"Fetching daily bars for {len(all_symbols)} symbols...")
    print("-" * 80)

    # Tracking
    success_count = 0
    fail_count = 0
    incremental_count = 0
    total_bars = 0
    categories: Dict[str, str] = {}

    for i, symbol_info in enumerate(all_symbols, 1):
        symbol = symbol_info["symbol"]
        category = symbol_info.get("category", "unknown")
        is_macro = symbol_info.get("is_macro", False)
        categories[symbol] = category

        # Determine fetch range
        start_date, end_date, is_incremental = get_fetch_range(
            symbol, output_dir, default_start, default_end
        )

        mode_indicator = "[INC]" if is_incremental else "[FULL]"
        category_indicator = f"[{category.upper()[:3]}]"
        print(
            f"[{i}/{len(all_symbols)}] {symbol} {category_indicator} {mode_indicator}...",
            end=" ",
        )

        df = fetch_daily_bars(client, symbol, start_date, end_date)

        if df is not None and not df.empty:
            output_path = output_dir / f"{symbol}.csv"
            bar_count = merge_and_save(df, output_path, is_incremental)
            new_bars = len(df)
            total_bars += new_bars
            success_count += 1
            if is_incremental:
                incremental_count += 1
            print(f"✓ +{new_bars} bars (total: {bar_count})")
        else:
            fail_count += 1
            print("✗ No data")

        time.sleep(API_DELAY_SECONDS)

    # Save manifest
    print()
    print("Saving fetch manifest...")
    manifest_path = save_fetch_manifest(
        metadata_dir,
        target_symbols=[t["symbol"] for t in targets],
        macro_symbols=[m["symbol"] for m in macros],
        categories=categories,
        stats={
            "total_symbols": len(all_symbols),
            "successful": success_count,
            "failed": fail_count,
            "incremental_updates": incremental_count,
            "total_new_bars": total_bars,
        },
    )
    print(f"  Manifest: {manifest_path}")

    # Summary
    print_summary(
        total_symbols=len(all_symbols),
        target_etfs=len(targets),
        macro_symbols=len(macros),
        successful=success_count,
        failed=fail_count,
        incremental_updates=incremental_count,
        total_new_bars=total_bars,
        data_tier=DATA_TIER,
        output_directory=str(output_dir),
    )


if __name__ == "__main__":
    main()

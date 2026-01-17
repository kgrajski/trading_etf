#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch daily ETF data from Alpaca.

Fetches daily OHLCV bars for all filtered ETFs and stores them
in a tier-aware directory structure.

Input: data/metadata/filtered_etfs.json (from 01-filter-etf-universe.py)
Output: data/historical/{tier}/daily/{symbol}.csv
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.workflow.config import (
    API_DELAY_SECONDS,
    DATA_TIER,
    DATE_RANGE_END,
    DATE_RANGE_START,
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


def load_filtered_etfs(input_file: str) -> List[Dict]:
    """Load filtered ETF list from script 01's output."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data.get("etfs", [])


def get_alpaca_client() -> StockHistoricalDataClient:
    """Create Alpaca historical data client."""
    load_dotenv()
    api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. "
            "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env file."
        )

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def fetch_daily_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.DataFrame]:
    """Fetch daily bars for a single symbol."""
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
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


def save_daily_data(df: pd.DataFrame, output_dir: str, symbol: str) -> str:
    """Save daily bar data to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}.csv")
    df.to_csv(output_file, index=False)
    return output_file


@workflow_script("02-fetch-daily-data")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    input_file = metadata_dir / "filtered_etfs.json"
    output_dir = get_historical_dir(DATA_TIER) / "daily"

    # Calculate date range
    if DATE_RANGE_END:
        end_date = datetime.strptime(DATE_RANGE_END, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if DATE_RANGE_START:
        start_date = datetime.strptime(DATE_RANGE_START, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=YEARS_OF_HISTORY * 365)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(
        f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    print(f"  Symbol filter: {SYMBOL_PREFIX_FILTER or 'None (all symbols)'}")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_dir}")
    print()

    # Load symbols
    print("Loading filtered ETF universe...")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run 00 and 01 first.")
        return

    etfs = load_filtered_etfs(str(input_file))
    print(f"  Loaded {len(etfs)} ETFs from filter")

    if SYMBOL_PREFIX_FILTER:
        etfs = [etf for etf in etfs if etf["symbol"].startswith(SYMBOL_PREFIX_FILTER)]
        print(f"  After prefix filter '{SYMBOL_PREFIX_FILTER}': {len(etfs)} ETFs")

    symbols = [etf["symbol"] for etf in etfs]
    print()

    if not symbols:
        logger.warning("No symbols to fetch. Exiting.")
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
    print(f"Fetching daily bars for {len(symbols)} symbols...")
    print("-" * 80)

    success_count = 0
    fail_count = 0
    total_bars = 0

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end=" ")

        df = fetch_daily_bars(client, symbol, start_date, end_date)

        if df is not None and not df.empty:
            save_daily_data(df, str(output_dir), symbol)
            bar_count = len(df)
            total_bars += bar_count
            success_count += 1
            print(f"✓ {bar_count} bars saved")
        else:
            fail_count += 1
            print("✗ No data")

        time.sleep(API_DELAY_SECONDS)

    # Summary
    print_summary(
        symbols_processed=len(symbols),
        successful=success_count,
        failed=fail_count,
        total_bars_fetched=total_bars,
        data_tier=DATA_TIER,
        output_directory=str(output_dir),
    )


if __name__ == "__main__":
    main()

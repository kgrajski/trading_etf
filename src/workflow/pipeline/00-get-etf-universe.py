#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Workflow script to get ETF universe.

Fetches ETF list from Alpaca API, filtered to iShares (BlackRock),
Vanguard, and State Street (SPDR) sponsors.

Output: data/metadata/symbols.json, symbols.csv, alpaca_assets.json, alpaca_assets.csv
"""

from collections import Counter
from datetime import datetime

from src.data.symbol_list_manager import SymbolListManager
from src.workflow.workflow_utils import (
    get_metadata_dir,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


def print_etf_table(etf_data: list) -> None:
    """Print ETF symbols in a formatted table.

    Args:
        etf_data: List of dictionaries with ETF information from Alpaca
    """
    if not etf_data:
        print("No ETFs found.")
        return

    # Calculate column widths
    max_symbol_len = (
        max(len(etf.get("symbol", "")) for etf in etf_data) if etf_data else 4
    )
    max_name_len = max(len(etf.get("name", "")) for etf in etf_data) if etf_data else 7
    max_sponsor_len = (
        max(len(etf.get("sponsor", "Unknown")) for etf in etf_data) if etf_data else 7
    )
    symbol_width = max(max_symbol_len, 6)
    name_width = max(max_name_len, 12)
    sponsor_width = max(max_sponsor_len, 7)

    # Print header
    print("ETF Universe:")
    separator = "-" * (symbol_width + name_width + sponsor_width + 8)
    print(separator)
    header = (
        f"{'Symbol':<{symbol_width}}| "
        f"{'Name':<{name_width}}| "
        f"{'Sponsor':<{sponsor_width}}"
    )
    print(header)
    print(separator)

    # Sort and print rows
    sorted_etfs = sorted(etf_data, key=lambda x: x.get("symbol", ""))
    for etf in sorted_etfs:
        symbol = etf.get("symbol", "Unknown")
        name = etf.get("name", "Unknown")
        sponsor = etf.get("sponsor", "Unknown")
        row = (
            f"{symbol:<{symbol_width}}| "
            f"{name:<{name_width}}| "
            f"{sponsor:<{sponsor_width}}"
        )
        print(row)

    print(separator)
    print()


@workflow_script("00-get-etf-universe")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    manager = SymbolListManager(str(metadata_dir))

    print("Configuration:")
    print(f"  Output: {metadata_dir}")
    print("  Sponsors: iShares (BlackRock), Vanguard, State Street (SPDR)")
    print("  Tradable only: True")
    print()

    # Fetch ETFs from Alpaca
    print("Fetching ETF universe from Alpaca API...")
    print()

    try:
        etf_data = manager.fetch_from_alpaca(filter_by_sponsor=True, tradable_only=True)

        if not etf_data:
            logger.warning(
                "No ETFs found from Alpaca. Using default symbols as fallback."
            )
            symbols = manager.get_default_symbols()
            etf_data = [
                {"symbol": s, "name": f"{s} ETF", "sponsor": "Unknown"} for s in symbols
            ]
        else:
            symbols = [etf["symbol"] for etf in etf_data]

        print(f"ETF Universe: {len(etf_data)} ETFs")
        print()
        print_etf_table(etf_data)

        # Sponsor breakdown
        sponsor_counts = Counter(etf.get("sponsor", "Unknown") for etf in etf_data)
        print("Summary by Sponsor:")
        print("-" * 40)
        for sponsor, count in sponsor_counts.most_common():
            print(f"  {sponsor}: {count} ETFs")
        print()

        # Save
        manager.save_symbols(
            symbols,
            metadata={
                "source": "alpaca",
                "description": "ETFs from iShares, Vanguard, and State Street (SPDR)",
                "filter_by_sponsor": True,
                "tradable_only": True,
                "sponsor_breakdown": dict(sponsor_counts),
                "fetched_at": datetime.now().isoformat(),
            },
        )

        print("Saved to:")
        print(f"  JSON: {manager.symbols_file}")
        print(f"  CSV:  {manager.symbols_csv_file}")
        print(f"  Alpaca cache: {manager.alpaca_assets_file}")

    except Exception as e:
        logger.error(f"Error fetching from Alpaca: {e}")
        logger.info("Falling back to default symbols")
        symbols = manager.get_default_symbols()
        manager.save_symbols(
            symbols,
            metadata={
                "source": "default_fallback",
                "description": "Default symbols (Alpaca fetch failed)",
                "error": str(e),
            },
        )
        print(f"Using default symbols: {len(symbols)} symbols")


if __name__ == "__main__":
    main()

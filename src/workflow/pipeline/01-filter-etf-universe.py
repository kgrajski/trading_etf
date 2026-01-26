#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Static filtering of ETF universe.

Applies static filters to the ETF universe from script 00:
- Exchange filter: Keep only main exchanges (ARCA, NYSE, NASDAQ, BATS, AMEX)
- Excludes leveraged and inverse ETFs

Input: data/metadata/alpaca_assets.json (from 00-get-etf-universe.py)
Output: data/metadata/filtered_etfs.json, filtered_etfs.csv
"""

import json
from collections import Counter
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.workflow.config import ALLOWED_EXCHANGES, LEVERAGED_INVERSE_PATTERNS
from src.workflow.workflow_utils import (
    get_metadata_dir,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


def load_etf_universe(input_file: str) -> List[Dict]:
    """Load ETF universe from 00's output."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data.get("etfs", [])


def filter_by_exchange(etfs: List[Dict]) -> List[Dict]:
    """Filter ETFs to only include those on allowed exchanges."""
    filtered = []
    excluded_exchanges = Counter()

    for etf in etfs:
        exchange = etf.get("exchange", "").replace("AssetExchange.", "")
        if exchange in ALLOWED_EXCHANGES:
            filtered.append(etf)
        else:
            excluded_exchanges[exchange] += 1

    if excluded_exchanges:
        logger.info(f"Excluded by exchange: {dict(excluded_exchanges)}")

    return filtered


def filter_leveraged_inverse(etfs: List[Dict]) -> List[Dict]:
    """Filter out leveraged and inverse ETFs."""
    filtered = []
    excluded_count = 0

    for etf in etfs:
        name_lower = etf.get("name", "").lower()
        is_excluded = any(
            pattern in name_lower for pattern in LEVERAGED_INVERSE_PATTERNS
        )

        if is_excluded:
            excluded_count += 1
            logger.debug(
                f"Excluded leveraged/inverse: {etf.get('symbol')} - {etf.get('name')}"
            )
        else:
            filtered.append(etf)

    if excluded_count:
        logger.info(f"Excluded leveraged/inverse ETFs: {excluded_count}")

    return filtered


def save_filtered_etfs(
    etfs: List[Dict], json_file: str, csv_file: str, metadata: Dict
) -> None:
    """Save filtered ETFs to JSON and CSV."""
    # Save JSON
    output_data = {"etfs": etfs, "metadata": metadata}
    with open(json_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Save CSV
    if etfs:
        df = pd.DataFrame(etfs)
        column_order = ["symbol", "name", "sponsor", "exchange", "tradable", "status"]
        df = df[[col for col in column_order if col in df.columns]]
        df = df.sort_values("symbol")
        df.to_csv(csv_file, index=False)


def print_summary(etfs: List[Dict]) -> Dict[str, int]:
    """Print summary of filtered ETFs."""
    print(f"\nFiltered ETF Universe: {len(etfs)} ETFs")
    print("-" * 40)

    # Sponsor breakdown
    sponsor_counts = Counter(etf.get("sponsor", "Unknown") for etf in etfs)
    print("\nBy Sponsor:")
    for sponsor, count in sponsor_counts.most_common():
        print(f"  {sponsor}: {count}")

    # Exchange breakdown
    exchange_counts = Counter(
        etf.get("exchange", "").replace("AssetExchange.", "") for etf in etfs
    )
    print("\nBy Exchange:")
    for exchange, count in exchange_counts.most_common():
        print(f"  {exchange}: {count}")

    return dict(sponsor_counts)


@workflow_script("01-filter-etf-universe")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    input_file = metadata_dir / "alpaca_assets.json"
    output_json = metadata_dir / "filtered_etfs.json"
    output_csv = metadata_dir / "filtered_etfs.csv"

    print("Configuration:")
    print(f"  Allowed exchanges: {sorted(ALLOWED_EXCHANGES)}")
    print("  Exclude leveraged/inverse: Yes")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_json}")
    print()

    # Load data
    print("Loading ETF universe from script 00...")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error("Please run 00-get-etf-universe.py first.")
        return

    etfs = load_etf_universe(str(input_file))
    print(f"  Loaded {len(etfs)} ETFs")

    # Apply filters
    print("\nApplying static filters...")

    print("  Filter 1: Exchange...")
    etfs = filter_by_exchange(etfs)
    print(f"    After exchange filter: {len(etfs)} ETFs")

    print("  Filter 2: Leveraged/Inverse exclusion...")
    etfs = filter_leveraged_inverse(etfs)
    print(f"    After leveraged/inverse filter: {len(etfs)} ETFs")

    # Summary
    sponsor_counts = print_summary(etfs)

    # Save
    print("\nSaving filtered ETF universe...")
    metadata = {
        "source": "01-filter-etf-universe",
        "input_file": str(input_file),
        "filters_applied": {
            "allowed_exchanges": sorted(ALLOWED_EXCHANGES),
            "exclude_leveraged_inverse": True,
        },
        "total_etfs": len(etfs),
        "sponsor_breakdown": sponsor_counts,
        "filtered_at": datetime.now().isoformat(),
    }
    save_filtered_etfs(etfs, str(output_json), str(output_csv), metadata)
    print(f"  JSON: {output_json}")
    print(f"  CSV:  {output_csv}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Symbol list management for ETF universe.

This module manages the list of ETFs to trade, including fetching from
Alpaca and applying filters.

Author: kag
Created: 2025-12-01
"""

import csv
import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest
from dotenv import load_dotenv

from src.data.etf_filter import extract_sponsor, is_etf, is_target_sponsor
from src.data.symbol_filter import SymbolFilter

# Set up logging
logger = logging.getLogger(__name__)


class SymbolListManager:
    """Manages the list of ETF symbols for trading."""

    # Default small list of major ETFs for initial testing
    DEFAULT_SYMBOLS = [
        "SPY",  # S&P 500
        "QQQ",  # Nasdaq 100
        "IWM",  # Russell 2000
        "DIA",  # Dow Jones
        "VTI",  # Total Stock Market
        "XLE",  # Energy Sector
        "XLF",  # Financial Sector
        "XLI",  # Industrial Sector
        "XLK",  # Technology Sector
        "XLV",  # Healthcare Sector
        "XLP",  # Consumer Staples
        "XLY",  # Consumer Discretionary
        "XLB",  # Materials
        "XLU",  # Utilities
        "XLV",  # Healthcare
    ]

    def __init__(self, metadata_dir: str):
        """Initialize symbol list manager.

        Args:
            metadata_dir: Directory to store symbol lists and metadata
        """
        self.metadata_dir = metadata_dir
        os.makedirs(metadata_dir, exist_ok=True)
        self.symbols_file = os.path.join(metadata_dir, "symbols.json")
        self.symbols_csv_file = os.path.join(metadata_dir, "symbols.csv")
        self.alpaca_assets_file = os.path.join(metadata_dir, "alpaca_assets.json")
        self.alpaca_assets_csv_file = os.path.join(metadata_dir, "alpaca_assets.csv")
        self._alpaca_client = None

    def get_default_symbols(self) -> List[str]:
        """Get default small list of major ETFs.

        Returns:
            List of symbol strings
        """
        return self.DEFAULT_SYMBOLS.copy()

    def save_symbols(self, symbols: List[str], metadata: Optional[Dict] = None) -> None:
        """Save symbol list to file (JSON and CSV).

        Args:
            symbols: List of symbol strings
            metadata: Optional metadata about the symbol list
        """
        data = {"symbols": symbols, "metadata": metadata or {}}

        # Save JSON
        with open(self.symbols_file, "w") as f:
            json.dump(data, f, indent=2)

        # Save CSV
        with open(self.symbols_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            # Write metadata as header comments
            if metadata:
                writer.writerow(["# Metadata"])
                for key, value in metadata.items():
                    if isinstance(value, dict):
                        writer.writerow([f"# {key}", json.dumps(value)])
                    else:
                        writer.writerow([f"# {key}", str(value)])
                writer.writerow([])  # Empty row
            # Write header
            writer.writerow(["symbol"])
            # Write symbols
            for symbol in sorted(symbols):
                writer.writerow([symbol])

    def load_symbols(self) -> List[str]:
        """Load symbol list from file.

        Returns:
            List of symbol strings, or default list if file doesn't exist
        """
        if os.path.exists(self.symbols_file):
            with open(self.symbols_file, "r") as f:
                data = json.load(f)
                return data.get("symbols", self.DEFAULT_SYMBOLS)

        return self.DEFAULT_SYMBOLS

    def _get_alpaca_client(self) -> TradingClient:
        """Get or create Alpaca TradingClient.

        Returns:
            TradingClient instance
        """
        if self._alpaca_client is None:
            load_dotenv()
            api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv(
                "ALPACA_SECRET_KEY"
            )

            if not api_key or not secret_key:
                raise ValueError(
                    "Alpaca API credentials not found. "
                    "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env file"
                )

            self._alpaca_client = TradingClient(
                api_key=api_key, secret_key=secret_key, paper=True
            )

        return self._alpaca_client

    def fetch_from_alpaca(
        self, filter_by_sponsor: bool = True, tradable_only: bool = True
    ) -> List[Dict]:
        """Fetch ETF list from Alpaca API.

        Fetches all US equity assets, filters to ETFs, and optionally filters
        to target sponsors (iShares, Vanguard, State Street).

        Args:
            filter_by_sponsor: If True, only return ETFs from target sponsors
            tradable_only: If True, only return tradable assets

        Returns:
            List of dictionaries with ETF information including:
            - symbol: ETF ticker
            - name: Full name from Alpaca
            - sponsor: Extracted sponsor name
            - tradable: Whether asset is tradable
            - exchange: Exchange where asset trades
        """
        client = self._get_alpaca_client()

        # Build request
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE
        )

        # Fetch all assets
        logger.info("Fetching assets from Alpaca API...")
        assets = client.get_all_assets(request)

        logger.info(f"Fetched {len(assets)} assets from Alpaca")

        # Convert to list of dicts and filter
        etf_data = []
        sponsor_extraction_failures = []

        for asset in assets:
            asset_dict = asset.model_dump()

            # Filter by tradable if requested
            if tradable_only and not asset_dict.get("tradable", False):
                continue

            asset_name = asset_dict.get("name", "")
            symbol = asset_dict.get("symbol", "")

            # Check if it's an ETF
            if not is_etf(asset_name):
                continue

            # Extract sponsor
            sponsor = extract_sponsor(asset_name)

            # Log extraction failures for target sponsors
            if filter_by_sponsor and not sponsor:
                # Check if name suggests it might be from a target sponsor
                name_lower = asset_name.lower()
                if any(keyword in name_lower for keyword in ["etf", "trust", "fund"]):
                    sponsor_extraction_failures.append(
                        {"symbol": symbol, "name": asset_name}
                    )

            # Filter by sponsor if requested
            if filter_by_sponsor and not is_target_sponsor(sponsor):
                continue

            etf_data.append(
                {
                    "symbol": symbol,
                    "name": asset_name,
                    "sponsor": sponsor or "Unknown",
                    "tradable": asset_dict.get("tradable", False),
                    "exchange": str(asset_dict.get("exchange", "")),
                    "status": str(asset_dict.get("status", "")),
                }
            )

        # Log extraction failures
        if sponsor_extraction_failures:
            logger.warning(
                f"Found {len(sponsor_extraction_failures)} ETFs that might be "
                "from target sponsors but failed sponsor extraction:"
            )
            for item in sponsor_extraction_failures[:10]:  # Log first 10
                logger.warning(f"  {item['symbol']}: {item['name']}")
            if len(sponsor_extraction_failures) > 10:
                logger.warning(
                    f"  ... and {len(sponsor_extraction_failures) - 10} more"
                )

        # Save to cache (JSON)
        cache_data = {
            "etfs": etf_data,
            "extraction_failures": sponsor_extraction_failures,
            "total_assets_fetched": len(assets),
        }
        with open(self.alpaca_assets_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Save to CSV for easy inspection
        if etf_data:
            df = pd.DataFrame(etf_data)
            # Reorder columns for readability
            column_order = [
                "symbol",
                "name",
                "sponsor",
                "exchange",
                "tradable",
                "status",
            ]
            df = df[[col for col in column_order if col in df.columns]]
            df = df.sort_values("symbol")
            df.to_csv(self.alpaca_assets_csv_file, index=False)

        logger.info(
            f"Found {len(etf_data)} ETFs"
            + (f" from target sponsors" if filter_by_sponsor else "")
        )

        return etf_data

    def apply_filters(
        self, symbols_data: List[Dict], filter_config: Optional[Dict] = None
    ) -> List[str]:
        """Apply filters to symbol data and return filtered symbol list.

        Args:
            symbols_data: List of dictionaries with symbol information
            filter_config: Optional filter configuration dict

        Returns:
            List of filtered symbol strings
        """
        if filter_config is None:
            filter_config = {
                "min_avg_volume": 500000,
                "exclude_leveraged": True,
                "exclude_inverse": True,
                "min_age_years": 1.0,
            }

        filter_obj = SymbolFilter()

        if filter_config.get("min_avg_volume"):
            filter_obj.add_min_volume_filter(filter_config["min_avg_volume"])

        if filter_config.get("exclude_leveraged", True):
            filter_obj.add_exclude_leveraged_filter()

        if filter_config.get("exclude_inverse", True):
            filter_obj.add_exclude_inverse_filter()

        if filter_config.get("min_age_years"):
            filter_obj.add_min_age_filter(filter_config["min_age_years"])

        filtered_data = filter_obj.apply(symbols_data)
        return [s["symbol"] for s in filtered_data]

# -*- coding: utf-8 -*-
"""Metadata management for market data collection.

This module provides the MetadataManager class for tracking which dates
have been fetched for each symbol, along with other metadata about the
data collection process.

Author: kag
Created: 2025-12-01
"""

import json
import os
from datetime import datetime
from typing import Dict, List


class MetadataManager:
    """Manages metadata for market data collection.

    Handles loading, saving, and updating metadata JSON files that track
    which dates have been fetched for a given symbol.
    """

    def __init__(self, symbol_dir: str):
        """Initialize MetadataManager for a symbol directory.

        Args:
            symbol_dir: Path to the symbol's data directory
        """
        self.symbol_dir = symbol_dir
        self.metadata_path = os.path.join(symbol_dir, "metadata.json")
        self.metadata = self._load()

    def _load(self) -> Dict:
        """Load metadata from JSON file.

        Returns:
            Dictionary with metadata, or empty dict if file doesn't exist
        """
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        """Save metadata to JSON file."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_fetched_weeks(self) -> List[str]:
        """Get list of week start dates (Mondays) that have been fetched.

        Returns:
            List of week start date strings in YYYY-MM-DD format
        """
        return self.metadata.get("weeks_fetched", [])

    def add_week(self, week_start: str):
        """Add a week to the fetched weeks list.

        Args:
            week_start: Week start date (Monday) in YYYY-MM-DD format
        """
        weeks = self.get_fetched_weeks()
        if week_start not in weeks:
            weeks.append(week_start)
            self.metadata["weeks_fetched"] = sorted(weeks)

    def add_weeks(self, weeks: List[str]):
        """Add multiple weeks to the fetched weeks list.

        Args:
            weeks: List of week start date strings
        """
        existing = set(self.get_fetched_weeks())
        existing.update(weeks)
        self.metadata["weeks_fetched"] = sorted(list(existing))

    def update_collection_info(
        self, symbol: str, source_name: str, time_unit: str = "week"
    ):
        """Update metadata with collection information.

        Args:
            symbol: Stock ticker symbol
            source_name: Name of data source (e.g., 'alpaca')
            time_unit: Time unit for data ('week')
        """
        self.metadata["symbol"] = symbol
        self.metadata["data_source"] = source_name
        self.metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata["time_unit"] = time_unit

    def get_summary(self) -> Dict:
        """Get summary statistics about the data collection.

        Returns:
            Dictionary with summary information
        """
        weeks = self.get_fetched_weeks()

        summary = {
            "total_weeks": len(weeks),
            "earliest_week": min(weeks) if weeks else None,
            "latest_week": max(weeks) if weeks else None,
            "data_source": self.metadata.get("data_source", "unknown"),
            "last_updated": self.metadata.get("last_updated", "never"),
        }

        return summary

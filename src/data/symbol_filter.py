# -*- coding: utf-8 -*-
"""Symbol filtering for ETF selection.

This module provides filters for selecting ETFs based on various criteria
such as volume, age, leverage, etc.

Author: kag
Created: 2025-12-01
"""

from typing import Dict, List, Optional

import pandas as pd


class SymbolFilter:
    """Filter symbols based on various criteria."""

    def __init__(self):
        """Initialize filter."""
        self.filters = []

    def add_min_volume_filter(self, min_avg_volume: float):
        """Add filter for minimum average weekly volume.

        Args:
            min_avg_volume: Minimum average weekly volume in shares
        """

        def filter_func(symbol_data: Dict) -> bool:
            avg_volume = symbol_data.get("avg_weekly_volume", 0)
            return avg_volume >= min_avg_volume

        self.filters.append(
            ("min_volume", filter_func, {"min_avg_volume": min_avg_volume})
        )
        return self

    def add_exclude_leveraged_filter(self):
        """Exclude leveraged ETFs (2x, 3x, etc.)."""

        def filter_func(symbol_data: Dict) -> bool:
            symbol = symbol_data.get("symbol", "").upper()
            name = symbol_data.get("name", "").upper()
            # Check for leverage indicators
            leveraged_keywords = ["2X", "3X", "LEVERAGED", "ULTRA"]
            return not any(kw in symbol or kw in name for kw in leveraged_keywords)

        self.filters.append(("exclude_leveraged", filter_func, {}))
        return self

    def add_exclude_inverse_filter(self):
        """Exclude inverse ETFs."""

        def filter_func(symbol_data: Dict) -> bool:
            symbol = symbol_data.get("symbol", "").upper()
            name = symbol_data.get("name", "").upper()
            # Check for inverse indicators
            inverse_keywords = ["INVERSE", "SHORT", "-1X"]
            return not any(kw in symbol or kw in name for kw in inverse_keywords)

        self.filters.append(("exclude_inverse", filter_func, {}))
        return self

    def add_min_age_filter(self, min_years: float):
        """Add filter for minimum age (years since listing).

        Args:
            min_years: Minimum years since listing
        """

        def filter_func(symbol_data: Dict) -> bool:
            age_years = symbol_data.get("age_years", 0)
            return age_years >= min_years

        self.filters.append(("min_age", filter_func, {"min_years": min_years}))
        return self

    def add_custom_filter(self, name: str, filter_func):
        """Add a custom filter function.

        Args:
            name: Name of the filter
            filter_func: Function that takes symbol_data dict and returns bool
        """
        self.filters.append((name, filter_func, {}))
        return self

    def apply(self, symbols_data: List[Dict]) -> List[Dict]:
        """Apply all filters to a list of symbol data.

        Args:
            symbols_data: List of dictionaries with symbol information

        Returns:
            Filtered list of symbol data
        """
        filtered = []

        for symbol_data in symbols_data:
            passes = True
            for filter_name, filter_func, _ in self.filters:
                if not filter_func(symbol_data):
                    passes = False
                    break

            if passes:
                filtered.append(symbol_data)

        return filtered

    def get_filter_summary(self) -> Dict:
        """Get summary of active filters.

        Returns:
            Dictionary describing active filters
        """
        summary = {
            "num_filters": len(self.filters),
            "filters": [name for name, _, _ in self.filters],
        }
        return summary

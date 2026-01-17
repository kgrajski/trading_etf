# -*- coding: utf-8 -*-
"""Abstract base class for market data sources.

This module defines the interface that all market data sources must implement,
allowing easy swapping between data providers (yfinance, alpaca, etc.).

Author: kag
Created: 2025-12-01
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class MarketDataSource(ABC):
    """Abstract base class for market data sources.

    All data source implementations must inherit from this class and implement
    the fetch methods to retrieve raw market data.
    """

    @abstractmethod
    def fetch_day(self, symbol: str, date: str) -> Optional[pd.DataFrame]:
        """Fetch raw intraday market data for a single trading day.

        Args:
            symbol: Stock ticker symbol (e.g., 'SPY')
            date: Date string in YYYY-MM-DD format

        Returns:
            DataFrame with raw data as provided by the source, or None if no data
        """
        pass

    @abstractmethod
    def fetch_weekly(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch weekly aggregated market data.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with weekly bars (OHLCV), or None if no data
        """
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return the name of this data source.

        Returns:
            String identifier for the data source (e.g., 'alpaca')
        """
        pass

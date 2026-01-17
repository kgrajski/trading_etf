# -*- coding: utf-8 -*-
"""Factory for creating market data source instances.

This module provides a factory pattern for instantiating data source objects,
making it easy to swap between different data providers.

Author: kag
Created: 2025-12-01
"""

from src.data.alpaca_source import AlpacaSource
from src.data.market_data_source import MarketDataSource


class DataSourceFactory:
    """Factory for creating market data source instances.

    Usage:
        source = DataSourceFactory.create('alpaca')
        data = source.fetch_weekly('SPY', '2024-01-01', '2024-12-31')
    """

    @staticmethod
    def create(source_name: str) -> MarketDataSource:
        """Create a data source instance.

        Args:
            source_name: Name of the data source ('alpaca', etc.)

        Returns:
            MarketDataSource instance

        Raises:
            ValueError: If source_name is not recognized
        """
        source_name = source_name.lower()

        if source_name == "alpaca":
            return AlpacaSource()
        else:
            raise ValueError(
                f"Unknown data source: {source_name}. " f"Available sources: alpaca"
            )

    @staticmethod
    def list_available_sources() -> list:
        """Return list of available data sources.

        Returns:
            List of source name strings
        """
        return ["alpaca"]

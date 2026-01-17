# -*- coding: utf-8 -*-
"""Weekly feature engineering for ETF trading.

This module handles creating derived features from weekly market data,
including momentum, volume, and relative strength indicators.

Author: kag
Created: 2025-12-01
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class WeeklyFeatureEngineer:
    """Feature engineering for weekly ETF data.

    Transforms weekly OHLCV data into analysis-ready format with
    derived features for trading strategy development.
    """

    def __init__(self):
        """Initialize feature engineer."""
        pass

    def engineer_features(
        self, weekly_data: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Engineer features for weekly data.

        Args:
            weekly_data: DataFrame with weekly bars (week_start, open, high, low, close, volume)
            benchmark_data: Optional benchmark data (e.g., SPY) for relative strength

        Returns:
            DataFrame with added features
        """
        df = weekly_data.copy()

        # Sort by week_start
        df = df.sort_values("week_start").reset_index(drop=True)

        # Price-based features
        df = self._add_price_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Momentum features
        df = self._add_momentum_features(df)

        # Relative strength (if benchmark provided)
        if benchmark_data is not None:
            df = self._add_relative_strength(df, benchmark_data)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features.

        Args:
            df: DataFrame with weekly bars

        Returns:
            DataFrame with added features
        """
        # Weekly return
        df["weekly_return"] = df["close"].pct_change() * 100

        # High-Low range
        df["hl_range"] = ((df["high"] - df["low"]) / df["close"]) * 100

        # Close position within range
        df["close_position"] = (
            (df["close"] - df["low"]) / (df["high"] - df["low"])
        ) * 100
        df["close_position"] = df["close_position"].fillna(50.0)  # Default to middle

        # Price change from open
        df["change_from_open"] = ((df["close"] - df["open"]) / df["open"]) * 100

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.

        Args:
            df: DataFrame with weekly bars

        Returns:
            DataFrame with added features
        """
        # Volume change
        df["volume_change"] = df["volume"].pct_change() * 100

        # Volume moving average (4-week)
        df["volume_ma4"] = df["volume"].rolling(window=4, min_periods=1).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma4"]

        # Volume trend (slope over 4 weeks)
        df["volume_trend"] = (
            df["volume"]
            .rolling(window=4, min_periods=1)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        )

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features.

        Args:
            df: DataFrame with weekly bars

        Returns:
            DataFrame with added features
        """
        # N-week returns (momentum)
        for n in [2, 4, 8, 12]:
            df[f"return_{n}w"] = df["close"].pct_change(n) * 100

        # Price momentum (rate of change)
        df["momentum_4w"] = df["close"].pct_change(4) * 100
        df["momentum_8w"] = df["close"].pct_change(8) * 100

        # Moving averages
        df["ma4"] = df["close"].rolling(window=4, min_periods=1).mean()
        df["ma8"] = df["close"].rolling(window=8, min_periods=1).mean()
        df["ma12"] = df["close"].rolling(window=12, min_periods=1).mean()

        # Price vs moving averages
        df["price_vs_ma4"] = ((df["close"] - df["ma4"]) / df["ma4"]) * 100
        df["price_vs_ma8"] = ((df["close"] - df["ma8"]) / df["ma8"]) * 100

        # Volatility (rolling std of returns)
        df["volatility_4w"] = df["weekly_return"].rolling(window=4, min_periods=1).std()
        df["volatility_8w"] = df["weekly_return"].rolling(window=8, min_periods=1).std()

        return df

    def _add_relative_strength(
        self, df: pd.DataFrame, benchmark_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add relative strength vs benchmark.

        Args:
            df: DataFrame with weekly bars for symbol
            benchmark_data: DataFrame with weekly bars for benchmark (e.g., SPY)

        Returns:
            DataFrame with added relative strength features
        """
        # Merge on week_start
        benchmark_returns = benchmark_data[["week_start", "weekly_return"]].copy()
        benchmark_returns = benchmark_returns.rename(
            columns={"weekly_return": "benchmark_return"}
        )

        df = df.merge(benchmark_returns, on="week_start", how="left")

        # Relative return
        df["relative_return"] = df["weekly_return"] - df["benchmark_return"]

        # Relative strength (cumulative)
        df["relative_strength"] = (1 + df["relative_return"] / 100).cumprod() - 1
        df["relative_strength"] = df["relative_strength"] * 100

        return df

    @staticmethod
    def get_required_columns() -> list:
        """Get list of required input columns.

        Returns:
            List of column names
        """
        return ["week_start", "open", "high", "low", "close", "volume"]

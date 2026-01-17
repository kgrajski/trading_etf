"""
Alpaca data source implementation for ETF trading.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.data.market_data_source import MarketDataSource


class AlpacaSource(MarketDataSource):
    """Fetch market data from Alpaca Markets API."""

    def __init__(self):
        """Initialize Alpaca client with API credentials from environment."""
        load_dotenv()

        # Try both naming conventions
        api_key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials not found. "
                "Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env file"
            )

        self.client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def fetch_day(self, symbol: str, date: str, interval: str = "1m") -> pd.DataFrame:
        """Fetch raw data for one day from Alpaca, return as-is.

        Args:
            symbol: Stock symbol (e.g., 'SPY')
            date: Date string in YYYY-MM-DD format
            interval: Data interval (e.g., '1Min', '5Min', '30Min', '1Hour')

        Returns:
            DataFrame with raw data, or None if no data.
        """
        # Convert interval format from '1m' to '1Min' for Alpaca
        interval_map = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, "Min"),
            "15m": TimeFrame(15, "Min"),
            "30m": TimeFrame(30, "Min"),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
        }

        if interval not in interval_map:
            raise ValueError(
                f"Unsupported interval: {interval}. Use: {list(interval_map.keys())}"
            )

        alpaca_timeframe = interval_map[interval]

        # Parse date and create datetime range
        start_dt = datetime.strptime(date, "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=1)

        try:
            from alpaca.data.enums import DataFeed

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=alpaca_timeframe,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.SIP,
                adjustment=Adjustment.SPLIT,
            )

            bars = self.client.get_stock_bars(request_params)
            df = bars.df

            if df.empty:
                return None

            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]

            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "datetime"})

            return df

        except Exception as e:
            print(f"  Error fetching {symbol} for {date}: {e}")
            return None

    def fetch_weekly(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch weekly aggregated bars from Alpaca.

        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with weekly bars (week_start, week_end, open, high, low, close, volume)
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        try:
            from alpaca.data.enums import DataFeed

            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Week,
                start=start_dt,
                end=end_dt,
                feed=DataFeed.SIP,
                adjustment=Adjustment.SPLIT,
            )

            bars = self.client.get_stock_bars(request_params)
            df = bars.df

            if df.empty:
                return None

            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]

            if "timestamp" in df.columns:
                df = df.rename(columns={"timestamp": "datetime"})

            # Add week_start and week_end columns
            df["week_start"] = df["datetime"].dt.to_period("W").dt.start_time
            df["week_end"] = df["datetime"].dt.to_period("W").dt.end_time

            return df

        except Exception as e:
            print(f"  Error fetching weekly data for {symbol}: {e}")
            return None

    def get_source_name(self) -> str:
        """Return the source name."""
        return "alpaca"

# -*- coding: utf-8 -*-
"""Portfolio backtester for weekly ETF trading.

This module provides backtesting functionality for portfolio-level strategies
with multiple positions and weekly rebalancing.

Author: kag
Created: 2025-12-01
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal


class Position:
    """Represents a position in the portfolio."""

    def __init__(
        self,
        symbol: str,
        entry_week: str,
        entry_price: float,
        shares: float,
        entry_signal: Optional[Signal] = None,
    ):
        """Initialize position.

        Args:
            symbol: ETF symbol
            entry_week: Week start date when entered
            entry_price: Entry price
            shares: Number of shares
            entry_signal: Optional entry signal
        """
        self.symbol = symbol
        self.entry_week = entry_week
        self.entry_price = entry_price
        self.shares = shares
        self.entry_signal = entry_signal
        self.exit_week: Optional[str] = None
        self.exit_price: Optional[float] = None
        self.exit_signal: Optional[Signal] = None

    def exit(self, exit_week: str, exit_price: float, exit_signal: Signal):
        """Exit the position.

        Args:
            exit_week: Week start date when exited
            exit_price: Exit price
            exit_signal: Exit signal
        """
        self.exit_week = exit_week
        self.exit_price = exit_price
        self.exit_signal = exit_signal

    def get_return_pct(self) -> float:
        """Calculate return percentage.

        Returns:
            Return percentage
        """
        if self.exit_price is None:
            return 0.0
        return ((self.exit_price - self.entry_price) / self.entry_price) * 100

    def get_pnl(self) -> float:
        """Calculate profit/loss in dollars.

        Returns:
            P&L in dollars
        """
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares

    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            "symbol": self.symbol,
            "entry_week": self.entry_week,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "exit_week": self.exit_week,
            "exit_price": self.exit_price,
            "return_pct": self.get_return_pct(),
            "pnl": self.get_pnl(),
            "entry_signal": self.entry_signal.to_dict() if self.entry_signal else None,
            "exit_signal": self.exit_signal.to_dict() if self.exit_signal else None,
        }


class PortfolioBacktester:
    """Backtester for portfolio-level weekly trading strategies."""

    def __init__(self, initial_capital: float = 100000.0):
        """Initialize portfolio backtester.

        Args:
            initial_capital: Starting capital in dollars
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.closed_positions: List[Position] = []
        self.weekly_records: List[Dict] = []

    def run_backtest(
        self,
        weekly_data: Dict[str, pd.DataFrame],
        strategy: BaseStrategy,
        params: Dict,
        weeks: List[str],
    ) -> pd.DataFrame:
        """Run backtest over a list of weeks.

        Args:
            weekly_data: Dictionary mapping symbol to DataFrame with weekly features
            weeks: List of week start dates (Mondays) in chronological order
            strategy: Strategy instance
            params: Strategy parameters

        Returns:
            DataFrame with weekly portfolio metrics
        """
        for week_start in weeks:
            # Get current positions list
            current_symbols = list(self.positions.keys())

            # Generate signals
            signals = strategy.generate_signals(
                weekly_data=weekly_data,
                week_start=week_start,
                current_positions=current_symbols,
                params=params,
            )

            # Process exit signals first
            for signal in signals:
                if signal.action == Signal.EXIT and signal.symbol in self.positions:
                    self._exit_position(signal, weekly_data, week_start)

            # Process enter signals
            for signal in signals:
                if signal.action == Signal.ENTER:
                    self._enter_position(signal, weekly_data, week_start, params)

            # Update portfolio value and record metrics
            self._record_week(week_start, weekly_data)

        # Close any remaining positions at end
        final_week = weeks[-1] if weeks else None
        if final_week:
            for symbol in list(self.positions.keys()):
                signal = Signal(
                    symbol, Signal.EXIT, final_week, 1.0, {"reason": "end_of_backtest"}
                )
                self._exit_position(signal, weekly_data, final_week)
            self._record_week(final_week, weekly_data)

        return pd.DataFrame(self.weekly_records)

    def _enter_position(
        self,
        signal: Signal,
        weekly_data: Dict[str, pd.DataFrame],
        week_start: str,
        params: Dict,
    ):
        """Enter a new position.

        Args:
            signal: Enter signal
            weekly_data: Weekly data for all symbols
            week_start: Week start date
            params: Strategy parameters (may include position sizing)
        """
        symbol = signal.symbol

        if symbol not in weekly_data:
            return

        df = weekly_data[symbol]
        week_row = df[df["week_start"] == week_start]

        if week_row.empty:
            return

        # Get entry price (use Monday open or previous Friday close)
        entry_price = week_row.iloc[0].get("open", week_row.iloc[0]["close"])

        # Position sizing: equal weight across max_positions
        max_positions = params.get("max_positions", 5)
        position_size = self.capital / max_positions

        # Calculate shares
        shares = position_size / entry_price

        # Create position
        position = Position(
            symbol=symbol,
            entry_week=week_start,
            entry_price=entry_price,
            shares=shares,
            entry_signal=signal,
        )

        self.positions[symbol] = position
        self.capital -= position_size  # Reserve capital

    def _exit_position(
        self, signal: Signal, weekly_data: Dict[str, pd.DataFrame], week_start: str
    ):
        """Exit an existing position.

        Args:
            signal: Exit signal
            weekly_data: Weekly data for all symbols
            week_start: Week start date
        """
        symbol = signal.symbol

        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        if symbol not in weekly_data:
            # Use entry price if no data
            exit_price = position.entry_price
        else:
            df = weekly_data[symbol]
            week_row = df[df["week_start"] == week_start]
            if week_row.empty:
                exit_price = position.entry_price
            else:
                # Use Friday close as exit price
                exit_price = week_row.iloc[0]["close"]

        position.exit(week_start, exit_price, signal)

        # Return capital + P&L
        pnl = position.get_pnl()
        position_value = exit_price * position.shares
        self.capital += position_value

        # Move to closed positions
        del self.positions[symbol]
        self.closed_positions.append(position)

    def _record_week(self, week_start: str, weekly_data: Dict[str, pd.DataFrame]):
        """Record weekly portfolio metrics.

        Args:
            week_start: Week start date
            weekly_data: Weekly data for all symbols
        """
        # Calculate current portfolio value
        positions_value = 0.0
        for position in self.positions.values():
            symbol = position.symbol
            if symbol in weekly_data:
                df = weekly_data[symbol]
                week_row = df[df["week_start"] == week_start]
                if not week_row.empty:
                    current_price = week_row.iloc[0]["close"]
                    positions_value += current_price * position.shares
                else:
                    # Use entry price if no data
                    positions_value += position.entry_price * position.shares
            else:
                positions_value += position.entry_price * position.shares

        total_value = self.capital + positions_value

        # Calculate metrics
        total_return = (
            (total_value - self.initial_capital) / self.initial_capital
        ) * 100

        # Track drawdown
        if not self.weekly_records:
            peak_value = total_value
        else:
            peak_value = max(
                [r["total_value"] for r in self.weekly_records] + [total_value]
            )

        drawdown = (
            ((peak_value - total_value) / peak_value) * 100 if peak_value > 0 else 0
        )

        self.weekly_records.append(
            {
                "week_start": week_start,
                "capital": self.capital,
                "positions_value": positions_value,
                "total_value": total_value,
                "num_positions": len(self.positions),
                "total_return_pct": total_return,
                "peak_value": peak_value,
                "drawdown_pct": drawdown,
            }
        )

    def get_closed_positions(self) -> pd.DataFrame:
        """Get DataFrame of all closed positions.

        Returns:
            DataFrame with position details
        """
        return pd.DataFrame([p.to_dict() for p in self.closed_positions])

    def calculate_metrics(self) -> Dict:
        """Calculate overall portfolio performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.weekly_records:
            return {}

        df = pd.DataFrame(self.weekly_records)

        # Final metrics
        final_value = df["total_value"].iloc[-1]
        total_return = df["total_return_pct"].iloc[-1]
        max_drawdown = df["drawdown_pct"].max()

        # Returns for Sharpe calculation
        weekly_returns = df["total_value"].pct_change().dropna() * 100

        sharpe_ratio = 0.0
        if len(weekly_returns) > 0 and weekly_returns.std() > 0:
            sharpe_ratio = (weekly_returns.mean() / weekly_returns.std()) * np.sqrt(
                52
            )  # Annualized

        # Position metrics
        closed_df = self.get_closed_positions()
        num_trades = len(closed_df)
        win_rate = 0.0
        avg_return = 0.0

        if num_trades > 0:
            winners = closed_df[closed_df["return_pct"] > 0]
            win_rate = len(winners) / num_trades * 100
            avg_return = closed_df["return_pct"].mean()

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "num_trades": num_trades,
            "win_rate_pct": win_rate,
            "avg_return_pct": avg_return,
            "num_weeks": len(df),
        }

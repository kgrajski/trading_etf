# -*- coding: utf-8 -*-
"""
Mean-Reversion Backtester

Simulates a mean-reversion trading strategy:
- Entry: Buy bottom 5% weekly losers with loss >= MIN_LOSS_PCNT
- Exit: Stop-loss, profit-target, or max-hold timeout
- Capital: Fixed pool with losses reducing capacity

Author: Assistant
Created: 2026-01-24
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for mean-reversion backtest."""
    
    # Capital management
    initial_capital: float = 10_000.0
    max_active_trades: int = 10
    
    # Entry parameters
    bottom_percentile: float = 0.05  # Bottom 5%
    min_loss_pcnt: float = 2.0  # Minimum weekly loss to qualify (%)
    
    # Exit parameters
    stop_loss_pcnt: float = 10.0  # Stop-loss threshold (%)
    profit_exit_pcnt: float = 10.0  # Profit target (%)
    max_hold_weeks: int = 8  # Force exit after N weeks
    
    # Execution
    limit_order_at_close: bool = True  # Entry limit at prior week's close
    
    # Future extensions (not implemented yet)
    trailing_stop: bool = False
    trailing_profit: bool = False


@dataclass
class Trade:
    """Represents a single trade."""
    
    trade_id: int
    symbol: str
    name: str
    
    # Entry
    signal_week_start: str  # Week when signal was generated
    entry_date: Optional[str] = None
    entry_price: Optional[float] = None
    shares: int = 0
    position_value: float = 0.0
    
    # Signal data
    weekly_return_pcnt: float = 0.0  # The loss that triggered entry
    
    # Exit
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop_loss', 'profit_exit', 'max_hold', 'end_of_data'
    
    # P&L
    pnl_dollars: float = 0.0
    pnl_pcnt: float = 0.0
    hold_days: int = 0
    
    # Status
    is_open: bool = True
    limit_price: float = 0.0  # Limit order price for entry
    limit_expires: Optional[str] = None  # Expiry date for limit order


@dataclass
class WeeklySummary:
    """Weekly summary of backtest activity."""
    
    week_start: str
    week_end: str
    
    # Signal generation
    n_candidates: int = 0  # Symbols in bottom 5%
    n_qualified: int = 0  # Symbols meeting MIN_LOSS_PCNT
    
    # Trading activity
    new_entries: int = 0
    exits_stop_loss: int = 0
    exits_profit: int = 0
    exits_timeout: int = 0
    pending_orders: int = 0  # Limit orders not yet filled
    
    # Positions
    active_positions: int = 0
    
    # Capital
    committed_capital: float = 0.0
    available_capital: float = 0.0
    total_capital: float = 0.0
    
    # P&L
    realized_pnl_week: float = 0.0
    unrealized_pnl: float = 0.0
    cumulative_realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    weekly_summaries: List[WeeklySummary] = field(default_factory=list)
    
    # Aggregate metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_return_pcnt: float = 0.0
    max_drawdown_pcnt: float = 0.0
    
    # Time series for plotting
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


class MeanReversionBacktester:
    """
    Backtester for mean-reversion strategy on ETFs.
    
    Strategy:
    - Each week, identify bottom 5% performers with loss >= MIN_LOSS_PCNT
    - Place limit orders at prior week's close price
    - If filled, hold until stop-loss, profit-target, or max-hold
    - Track P&L with realistic position sizing
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[Trade] = []
        self.weekly_summaries: List[WeeklySummary] = []
        self.next_trade_id = 1
        
        # Capital tracking
        self.current_capital = config.initial_capital
        self.committed_capital = 0.0
        self.cumulative_realized_pnl = 0.0
        
        # Data cache
        self.daily_data: Dict[str, pd.DataFrame] = {}
        self.weekly_data: Dict[str, pd.DataFrame] = {}
        
    def load_data(
        self,
        daily_dir: Path,
        weekly_dir: Path,
        symbols: Optional[List[str]] = None,
    ) -> None:
        """
        Load daily and weekly data for all symbols.
        
        Args:
            daily_dir: Path to daily CSV files
            weekly_dir: Path to weekly CSV files
            symbols: Optional list of symbols to load (None = all)
        """
        logger.info(f"Loading data from {daily_dir} and {weekly_dir}")
        
        # Load daily data
        daily_files = list(daily_dir.glob("*.csv"))
        for csv_path in daily_files:
            symbol = csv_path.stem
            if symbols and symbol not in symbols:
                continue
            
            df = pd.read_csv(csv_path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            self.daily_data[symbol] = df
        
        # Load weekly data
        weekly_files = list(weekly_dir.glob("*.csv"))
        for csv_path in weekly_files:
            symbol = csv_path.stem
            if symbols and symbol not in symbols:
                continue
            
            df = pd.read_csv(csv_path, parse_dates=["week_start", "week_end"])
            df = df.sort_values("week_start").reset_index(drop=True)
            self.weekly_data[symbol] = df
        
        logger.info(f"Loaded {len(self.daily_data)} daily, {len(self.weekly_data)} weekly files")
    
    def get_trading_weeks(self) -> List[Tuple[str, str]]:
        """
        Get list of (week_start, week_end) tuples from the data.
        
        Returns weeks that have data for at least some symbols.
        """
        all_weeks = set()
        for symbol, df in self.weekly_data.items():
            for _, row in df.iterrows():
                all_weeks.add((
                    row["week_start"].strftime("%Y-%m-%d"),
                    row["week_end"].strftime("%Y-%m-%d")
                ))
        
        return sorted(all_weeks, key=lambda x: x[0])
    
    def get_weekly_returns(self, week_start: str) -> pd.DataFrame:
        """
        Get all symbols' returns for a given week.
        
        Args:
            week_start: Week start date string (YYYY-MM-DD)
        
        Returns:
            DataFrame with symbol, name, log_return, close, etc.
        """
        records = []
        week_start_dt = pd.Timestamp(week_start)
        
        for symbol, df in self.weekly_data.items():
            week_row = df[df["week_start"] == week_start_dt]
            if len(week_row) == 0:
                continue
            
            row = week_row.iloc[0]
            
            # Skip if no log_return
            if pd.isna(row.get("log_return")):
                continue
            
            records.append({
                "symbol": symbol,
                "week_start": week_start,
                "log_return": row["log_return"],
                "pct_return": (np.exp(row["log_return"]) - 1) * 100,
                "close": row["close"],
                "name": row.get("symbol", symbol),  # Use symbol as name fallback
            })
        
        return pd.DataFrame(records)
    
    def identify_candidates(self, week_start: str) -> pd.DataFrame:
        """
        Identify mean-reversion entry candidates.
        
        Args:
            week_start: Week when the loss occurred
        
        Returns:
            DataFrame of qualified candidates sorted by loss (worst first)
        """
        returns_df = self.get_weekly_returns(week_start)
        
        if len(returns_df) == 0:
            return pd.DataFrame()
        
        # Bottom percentile
        n_bottom = max(1, int(len(returns_df) * self.config.bottom_percentile))
        bottom_df = returns_df.nsmallest(n_bottom, "pct_return")
        
        # Filter by minimum loss
        qualified = bottom_df[bottom_df["pct_return"] <= -self.config.min_loss_pcnt]
        
        return qualified.reset_index(drop=True)
    
    def get_next_week_trading_days(self, week_end: str) -> List[str]:
        """
        Get trading days for the week after the signal week.
        
        Args:
            week_end: End date of signal week
        
        Returns:
            List of trading day date strings
        """
        # Find first symbol with daily data
        if not self.daily_data:
            return []
        
        sample_df = next(iter(self.daily_data.values()))
        week_end_dt = pd.Timestamp(week_end)
        
        # Get days in the next 7 calendar days
        next_week_start = week_end_dt + timedelta(days=1)
        next_week_end = next_week_start + timedelta(days=6)
        
        trading_days = sample_df[
            (sample_df["date"] > week_end_dt) &
            (sample_df["date"] <= next_week_end + timedelta(days=7))
        ]["date"].dt.strftime("%Y-%m-%d").tolist()
        
        return trading_days[:5]  # Max 5 trading days
    
    def check_limit_fill(self, symbol: str, date: str, limit_price: float) -> Optional[float]:
        """
        Check if a limit buy order would fill on a given day.
        
        Args:
            symbol: Stock symbol
            date: Date to check
            limit_price: Limit order price
        
        Returns:
            Fill price if filled, None if not filled
        """
        if symbol not in self.daily_data:
            return None
        
        df = self.daily_data[symbol]
        date_dt = pd.Timestamp(date)
        day_row = df[df["date"] == date_dt]
        
        if len(day_row) == 0:
            return None
        
        row = day_row.iloc[0]
        
        # Check if low touched or went below limit price
        if row["low"] <= limit_price:
            # Fill at limit price (optimistic) or open if gapped below
            fill_price = min(limit_price, row["open"])
            return fill_price
        
        return None
    
    def check_exit_conditions(
        self,
        trade: Trade,
        date: str,
    ) -> Optional[Tuple[str, float]]:
        """
        Check if exit conditions are met for an open trade.
        
        Args:
            trade: Open trade to check
            date: Current date
        
        Returns:
            (exit_reason, exit_price) if should exit, None otherwise
        """
        if trade.symbol not in self.daily_data:
            return None
        
        df = self.daily_data[trade.symbol]
        date_dt = pd.Timestamp(date)
        day_row = df[df["date"] == date_dt]
        
        if len(day_row) == 0:
            return None
        
        row = day_row.iloc[0]
        entry_price = trade.entry_price
        
        # Calculate price levels
        stop_price = entry_price * (1 - self.config.stop_loss_pcnt / 100)
        profit_price = entry_price * (1 + self.config.profit_exit_pcnt / 100)
        
        # Check stop-loss (hit if low goes below stop)
        if row["low"] <= stop_price:
            # Exit at stop price (or open if gapped below)
            exit_price = min(stop_price, row["open"])
            return ("stop_loss", exit_price)
        
        # Check profit target (hit if high goes above target)
        if row["high"] >= profit_price:
            # Exit at profit price (or open if gapped above)
            exit_price = max(profit_price, row["open"])
            return ("profit_exit", exit_price)
        
        # Check max hold (timeout)
        entry_date = pd.Timestamp(trade.entry_date)
        hold_weeks = (date_dt - entry_date).days / 7
        
        if hold_weeks >= self.config.max_hold_weeks:
            # Exit at close
            return ("max_hold", row["close"])
        
        return None
    
    def calculate_position_size(self, price: float) -> Tuple[int, float]:
        """
        Calculate position size based on available capital.
        
        Args:
            price: Entry price
        
        Returns:
            (shares, position_value)
        """
        available = self.current_capital - self.committed_capital
        
        if available <= 0:
            return (0, 0.0)
        
        # Target position size
        target_size = self.config.initial_capital / self.config.max_active_trades
        position_budget = min(available, target_size)
        
        # Calculate shares (integer)
        shares = int(position_budget / price)
        
        if shares == 0:
            return (0, 0.0)
        
        position_value = shares * price
        return (shares, position_value)
    
    def get_all_trading_days(self) -> List[str]:
        """Get all trading days from the data."""
        if not self.daily_data:
            return []
        
        sample_df = next(iter(self.daily_data.values()))
        return sample_df["date"].dt.strftime("%Y-%m-%d").tolist()
    
    def get_week_for_date(self, date: str) -> Optional[Tuple[str, str]]:
        """Get the (week_start, week_end) for a given date."""
        date_dt = pd.Timestamp(date)
        
        for week_start, week_end in self.get_trading_weeks():
            ws = pd.Timestamp(week_start)
            we = pd.Timestamp(week_end)
            if ws <= date_dt <= we:
                return (week_start, week_end)
        
        return None
    
    def run(self, start_week: Optional[str] = None, end_week: Optional[str] = None) -> BacktestResult:
        """
        Run the backtest.
        
        Logic:
        1. At end of each week, identify bottom 5% losers as candidates
        2. On Monday of next week, place limit orders at prior week's close
        3. Each day, check for limit fills and exit conditions
        4. Continue until end of data
        
        Args:
            start_week: Optional start week (YYYY-MM-DD), defaults to first available
            end_week: Optional end week (YYYY-MM-DD), defaults to last available
        
        Returns:
            BacktestResult with all trades and metrics
        """
        trading_weeks = self.get_trading_weeks()
        
        if not trading_weeks:
            logger.error("No trading weeks available")
            return BacktestResult(config=self.config)
        
        # Filter weeks
        if start_week:
            trading_weeks = [(s, e) for s, e in trading_weeks if s >= start_week]
        if end_week:
            trading_weeks = [(s, e) for s, e in trading_weeks if s <= end_week]
        
        logger.info(f"Running backtest over {len(trading_weeks)} weeks")
        logger.info(f"Config: SL={self.config.stop_loss_pcnt}%, TP={self.config.profit_exit_pcnt}%, "
                   f"MaxHold={self.config.max_hold_weeks}w, MinLoss={self.config.min_loss_pcnt}%")
        
        # Track pending orders from previous week
        pending_orders: List[Trade] = []
        
        # Process each week
        for week_idx, (week_start, week_end) in enumerate(trading_weeks):
            summary = self._process_week_v2(week_start, week_end, week_idx, pending_orders)
            self.weekly_summaries.append(summary)
            
            # Generate new candidates for NEXT week (at end of current week)
            if week_idx < len(trading_weeks) - 1:
                pending_orders = self._generate_candidates_for_next_week(week_start, week_end)
            else:
                pending_orders = []
        
        # Close any remaining open positions at end
        self._close_all_positions("end_of_data")
        
        # Calculate final metrics
        result = self._build_result()
        
        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Win rate: {result.win_rate:.1f}%, Total P&L: ${result.total_pnl:.2f}")
        
        return result
    
    def _generate_candidates_for_next_week(self, week_start: str, week_end: str) -> List[Trade]:
        """
        Generate candidate trades from this week's losers.
        These will be processed as limit orders next week.
        """
        candidates = self.identify_candidates(week_start)
        
        if len(candidates) == 0:
            return []
        
        # Check available slots
        active_trades = sum(1 for t in self.trades if t.is_open and t.entry_date is not None)
        available_slots = self.config.max_active_trades - active_trades
        
        if available_slots <= 0:
            return []
        
        pending = []
        for _, candidate in candidates.head(available_slots).iterrows():
            trade = Trade(
                trade_id=self.next_trade_id,
                symbol=candidate["symbol"],
                name=candidate.get("name", candidate["symbol"]),
                signal_week_start=week_start,
                weekly_return_pcnt=candidate["pct_return"],
                limit_price=candidate["close"],
            )
            self.next_trade_id += 1
            pending.append(trade)
        
        return pending
    
    def _process_week_v2(
        self,
        week_start: str,
        week_end: str,
        week_idx: int,
        pending_orders: List[Trade],
    ) -> WeeklySummary:
        """
        Process a single week of the backtest.
        
        Args:
            week_start: Start of this week
            week_end: End of this week  
            week_idx: Week index
            pending_orders: Limit orders from previous week to try to fill
        """
        summary = WeeklySummary(week_start=week_start, week_end=week_end)
        
        # Get trading days for THIS week
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        all_days = self.get_all_trading_days()
        trading_days = [d for d in all_days if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            return summary
        
        first_day = trading_days[0]
        
        # 1. Try to fill pending limit orders from last week (on Monday only)
        for trade in pending_orders:
            fill_price = self.check_limit_fill(trade.symbol, first_day, trade.limit_price)
            
            if fill_price:
                shares, position_value = self.calculate_position_size(fill_price)
                if shares > 0:
                    trade.entry_date = first_day
                    trade.entry_price = fill_price
                    trade.shares = shares
                    trade.position_value = position_value
                    trade.is_open = True
                    self.committed_capital += position_value
                    self.trades.append(trade)
                    summary.new_entries += 1
                    logger.debug(f"Filled {trade.symbol} @ {fill_price:.2f}, {shares} shares")
                else:
                    trade.is_open = False
                    trade.exit_reason = "no_capital"
                    self.trades.append(trade)
            else:
                # Limit not hit on Monday - order expires
                trade.is_open = False
                trade.exit_reason = "expired"
                self.trades.append(trade)
        
        # 2. Check exits for all open positions
        for trade in self.trades:
            if not trade.is_open or trade.entry_date is None:
                continue
            
            for day in trading_days:
                # Skip days before entry
                if pd.Timestamp(day) <= pd.Timestamp(trade.entry_date):
                    continue
                
                exit_result = self.check_exit_conditions(trade, day)
                if exit_result:
                    reason, price = exit_result
                    self._close_trade(trade, day, price, reason)
                    
                    if reason == "stop_loss":
                        summary.exits_stop_loss += 1
                    elif reason == "profit_exit":
                        summary.exits_profit += 1
                    elif reason == "max_hold":
                        summary.exits_timeout += 1
                    
                    summary.realized_pnl_week += trade.pnl_dollars
                    break
        
        # 3. Update summary counts
        returns_df = self.get_weekly_returns(week_start)
        summary.n_candidates = len(returns_df)
        
        candidates = self.identify_candidates(week_start)
        summary.n_qualified = len(candidates)
        
        summary.active_positions = sum(1 for t in self.trades if t.is_open and t.entry_date is not None)
        summary.committed_capital = self.committed_capital
        summary.available_capital = self.current_capital - self.committed_capital
        summary.total_capital = self.current_capital
        summary.cumulative_realized_pnl = self.cumulative_realized_pnl
        
        return summary
    
    def _close_trade(self, trade: Trade, date: str, price: float, reason: str) -> None:
        """Close a trade and update capital."""
        trade.exit_date = date
        trade.exit_price = price
        trade.exit_reason = reason
        trade.is_open = False
        
        # Calculate P&L
        trade.pnl_dollars = (price - trade.entry_price) * trade.shares
        trade.pnl_pcnt = ((price / trade.entry_price) - 1) * 100
        
        # Calculate hold days
        entry_dt = pd.Timestamp(trade.entry_date)
        exit_dt = pd.Timestamp(date)
        trade.hold_days = (exit_dt - entry_dt).days
        
        # Update capital
        self.committed_capital -= trade.position_value
        self.current_capital += trade.pnl_dollars
        self.cumulative_realized_pnl += trade.pnl_dollars
        
        logger.debug(f"Closed {trade.symbol}: {reason}, P&L ${trade.pnl_dollars:.2f} ({trade.pnl_pcnt:.1f}%)")
    
    def _close_all_positions(self, reason: str) -> None:
        """Close all open positions at last available price."""
        for trade in self.trades:
            if not trade.is_open:
                continue
            
            if trade.entry_date is None:
                # Never filled, just cancel
                trade.is_open = False
                trade.exit_reason = "cancelled"
                continue
            
            # Find last available price
            if trade.symbol in self.daily_data:
                df = self.daily_data[trade.symbol]
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    self._close_trade(trade, last_row["date"].strftime("%Y-%m-%d"),
                                     last_row["close"], reason)
    
    def _build_result(self) -> BacktestResult:
        """Build the final backtest result."""
        result = BacktestResult(config=self.config)
        result.trades = self.trades
        result.weekly_summaries = self.weekly_summaries
        
        # Aggregate trade metrics
        closed_trades = [t for t in self.trades if t.exit_date is not None and t.entry_date is not None]
        result.total_trades = len(closed_trades)
        result.winning_trades = sum(1 for t in closed_trades if t.pnl_dollars > 0)
        result.losing_trades = sum(1 for t in closed_trades if t.pnl_dollars <= 0)
        result.win_rate = (result.winning_trades / result.total_trades * 100) if result.total_trades > 0 else 0
        
        result.total_pnl = sum(t.pnl_dollars for t in closed_trades)
        result.total_return_pcnt = (result.total_pnl / self.config.initial_capital) * 100
        
        # Build equity curve
        result.equity_curve = self._build_equity_curve(closed_trades)
        
        # Calculate max drawdown
        if len(result.equity_curve) > 0:
            equity = result.equity_curve["capital"]
            peak = equity.cummax()
            drawdown = (equity - peak) / peak * 100
            result.max_drawdown_pcnt = abs(drawdown.min())
        
        return result
    
    def _build_equity_curve(self, trades: List[Trade]) -> pd.DataFrame:
        """Build equity curve from closed trades."""
        if not trades:
            return pd.DataFrame()
        
        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)
        
        records = []
        capital = self.config.initial_capital
        peak_capital = capital
        
        for trade in sorted_trades:
            capital += trade.pnl_dollars
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
            
            records.append({
                "date": trade.exit_date,
                "symbol": trade.symbol,
                "pnl_dollars": trade.pnl_dollars,
                "pnl_pcnt": trade.pnl_pcnt,
                "capital": capital,
                "peak_capital": peak_capital,
                "drawdown_pcnt": drawdown,
                "cumulative_return_pcnt": ((capital / self.config.initial_capital) - 1) * 100,
            })
        
        return pd.DataFrame(records)

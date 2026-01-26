#!/usr/bin/env python3
"""
19.1-full-backtest.py

Complete Mean-Reversion Backtester with All Features.

Consolidates all bells & whistles from 19, 19d, and 21a:
- Regime awareness (SPY vs MA50)
- Configurable boost direction (bull/bear/none)
- Detailed trade logging
- Period analysis (historical / prior 52w / recent 52w)
- Cumulative P&L by exit reason
- Slippage detection with alarms

This is the reference single-run backtester. The backtest logic here
should match 19.1b-grid-search exactly.

Outputs:
- experiments/exp019_1_full/{date}/
  - config.json
  - trades.csv
  - summary.json
  - dashboard.html
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp019_1_full"

# Capital parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10

# Entry parameters
BOTTOM_PERCENTILE = 0.05  # Bottom 5% losers
MIN_LOSS_PCNT = 2.0  # Minimum weekly loss to qualify

# Exit parameters
STOP_LOSS_PCNT = 16.0
PROFIT_EXIT_PCNT = 10.0
MAX_HOLD_WEEKS = 1

# Regime parameters
USE_REGIME = True
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50

# Boost parameters
BOOST_DIRECTION = "bear"  # Options: "bull", "bear", "none"
BOOST_MULTIPLIER = 1.10   # 10% larger positions when condition met

# Analysis parameters
RECENT_WEEKS = 52  # Split point for period analysis
SLIPPAGE_ALARM_THRESHOLD = 5.0  # Warn if win rate drops > this %

# Risk metrics
RISK_FREE_RATE = 0.05  # Annualized risk-free rate (~5% T-bill)

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME


# =============================================================================
# Data Loading
# =============================================================================

def get_target_symbols() -> List[str]:
    """Get list of target ETF symbols."""
    path = METADATA_DIR / "filtered_etfs.json"
    if not path.exists():
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    with open(path) as f:
        data = json.load(f)
    return [etf["symbol"] for etf in data.get("etfs", [])]


def load_daily_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV data for symbols."""
    data = {}
    for symbol in symbols:
        path = DAILY_DATA_DIR / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date")
            data[symbol] = df
    return data


def load_weekly_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load weekly data for symbols."""
    data = {}
    for symbol in symbols:
        path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["week_start"])
            df = df.sort_values("week_start")
            data[symbol] = df
    return data


def calculate_regime(daily_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Calculate market regime for each trading day."""
    if not USE_REGIME or REGIME_SYMBOL not in daily_data:
        return {}
    
    df = daily_data[REGIME_SYMBOL].copy()
    df["ma"] = df["close"].rolling(window=REGIME_MA_PERIOD).mean()
    df["is_bull"] = df["close"] > df["ma"]
    
    regime_by_date = {}
    for _, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        regime_by_date[date_str] = {
            "is_bull": bool(row["is_bull"]) if pd.notna(row["is_bull"]) else True,
            "close": float(row["close"]),
            "ma": float(row["ma"]) if pd.notna(row["ma"]) else None,
        }
    return regime_by_date


def get_trading_weeks(weekly_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
    """Get list of (week_start, week_end) tuples."""
    all_weeks = set()
    for df in weekly_data.values():
        for _, row in df.iterrows():
            week_start = row["week_start"]
            if pd.notna(week_start):
                all_weeks.add(week_start)
    
    sorted_weeks = sorted(all_weeks)
    trading_weeks = []
    for i, ws in enumerate(sorted_weeks):
        if i + 1 < len(sorted_weeks):
            we = sorted_weeks[i + 1] - pd.Timedelta(days=1)
        else:
            we = ws + pd.Timedelta(days=6)
        trading_weeks.append((ws.strftime("%Y-%m-%d"), we.strftime("%Y-%m-%d")))
    return trading_weeks


# =============================================================================
# Risk Metrics (MUST MATCH 19.1b)
# =============================================================================

def compute_sharpe_ratio(pnl_series: List[float], periods_per_year: int = 52) -> float:
    """
    Compute annualized Sharpe Ratio.
    
    Args:
        pnl_series: List of P&L values per period (e.g., weekly)
        periods_per_year: Number of periods in a year (52 for weekly)
    
    Returns:
        Annualized Sharpe Ratio
    """
    if len(pnl_series) < 2:
        return 0.0
    
    # Convert P&L to returns (as % of initial capital)
    returns = [pnl / INITIAL_CAPITAL for pnl in pnl_series]
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Annualize: multiply mean by periods, std by sqrt(periods)
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    sharpe = (annual_return - RISK_FREE_RATE) / annual_std
    return sharpe


def compute_sortino_ratio(pnl_series: List[float], periods_per_year: int = 52) -> float:
    """
    Compute annualized Sortino Ratio (only penalizes downside volatility).
    
    Args:
        pnl_series: List of P&L values per period
        periods_per_year: Number of periods in a year
    
    Returns:
        Annualized Sortino Ratio
    """
    if len(pnl_series) < 2:
        return 0.0
    
    returns = [pnl / INITIAL_CAPITAL for pnl in pnl_series]
    
    mean_return = np.mean(returns)
    
    # Downside deviation: std of returns below target (0)
    downside_returns = [r for r in returns if r < 0]
    if len(downside_returns) < 2:
        # No downside volatility - infinite Sortino, cap at large value
        return 10.0 if mean_return > 0 else 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 10.0 if mean_return > 0 else 0.0
    
    annual_return = mean_return * periods_per_year
    annual_downside_std = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annual_return - RISK_FREE_RATE) / annual_downside_std
    return sortino


def compute_top5_concentration(pnl_list: List[float]) -> float:
    """
    Compute what percentage of total P&L comes from top 5 trades.
    
    Lower is better - means gains are diversified.
    
    Returns:
        Percentage (0-100) of total P&L from top 5 trades
    """
    if not pnl_list:
        return 0.0
    
    total_pnl = sum(pnl_list)
    if total_pnl <= 0:
        return 100.0  # All losses or zero - concentration metric not meaningful
    
    # Sort by P&L descending, take top 5
    sorted_pnl = sorted(pnl_list, reverse=True)
    top5_pnl = sum(sorted_pnl[:5])
    
    concentration = (top5_pnl / total_pnl) * 100 if total_pnl > 0 else 100.0
    return min(concentration, 100.0)  # Cap at 100%


def compute_profit_factor(pnl_list: List[float]) -> float:
    """
    Compute profit factor: gross profit / gross loss.
    
    > 1 means wins outpace losses.
    """
    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    
    if gross_loss == 0:
        return 10.0 if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


# =============================================================================
# Core Backtest Logic (MUST MATCH 19.1b)
# =============================================================================

def get_position_multiplier(is_bull_week: bool) -> float:
    """
    Get position size multiplier based on regime and boost config.
    
    This function encapsulates the boost logic to ensure consistency
    between 19.1 and 19.1b.
    """
    if not USE_REGIME or BOOST_DIRECTION == "none":
        return 1.0
    
    if BOOST_DIRECTION == "bear" and not is_bull_week:
        return BOOST_MULTIPLIER
    elif BOOST_DIRECTION == "bull" and is_bull_week:
        return BOOST_MULTIPLIER
    else:
        return 1.0


def run_backtest(
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    regime_by_date: Dict[str, Dict],
    stop_loss_pcnt: float = STOP_LOSS_PCNT,
    profit_exit_pcnt: float = PROFIT_EXIT_PCNT,
    max_hold_weeks: int = MAX_HOLD_WEEKS,
) -> Tuple[List[Dict], Dict]:
    """
    Run backtest with full trade logging.
    
    Returns:
        trades: List of trade dictionaries with full metadata
        summary: Aggregate metrics including regime breakdown
    """
    # Build trading calendar
    all_trading_days = set()
    for df in daily_data.values():
        for d in df["date"]:
            all_trading_days.add(d.strftime("%Y-%m-%d"))
    all_trading_days = sorted(all_trading_days)
    
    trading_weeks = get_trading_weeks(weekly_data)
    
    # State
    trades = []
    next_trade_id = 1
    current_capital = INITIAL_CAPITAL
    committed_capital = 0.0
    pending_orders = []
    
    # Regime tracking
    bull_weeks = 0
    bear_weeks = 0
    
    # Process each week
    for week_idx, (week_start, week_end) in enumerate(trading_weeks):
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        # Get trading days for this week
        trading_days = [d for d in all_trading_days 
                        if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            continue
        
        first_day = trading_days[0]
        last_day = trading_days[-1]
        
        # Determine regime
        regime_info = regime_by_date.get(first_day, {"is_bull": True})
        is_bull_week = regime_info.get("is_bull", True) if USE_REGIME else True
        
        if is_bull_week:
            bull_weeks += 1
        else:
            bear_weeks += 1
        
        position_multiplier = get_position_multiplier(is_bull_week)
        
        # 1. Try to fill pending orders
        for order in pending_orders:
            symbol = order["symbol"]
            limit_price = order["limit_price"]
            signal_week = order["signal_week"]
            signal_return = order["signal_return"]
            
            if symbol not in daily_data:
                continue
            
            day_df = daily_data[symbol]
            day_row = day_df[day_df["date"] == pd.Timestamp(first_day)]
            
            if len(day_row) == 0:
                continue
            
            row = day_row.iloc[0]
            
            # Check if limit hit
            if row["low"] <= limit_price:
                fill_price = min(limit_price, row["open"])
                
                available = current_capital - committed_capital
                if available <= 0:
                    continue
                
                base_target_size = INITIAL_CAPITAL / MAX_ACTIVE_TRADES
                target_size = base_target_size * position_multiplier
                position_budget = min(available, target_size)
                shares = int(position_budget / fill_price)
                
                if shares > 0:
                    position_value = shares * fill_price
                    committed_capital += position_value
                    
                    trades.append({
                        "trade_id": next_trade_id,
                        "symbol": symbol,
                        "signal_week": signal_week,
                        "signal_return_pcnt": signal_return,
                        "entry_date": first_day,
                        "entry_week_idx": week_idx,
                        "entry_price": fill_price,
                        "limit_price": limit_price,
                        "shares": shares,
                        "position_value": position_value,
                        "position_multiplier": position_multiplier,
                        "is_bull_entry": is_bull_week,
                        "regime_close": regime_info.get("close"),
                        "regime_ma": regime_info.get("ma"),
                        "is_open": True,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "pnl_dollars": 0.0,
                        "pnl_pcnt": 0.0,
                        "hold_days": 0,
                    })
                    next_trade_id += 1
        
        # Mark unfilled orders as expired
        for order in pending_orders:
            symbol = order["symbol"]
            filled = any(t["symbol"] == symbol and t["entry_date"] == first_day 
                        for t in trades)
            if not filled:
                trades.append({
                    "trade_id": next_trade_id,
                    "symbol": symbol,
                    "signal_week": order["signal_week"],
                    "signal_return_pcnt": order["signal_return"],
                    "entry_date": None,
                    "entry_week_idx": None,
                    "entry_price": None,
                    "limit_price": order["limit_price"],
                    "shares": 0,
                    "position_value": 0,
                    "position_multiplier": None,
                    "is_bull_entry": None,
                    "regime_close": None,
                    "regime_ma": None,
                    "is_open": False,
                    "exit_date": None,
                    "exit_price": None,
                    "exit_reason": "expired",
                    "pnl_dollars": 0.0,
                    "pnl_pcnt": 0.0,
                    "hold_days": 0,
                })
                next_trade_id += 1
        
        pending_orders = []
        
        # 2. Check exits for open trades
        for trade in trades:
            if not trade["is_open"]:
                continue
            
            symbol = trade["symbol"]
            entry_price = trade["entry_price"]
            entry_date = trade["entry_date"]
            entry_week_idx = trade["entry_week_idx"]
            
            if symbol not in daily_data:
                continue
            
            stop_price = entry_price * (1 - stop_loss_pcnt / 100)
            profit_price = entry_price * (1 + profit_exit_pcnt / 100)
            max_hold_exit_week = entry_week_idx + max_hold_weeks - 1
            
            for day in trading_days:
                if pd.Timestamp(day) <= pd.Timestamp(entry_date):
                    continue
                
                day_df = daily_data[symbol]
                day_row = day_df[day_df["date"] == pd.Timestamp(day)]
                
                if len(day_row) == 0:
                    continue
                
                row = day_row.iloc[0]
                
                # Check stop loss (highest priority)
                if row["low"] <= stop_price:
                    exit_price = min(stop_price, row["open"])
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "stop_loss"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
                    trade["hold_days"] = (pd.Timestamp(day) - pd.Timestamp(entry_date)).days
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Check profit target
                if row["high"] >= profit_price:
                    exit_price = max(profit_price, row["open"])
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "profit_exit"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
                    trade["hold_days"] = (pd.Timestamp(day) - pd.Timestamp(entry_date)).days
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Check max hold (Friday exit at open)
                is_last_day = (day == last_day)
                is_max_hold_week = (week_idx >= max_hold_exit_week)
                
                if is_last_day and is_max_hold_week:
                    exit_price = row["open"]
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "max_hold"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
                    trade["hold_days"] = (pd.Timestamp(day) - pd.Timestamp(entry_date)).days
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
        
        # 3. Generate new candidates
        returns_data = []
        for symbol, df in weekly_data.items():
            week_row = df[df["week_start"] == week_start_dt]
            if len(week_row) == 0:
                continue
            
            row = week_row.iloc[0]
            if pd.isna(row.get("log_return")):
                continue
            
            returns_data.append({
                "symbol": symbol,
                "log_return": row["log_return"],
                "pct_return": (np.exp(row["log_return"]) - 1) * 100,
                "close": row["close"],
            })
        
        if not returns_data:
            continue
        
        returns_df = pd.DataFrame(returns_data)
        n_bottom = max(1, int(len(returns_df) * BOTTOM_PERCENTILE))
        bottom_df = returns_df.nsmallest(n_bottom, "pct_return")
        qualified = bottom_df[bottom_df["pct_return"] <= -MIN_LOSS_PCNT]
        
        active_count = sum(1 for t in trades if t["is_open"])
        available_slots = MAX_ACTIVE_TRADES - active_count
        
        for _, candidate in qualified.head(available_slots).iterrows():
            pending_orders.append({
                "symbol": candidate["symbol"],
                "limit_price": candidate["close"],
                "signal_week": week_start,
                "signal_return": candidate["pct_return"],
            })
    
    # Close remaining open trades
    for trade in trades:
        if trade["is_open"]:
            symbol = trade["symbol"]
            if symbol in daily_data:
                df = daily_data[symbol]
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    exit_price = last_row["close"]
                    trade["exit_date"] = last_row["date"].strftime("%Y-%m-%d")
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "end_of_data"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - trade["entry_price"]) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / trade["entry_price"]) - 1) * 100
                    trade["hold_days"] = (pd.Timestamp(trade["exit_date"]) - 
                                         pd.Timestamp(trade["entry_date"])).days
    
    # Build summary
    summary = {
        "bull_weeks": bull_weeks,
        "bear_weeks": bear_weeks,
        "total_weeks": bull_weeks + bear_weeks,
    }
    
    return trades, summary


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_trades(trades: List[Dict], period_name: str = "All") -> Dict:
    """Compute comprehensive metrics for a set of trades including risk metrics."""
    filled = [t for t in trades if t["entry_date"] is not None]
    expired = [t for t in trades if t["exit_reason"] == "expired"]
    
    if not filled:
        return {
            "period": period_name,
            "total_signals": len(trades),
            "filled_trades": 0,
            "expired_orders": len(expired),
            "win_rate": 0,
            "total_pnl": 0,
            "exit_reasons": {},
            "cumulative_pnl_by_reason": {},
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "top5_concentration": 100,
            "profit_factor": 0,
        }
    
    # Exit reason breakdown
    exit_reasons = {}
    pnl_by_reason = {}
    count_by_reason = {}
    
    for t in filled:
        reason = t["exit_reason"] or "open"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        pnl_by_reason[reason] = pnl_by_reason.get(reason, 0) + t["pnl_dollars"]
        count_by_reason[reason] = count_by_reason.get(reason, 0) + 1
    
    avg_pnl_by_reason = {r: pnl_by_reason[r] / count_by_reason[r] for r in pnl_by_reason}
    
    # Win/loss
    winners = [t for t in filled if t["pnl_dollars"] > 0]
    losers = [t for t in filled if t["pnl_dollars"] <= 0]
    
    # Regime breakdown
    bull_trades = [t for t in filled if t.get("is_bull_entry") is True]
    bear_trades = [t for t in filled if t.get("is_bull_entry") is False]
    
    bull_pnl = sum(t["pnl_dollars"] for t in bull_trades)
    bear_pnl = sum(t["pnl_dollars"] for t in bear_trades)
    
    bull_winners = sum(1 for t in bull_trades if t["pnl_dollars"] > 0)
    bear_winners = sum(1 for t in bear_trades if t["pnl_dollars"] > 0)
    
    total_pnl = sum(t["pnl_dollars"] for t in filled)
    pnl_list = [t["pnl_dollars"] for t in filled]
    
    # Compute weekly P&L for Sharpe/Sortino
    weekly_pnl = {}
    for t in filled:
        if t["exit_date"]:
            # Get the Monday of the exit week
            exit_dt = pd.Timestamp(t["exit_date"])
            week_start = (exit_dt - pd.Timedelta(days=exit_dt.dayofweek)).strftime("%Y-%m-%d")
            weekly_pnl[week_start] = weekly_pnl.get(week_start, 0) + t["pnl_dollars"]
    
    weekly_pnl_list = list(weekly_pnl.values())
    
    # Risk metrics
    sharpe = compute_sharpe_ratio(weekly_pnl_list)
    sortino = compute_sortino_ratio(weekly_pnl_list)
    top5_conc = compute_top5_concentration(pnl_list)
    profit_fact = compute_profit_factor(pnl_list)
    
    return {
        "period": period_name,
        "total_signals": len(trades),
        "filled_trades": len(filled),
        "expired_orders": len(expired),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": len(winners) / len(filled) * 100 if filled else 0,
        "total_pnl": total_pnl,
        "total_return_pcnt": total_pnl / INITIAL_CAPITAL * 100,
        "avg_pnl_per_trade": total_pnl / len(filled) if filled else 0,
        "exit_reasons": exit_reasons,
        "avg_pnl_by_reason": avg_pnl_by_reason,
        "cumulative_pnl_by_reason": pnl_by_reason,
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_pnl": bull_pnl,
        "bear_pnl": bear_pnl,
        "bull_win_rate": bull_winners / len(bull_trades) * 100 if bull_trades else 0,
        "bear_win_rate": bear_winners / len(bear_trades) * 100 if bear_trades else 0,
        # Risk metrics
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "top5_concentration": top5_conc,
        "profit_factor": profit_fact,
    }


def detect_slippage(prior_stats: Dict, recent_stats: Dict) -> Dict:
    """Compare periods for slippage detection."""
    alarms = []
    
    win_rate_delta = recent_stats["win_rate"] - prior_stats["win_rate"]
    if win_rate_delta < -SLIPPAGE_ALARM_THRESHOLD:
        alarms.append(f"⚠️ WIN RATE SLIPPAGE: {win_rate_delta:+.1f}% "
                     f"(Prior: {prior_stats['win_rate']:.1f}%, "
                     f"Recent: {recent_stats['win_rate']:.1f}%)")
    
    prior_avg = prior_stats.get("avg_pnl_per_trade", 0)
    recent_avg = recent_stats.get("avg_pnl_per_trade", 0)
    if prior_avg > 0 and recent_avg < prior_avg * 0.5:
        alarms.append(f"⚠️ AVG P&L SLIPPAGE: Dropped from ${prior_avg:.2f} to ${recent_avg:.2f}")
    
    # Max hold health check
    prior_max_hold = prior_stats.get("cumulative_pnl_by_reason", {}).get("max_hold", 0)
    recent_max_hold = recent_stats.get("cumulative_pnl_by_reason", {}).get("max_hold", 0)
    if recent_max_hold < 0 and prior_max_hold > 0:
        alarms.append(f"⚠️ MAX_HOLD DEGRADATION: Prior ${prior_max_hold:,.2f} → Recent ${recent_max_hold:,.2f}")
    
    return {
        "win_rate_delta": win_rate_delta,
        "avg_pnl_delta": recent_avg - prior_avg,
        "alarms": alarms,
    }


# =============================================================================
# Dashboard
# =============================================================================

def create_dashboard(
    trades: List[Dict],
    historical_stats: Dict,
    prior_stats: Dict,
    recent_stats: Dict,
    slippage: Dict,
    output_dir: Path,
) -> Path:
    """Create interactive dashboard with period comparison."""
    
    filled_trades = [t for t in trades if t["entry_date"] and t["exit_date"]]
    
    if not filled_trades:
        print("No filled trades to visualize")
        return None
    
    # Build equity curve
    sorted_trades = sorted(filled_trades, key=lambda t: t["exit_date"])
    equity = [INITIAL_CAPITAL]
    dates = [sorted_trades[0]["entry_date"]]
    
    capital = INITIAL_CAPITAL
    for t in sorted_trades:
        capital += t["pnl_dollars"]
        equity.append(capital)
        dates.append(t["exit_date"])
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "Equity Curve",
            "Cumulative P&L by Exit Reason",
            "P&L by Entry Regime",
        ),
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12,
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="#2E86AB", width=2),
        ),
        row=1, col=1,
    )
    
    # Exit reason P&L bars
    reasons = ["max_hold", "profit_exit", "stop_loss"]
    colors = {"max_hold": "#4CAF50", "profit_exit": "#2196F3", "stop_loss": "#F44336"}
    
    for i, (period_name, stats) in enumerate([("Historical", historical_stats), 
                                               ("Prior 52w", prior_stats),
                                               ("Recent 52w", recent_stats)]):
        cumul = stats.get("cumulative_pnl_by_reason", {})
        fig.add_trace(
            go.Bar(
                name=period_name,
                x=reasons,
                y=[cumul.get(r, 0) for r in reasons],
                text=[f"${cumul.get(r, 0):,.0f}" for r in reasons],
                textposition="outside",
            ),
            row=2, col=1,
        )
    
    # Regime P&L bars
    for period_name, stats in [("Historical", historical_stats), 
                                ("Prior 52w", prior_stats),
                                ("Recent 52w", recent_stats)]:
        fig.add_trace(
            go.Bar(
                name=period_name,
                x=["Bull Entry", "Bear Entry"],
                y=[stats.get("bull_pnl", 0), stats.get("bear_pnl", 0)],
                text=[f"${stats.get('bull_pnl', 0):,.0f}", f"${stats.get('bear_pnl', 0):,.0f}"],
                textposition="outside",
                showlegend=False,
            ),
            row=2, col=2,
        )
    
    fig.update_layout(
        title=f"<b>Full Backtest: {EXPERIMENT_NAME}</b><br>"
              f"<sub>SL={STOP_LOSS_PCNT}% | TP={PROFIT_EXIT_PCNT}% | MaxHold={MAX_HOLD_WEEKS}w | "
              f"Boost={BOOST_DIRECTION} {BOOST_MULTIPLIER}x</sub>",
        height=800,
        barmode="group",
    )
    
    # Build HTML with summary table
    alarm_html = ""
    if slippage["alarms"]:
        alarm_html = "<div style='background:#FFEBEE;padding:15px;border-radius:8px;margin:20px 0;'>"
        alarm_html += "<h3 style='color:#C62828;margin:0 0 10px 0;'>⚠️ SLIPPAGE ALARMS</h3>"
        for alarm in slippage["alarms"]:
            alarm_html += f"<p style='margin:5px 0;'>{alarm}</p>"
        alarm_html += "</div>"
    
    summary_table = f"""
    <table style="width:100%;border-collapse:collapse;margin:20px 0;">
        <tr style="background:#f0f0f0;">
            <th style="padding:10px;text-align:left;">Metric</th>
            <th style="padding:10px;text-align:right;">Historical</th>
            <th style="padding:10px;text-align:right;">Prior 52w</th>
            <th style="padding:10px;text-align:right;">Recent 52w</th>
        </tr>
        <tr>
            <td style="padding:8px;">Filled Trades</td>
            <td style="padding:8px;text-align:right;">{historical_stats['filled_trades']}</td>
            <td style="padding:8px;text-align:right;">{prior_stats['filled_trades']}</td>
            <td style="padding:8px;text-align:right;">{recent_stats['filled_trades']}</td>
        </tr>
        <tr style="background:#f9f9f9;">
            <td style="padding:8px;">Win Rate</td>
            <td style="padding:8px;text-align:right;">{historical_stats['win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{prior_stats['win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{recent_stats['win_rate']:.1f}%</td>
        </tr>
        <tr>
            <td style="padding:8px;">Total P&L</td>
            <td style="padding:8px;text-align:right;">${historical_stats['total_pnl']:,.2f}</td>
            <td style="padding:8px;text-align:right;">${prior_stats['total_pnl']:,.2f}</td>
            <td style="padding:8px;text-align:right;">${recent_stats['total_pnl']:,.2f}</td>
        </tr>
        <tr style="background:#f9f9f9;">
            <td style="padding:8px;">max_hold Cumul P&L</td>
            <td style="padding:8px;text-align:right;">${historical_stats.get('cumulative_pnl_by_reason', {}).get('max_hold', 0):,.2f}</td>
            <td style="padding:8px;text-align:right;">${prior_stats.get('cumulative_pnl_by_reason', {}).get('max_hold', 0):,.2f}</td>
            <td style="padding:8px;text-align:right;">${recent_stats.get('cumulative_pnl_by_reason', {}).get('max_hold', 0):,.2f}</td>
        </tr>
        <tr style="background:#e8f5e9;">
            <td style="padding:8px;"><strong>Sharpe Ratio</strong></td>
            <td style="padding:8px;text-align:right;">{historical_stats['sharpe_ratio']:.3f}</td>
            <td style="padding:8px;text-align:right;">{prior_stats['sharpe_ratio']:.3f}</td>
            <td style="padding:8px;text-align:right;">{recent_stats['sharpe_ratio']:.3f}</td>
        </tr>
        <tr style="background:#e3f2fd;">
            <td style="padding:8px;"><strong>Sortino Ratio</strong></td>
            <td style="padding:8px;text-align:right;">{historical_stats['sortino_ratio']:.3f}</td>
            <td style="padding:8px;text-align:right;">{prior_stats['sortino_ratio']:.3f}</td>
            <td style="padding:8px;text-align:right;">{recent_stats['sortino_ratio']:.3f}</td>
        </tr>
        <tr style="background:#fff3e0;">
            <td style="padding:8px;"><strong>Top-5 Concentration</strong></td>
            <td style="padding:8px;text-align:right;">{historical_stats['top5_concentration']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{prior_stats['top5_concentration']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{recent_stats['top5_concentration']:.1f}%</td>
        </tr>
        <tr style="background:#fce4ec;">
            <td style="padding:8px;"><strong>Profit Factor</strong></td>
            <td style="padding:8px;text-align:right;">{historical_stats['profit_factor']:.2f}</td>
            <td style="padding:8px;text-align:right;">{prior_stats['profit_factor']:.2f}</td>
            <td style="padding:8px;text-align:right;">{recent_stats['profit_factor']:.2f}</td>
        </tr>
    </table>
    """
    
    output_path = output_dir / "dashboard.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head><title>Full Backtest Dashboard</title>
<style>body {{ font-family: -apple-system, sans-serif; margin: 20px; }}</style>
</head>
<body>
    <h1>Full Backtest: {EXPERIMENT_NAME}</h1>
    <div style="background:#e8f4f8;padding:15px;border-radius:8px;">
        <strong>Config:</strong> SL={STOP_LOSS_PCNT}% | TP={PROFIT_EXIT_PCNT}% | MaxHold={MAX_HOLD_WEEKS}w | 
        Regime={REGIME_SYMBOL} MA{REGIME_MA_PERIOD} | Boost={BOOST_DIRECTION} {BOOST_MULTIPLIER}x
    </div>
    {alarm_html}
    <h2>Period Comparison</h2>
    {summary_table}
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>"""
    
    output_path.write_text(html_content)
    return output_path


# =============================================================================
# Helper Functions
# =============================================================================

def sortino_fmt(val: float) -> str:
    """Format Sortino ratio, handling the capped value of 10.0."""
    if val >= 10.0:
        return ">10"
    return f"{val:.3f}"


# =============================================================================
# Main
# =============================================================================

@workflow_script("19.1-full-backtest")
def main():
    """Run full backtest with all bells & whistles."""
    
    print("=" * 70)
    print("19.1 FULL BACKTEST")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Stop Loss:        {STOP_LOSS_PCNT}%")
    print(f"  Profit Target:    {PROFIT_EXIT_PCNT}%")
    print(f"  Max Hold:         {MAX_HOLD_WEEKS} week(s)")
    print(f"  Regime:           {REGIME_SYMBOL} vs MA{REGIME_MA_PERIOD}" if USE_REGIME else "  Regime:           Disabled")
    print(f"  Boost:            {BOOST_DIRECTION} {BOOST_MULTIPLIER}x" if BOOST_DIRECTION != "none" else "  Boost:            None")
    print(f"  Recent Period:    {RECENT_WEEKS} weeks")
    print()
    
    # Create output dir with date
    run_date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = OUTPUT_DIR / run_date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    symbols = get_target_symbols()
    symbols.append(REGIME_SYMBOL)
    symbols = list(set(symbols))
    
    daily_data = load_daily_data(symbols)
    weekly_data = load_weekly_data(symbols)
    regime_by_date = calculate_regime(daily_data)
    
    print(f"  Symbols: {len(daily_data)}")
    print()
    
    # Run backtest
    print("Running backtest...")
    trades, summary = run_backtest(daily_data, weekly_data, regime_by_date)
    print(f"  Total signals: {len(trades)}")
    print(f"  Bull weeks: {summary['bull_weeks']}, Bear weeks: {summary['bear_weeks']}")
    print()
    
    # Split by period
    trading_weeks = get_trading_weeks(weekly_data)
    total_weeks = len(trading_weeks)
    recent_split_idx = max(0, total_weeks - RECENT_WEEKS)
    prior_split_idx = max(0, total_weeks - 2 * RECENT_WEEKS)
    
    recent_split_date = trading_weeks[recent_split_idx][0] if recent_split_idx < len(trading_weeks) else None
    prior_split_date = trading_weeks[prior_split_idx][0] if prior_split_idx < len(trading_weeks) else None
    
    historical_trades = [t for t in trades if t["entry_date"] is None or t["entry_date"] < prior_split_date]
    prior_trades = [t for t in trades if t["entry_date"] and prior_split_date <= t["entry_date"] < recent_split_date]
    recent_trades = [t for t in trades if t["entry_date"] and t["entry_date"] >= recent_split_date]
    
    # Analyze
    print("Analyzing periods...")
    all_stats = analyze_trades(trades, "All")
    historical_stats = analyze_trades(historical_trades, "Historical")
    prior_stats = analyze_trades(prior_trades, f"Prior {RECENT_WEEKS}w")
    recent_stats = analyze_trades(recent_trades, f"Recent {RECENT_WEEKS}w")
    slippage = detect_slippage(prior_stats, recent_stats)
    
    # Print results
    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    print()
    print(f"{'Metric':<25} {'Historical':>18} {'Prior 52w':>18} {'Recent 52w':>18}")
    print("-" * 100)
    print(f"{'Filled Trades':<25} {historical_stats['filled_trades']:>18} {prior_stats['filled_trades']:>18} {recent_stats['filled_trades']:>18}")
    print(f"{'Win Rate':<25} {historical_stats['win_rate']:>17.1f}% {prior_stats['win_rate']:>17.1f}% {recent_stats['win_rate']:>17.1f}%")
    print(f"{'Total P&L':<25} ${historical_stats['total_pnl']:>16,.2f} ${prior_stats['total_pnl']:>16,.2f} ${recent_stats['total_pnl']:>16,.2f}")
    print()
    print("Risk Metrics:")
    print(f"{'  Sharpe Ratio':<25} {historical_stats['sharpe_ratio']:>18.3f} {prior_stats['sharpe_ratio']:>18.3f} {recent_stats['sharpe_ratio']:>18.3f}")
    print(f"{'  Sortino Ratio':<25} {sortino_fmt(historical_stats['sortino_ratio']):>18} {sortino_fmt(prior_stats['sortino_ratio']):>18} {sortino_fmt(recent_stats['sortino_ratio']):>18}")
    print(f"{'  Top-5 Concentration':<25} {historical_stats['top5_concentration']:>17.1f}% {prior_stats['top5_concentration']:>17.1f}% {recent_stats['top5_concentration']:>17.1f}%")
    print(f"{'  Profit Factor':<25} {historical_stats['profit_factor']:>18.2f} {prior_stats['profit_factor']:>18.2f} {recent_stats['profit_factor']:>18.2f}")
    print()
    
    for period_name, stats in [("Historical", historical_stats), 
                                (f"Prior {RECENT_WEEKS}w", prior_stats),
                                (f"Recent {RECENT_WEEKS}w", recent_stats)]:
        print(f"Exit Reasons ({period_name}):")
        for reason, count in stats.get("exit_reasons", {}).items():
            cumul = stats.get("cumulative_pnl_by_reason", {}).get(reason, 0)
            print(f"  {reason:<15}: {count:>5} trades, CUMULATIVE ${cumul:>10,.2f}")
        print()
    
    if slippage["alarms"]:
        print("=" * 90)
        print("⚠️  SLIPPAGE ALARMS")
        print("=" * 90)
        for alarm in slippage["alarms"]:
            print(alarm)
        print()
    
    # Save outputs
    config = {
        "parameters": {
            "stop_loss_pcnt": STOP_LOSS_PCNT,
            "profit_exit_pcnt": PROFIT_EXIT_PCNT,
            "max_hold_weeks": MAX_HOLD_WEEKS,
            "initial_capital": INITIAL_CAPITAL,
            "max_active_trades": MAX_ACTIVE_TRADES,
            "bottom_percentile": BOTTOM_PERCENTILE,
            "min_loss_pcnt": MIN_LOSS_PCNT,
        },
        "regime": {
            "enabled": USE_REGIME,
            "symbol": REGIME_SYMBOL,
            "ma_period": REGIME_MA_PERIOD,
            "boost_direction": BOOST_DIRECTION,
            "boost_multiplier": BOOST_MULTIPLIER,
        },
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    pd.DataFrame(trades).to_csv(output_dir / "trades.csv", index=False)
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"all": all_stats, "historical": historical_stats, 
                   "prior": prior_stats, "recent": recent_stats, "slippage": slippage}, 
                  f, indent=2, default=str)
    
    dashboard_path = create_dashboard(trades, historical_stats, prior_stats, recent_stats, slippage, output_dir)
    
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    print(f"  Dashboard: {dashboard_path}")


if __name__ == "__main__":
    main()

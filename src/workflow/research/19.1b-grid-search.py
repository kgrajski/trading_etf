#!/usr/bin/env python3
"""
19.1b-grid-search.py

Grid Search with All Features + Period Stability Analysis.

Consolidates all bells & whistles from 19b, 19d, and adds:
- Regime awareness (SPY vs MA50)
- Configurable boost direction (bull/bear/none) - NOW IN GRID
- MIN_LOSS_PCNT threshold - NOW IN GRID
- Period stability analysis per config
- Risk-adjusted metrics: Sharpe, Sortino, Top-5 Concentration
- Flags for configs that degraded recently

Grid Parameters:
- SL: 4% to 16% in steps of 2%
- TP: 2% to 16% in steps of 2%
- MaxHold: 1 to 8 weeks
- BoostDirection: bull, bear, none
- MinLossPcnt: 2, 4, 6, 8, 10

Total: 7 × 8 × 8 × 3 × 5 = 6,720 combinations

Outputs:
- experiments/exp019_1b_grid/
  - grid_results.csv (includes all metrics)
  - best_configs.json
  - grid_heatmaps.html
  - stability_analysis.html
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ray
from ray import tune
from ray.tune import RunConfig

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp019_1b_grid"

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Grid parameters - EXPANDED
STOP_LOSS_RANGE = [4, 6, 8, 10, 12, 14, 16]  # % (7 values)
PROFIT_EXIT_RANGE = [2, 4, 6, 8, 10, 12, 14, 16]  # % (8 values)
MAX_HOLD_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]  # weeks (8 values)
BOOST_DIRECTION_RANGE = ["bull", "bear", "none"]  # (3 values) - NEW
MIN_LOSS_PCNT_RANGE = [2, 4, 6, 8, 10]  # % (5 values) - NEW

# Fixed parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10
BOTTOM_PERCENTILE = 0.05

# Regime parameters
USE_REGIME = True
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50
BOOST_MULTIPLIER = 1.10  # Fixed multiplier, direction is searched

# Period analysis
RECENT_WEEKS = 52
STABILITY_THRESHOLD = 5.0  # Flag if win rate drops > this %

# Risk-free rate for Sharpe (annualized, ~5% T-bill)
RISK_FREE_RATE = 0.05


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


@ray.remote
def load_shared_data():
    """Load all data once for sharing across workers."""
    print("🔄 Loading shared dataset...")
    
    symbols = get_target_symbols()
    if REGIME_SYMBOL not in symbols:
        symbols.append(REGIME_SYMBOL)
    
    daily_data = {}
    weekly_data = {}
    
    for symbol in symbols:
        daily_path = DAILY_DATA_DIR / f"{symbol}.csv"
        if daily_path.exists():
            df = pd.read_csv(daily_path, parse_dates=["date"])
            df = df.sort_values("date")
            daily_data[symbol] = df
        
        weekly_path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if weekly_path.exists():
            df = pd.read_csv(weekly_path, parse_dates=["week_start"])
            df = df.sort_values("week_start")
            weekly_data[symbol] = df
    
    # Calculate regime
    regime_by_date = {}
    if USE_REGIME and REGIME_SYMBOL in daily_data:
        df = daily_data[REGIME_SYMBOL].copy()
        df["ma"] = df["close"].rolling(window=REGIME_MA_PERIOD).mean()
        df["is_bull"] = df["close"] > df["ma"]
        
        for _, row in df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            regime_by_date[date_str] = {
                "is_bull": bool(row["is_bull"]) if pd.notna(row["is_bull"]) else True,
            }
    
    # Calculate trading weeks and period split dates
    all_weeks = set()
    for df in weekly_data.values():
        for _, row in df.iterrows():
            if pd.notna(row["week_start"]):
                all_weeks.add(row["week_start"])
    
    sorted_weeks = sorted(all_weeks)
    trading_weeks = []
    for i, ws in enumerate(sorted_weeks):
        if i + 1 < len(sorted_weeks):
            we = sorted_weeks[i + 1] - pd.Timedelta(days=1)
        else:
            we = ws + pd.Timedelta(days=6)
        trading_weeks.append((ws.strftime("%Y-%m-%d"), we.strftime("%Y-%m-%d")))
    
    # Period split dates
    total_weeks = len(trading_weeks)
    recent_idx = max(0, total_weeks - RECENT_WEEKS)
    prior_idx = max(0, total_weeks - 2 * RECENT_WEEKS)
    
    recent_split = trading_weeks[recent_idx][0] if recent_idx < len(trading_weeks) else None
    prior_split = trading_weeks[prior_idx][0] if prior_idx < len(trading_weeks) else None
    
    print(f"✅ Loaded {len(daily_data)} symbols, {len(trading_weeks)} weeks")
    print(f"   Period splits: prior={prior_split}, recent={recent_split}")
    
    return {
        "daily": daily_data,
        "weekly": weekly_data,
        "regime_by_date": regime_by_date,
        "trading_weeks": trading_weeks,
        "prior_split": prior_split,
        "recent_split": recent_split,
    }


# =============================================================================
# Risk Metrics
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
# Core Backtest Logic
# =============================================================================

def get_position_multiplier(is_bull_week: bool, boost_direction: str) -> float:
    """Get position size multiplier based on regime and boost config."""
    if not USE_REGIME or boost_direction == "none":
        return 1.0
    
    if boost_direction == "bear" and not is_bull_week:
        return BOOST_MULTIPLIER
    elif boost_direction == "bull" and is_bull_week:
        return BOOST_MULTIPLIER
    else:
        return 1.0


def run_backtest_with_data(
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    regime_by_date: Dict[str, Dict],
    trading_weeks: List[Tuple[str, str]],
    prior_split: str,
    recent_split: str,
    stop_loss_pcnt: float,
    profit_exit_pcnt: float,
    max_hold_weeks: int,
    boost_direction: str,
    min_loss_pcnt: float,
) -> Dict[str, Any]:
    """
    Run backtest and return comprehensive metrics.
    """
    # Build trading calendar
    all_trading_days = set()
    for df in daily_data.values():
        for d in df["date"]:
            all_trading_days.add(d.strftime("%Y-%m-%d"))
    all_trading_days = sorted(all_trading_days)
    
    # State
    trades = []
    next_trade_id = 1
    current_capital = INITIAL_CAPITAL
    committed_capital = 0.0
    pending_orders = []
    
    bull_weeks = 0
    bear_weeks = 0
    
    # Track weekly P&L for Sharpe/Sortino
    weekly_pnl = {}  # week_start -> P&L realized that week
    
    # Process each week
    for week_idx, (week_start, week_end) in enumerate(trading_weeks):
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        trading_days = [d for d in all_trading_days 
                        if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            continue
        
        first_day = trading_days[0]
        last_day = trading_days[-1]
        
        regime_info = regime_by_date.get(first_day, {"is_bull": True})
        is_bull_week = regime_info.get("is_bull", True) if USE_REGIME else True
        
        if is_bull_week:
            bull_weeks += 1
        else:
            bear_weeks += 1
        
        position_multiplier = get_position_multiplier(is_bull_week, boost_direction)
        
        # Initialize weekly P&L tracking
        if week_start not in weekly_pnl:
            weekly_pnl[week_start] = 0.0
        
        # 1. Fill pending orders
        for order in pending_orders:
            symbol = order["symbol"]
            limit_price = order["limit_price"]
            
            if symbol not in daily_data:
                continue
            
            day_df = daily_data[symbol]
            day_row = day_df[day_df["date"] == pd.Timestamp(first_day)]
            
            if len(day_row) == 0:
                continue
            
            row = day_row.iloc[0]
            
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
                        "entry_date": first_day,
                        "entry_week_idx": week_idx,
                        "entry_week_start": week_start,
                        "entry_price": fill_price,
                        "shares": shares,
                        "position_value": position_value,
                        "is_bull_entry": is_bull_week,
                        "is_open": True,
                        "exit_date": None,
                        "exit_week_start": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "pnl_dollars": 0.0,
                    })
                    next_trade_id += 1
        
        pending_orders = []
        
        # 2. Check exits
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
                
                # Stop loss
                if row["low"] <= stop_price:
                    exit_price = min(stop_price, row["open"])
                    trade["exit_date"] = day
                    trade["exit_week_start"] = week_start
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "stop_loss"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    weekly_pnl[week_start] += trade["pnl_dollars"]
                    break
                
                # Profit target
                if row["high"] >= profit_price:
                    exit_price = max(profit_price, row["open"])
                    trade["exit_date"] = day
                    trade["exit_week_start"] = week_start
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "profit_exit"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    weekly_pnl[week_start] += trade["pnl_dollars"]
                    break
                
                # Max hold (Friday exit)
                is_last_day = (day == last_day)
                is_max_hold_week = (week_idx >= max_hold_exit_week)
                
                if is_last_day and is_max_hold_week:
                    exit_price = row["open"]
                    trade["exit_date"] = day
                    trade["exit_week_start"] = week_start
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "max_hold"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    weekly_pnl[week_start] += trade["pnl_dollars"]
                    break
        
        # 3. Generate candidates
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
                "pct_return": (np.exp(row["log_return"]) - 1) * 100,
                "close": row["close"],
            })
        
        if not returns_data:
            continue
        
        returns_df = pd.DataFrame(returns_data)
        n_bottom = max(1, int(len(returns_df) * BOTTOM_PERCENTILE))
        bottom_df = returns_df.nsmallest(n_bottom, "pct_return")
        qualified = bottom_df[bottom_df["pct_return"] <= -min_loss_pcnt]
        
        active_count = sum(1 for t in trades if t["is_open"])
        available_slots = MAX_ACTIVE_TRADES - active_count
        
        for _, candidate in qualified.head(available_slots).iterrows():
            pending_orders.append({
                "symbol": candidate["symbol"],
                "limit_price": candidate["close"],
            })
    
    # Close remaining
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
    
    # Calculate metrics
    closed_trades = [t for t in trades if t["exit_date"] is not None]
    
    if not closed_trades:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "sharpe_overall": 0, "sortino_overall": 0, "top5_concentration": 100,
        }
    
    # Overall metrics
    total_trades = len(closed_trades)
    winning = sum(1 for t in closed_trades if t["pnl_dollars"] > 0)
    total_pnl = sum(t["pnl_dollars"] for t in closed_trades)
    pnl_list = [t["pnl_dollars"] for t in closed_trades]
    
    # Max drawdown
    sorted_trades = sorted(closed_trades, key=lambda t: t["exit_date"])
    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0
    for t in sorted_trades:
        capital += t["pnl_dollars"]
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    # Risk metrics - Overall
    weekly_pnl_list = list(weekly_pnl.values())
    sharpe_overall = compute_sharpe_ratio(weekly_pnl_list)
    sortino_overall = compute_sortino_ratio(weekly_pnl_list)
    top5_concentration = compute_top5_concentration(pnl_list)
    profit_factor = compute_profit_factor(pnl_list)
    
    # Period splits
    prior_trades = [t for t in closed_trades if t["entry_date"] and prior_split and 
                    prior_split <= t["entry_date"] < recent_split]
    recent_trades = [t for t in closed_trades if t["entry_date"] and recent_split and
                     t["entry_date"] >= recent_split]
    
    # Period metrics
    prior_win_rate = (sum(1 for t in prior_trades if t["pnl_dollars"] > 0) / len(prior_trades) * 100) if prior_trades else 0
    recent_win_rate = (sum(1 for t in recent_trades if t["pnl_dollars"] > 0) / len(recent_trades) * 100) if recent_trades else 0
    
    prior_pnl = sum(t["pnl_dollars"] for t in prior_trades)
    recent_pnl = sum(t["pnl_dollars"] for t in recent_trades)
    
    # Risk metrics - Recent
    recent_pnl_list = [t["pnl_dollars"] for t in recent_trades]
    recent_weekly_pnl = {k: v for k, v in weekly_pnl.items() if recent_split and k >= recent_split}
    sharpe_recent = compute_sharpe_ratio(list(recent_weekly_pnl.values()))
    sortino_recent = compute_sortino_ratio(list(recent_weekly_pnl.values()))
    top5_concentration_recent = compute_top5_concentration(recent_pnl_list)
    profit_factor_recent = compute_profit_factor(recent_pnl_list) if recent_pnl_list else 0
    
    # Max hold P&L by period
    prior_max_hold_pnl = sum(t["pnl_dollars"] for t in prior_trades if t["exit_reason"] == "max_hold")
    recent_max_hold_pnl = sum(t["pnl_dollars"] for t in recent_trades if t["exit_reason"] == "max_hold")
    
    # Stability flag
    win_rate_delta = recent_win_rate - prior_win_rate
    is_degraded = win_rate_delta < -STABILITY_THRESHOLD or recent_max_hold_pnl < 0
    
    # Regime metrics
    bull_trades_list = [t for t in closed_trades if t.get("is_bull_entry")]
    bear_trades_list = [t for t in closed_trades if not t.get("is_bull_entry")]
    bull_pnl = sum(t["pnl_dollars"] for t in bull_trades_list)
    bear_pnl = sum(t["pnl_dollars"] for t in bear_trades_list)
    
    return {
        # Basic metrics
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": total_trades - winning,
        "win_rate": winning / total_trades * 100,
        "total_pnl": total_pnl,
        "total_return_pcnt": total_pnl / INITIAL_CAPITAL * 100,
        "max_drawdown_pcnt": max_dd,
        
        # Risk-adjusted metrics - Overall
        "sharpe_overall": sharpe_overall,
        "sortino_overall": sortino_overall,
        "top5_concentration": top5_concentration,
        "profit_factor": profit_factor,
        
        # Risk-adjusted metrics - Recent
        "sharpe_recent": sharpe_recent,
        "sortino_recent": sortino_recent,
        "top5_concentration_recent": top5_concentration_recent,
        "profit_factor_recent": profit_factor_recent,
        
        # Regime
        "bull_weeks": bull_weeks,
        "bear_weeks": bear_weeks,
        "bull_trades": len(bull_trades_list),
        "bear_trades": len(bear_trades_list),
        "bull_pnl": bull_pnl,
        "bear_pnl": bear_pnl,
        
        # Period stability
        "prior_trades": len(prior_trades),
        "recent_trades": len(recent_trades),
        "prior_win_rate": prior_win_rate,
        "recent_win_rate": recent_win_rate,
        "win_rate_delta": win_rate_delta,
        "prior_pnl": prior_pnl,
        "recent_pnl": recent_pnl,
        "prior_max_hold_pnl": prior_max_hold_pnl,
        "recent_max_hold_pnl": recent_max_hold_pnl,
        "is_degraded": is_degraded,
    }


def ray_tune_trainable(config: Dict[str, Any]) -> None:
    """Ray Tune trainable function."""
    shared_data = ray.get(config["shared_data_ref"])
    
    result = run_backtest_with_data(
        daily_data=shared_data["daily"],
        weekly_data=shared_data["weekly"],
        regime_by_date=shared_data["regime_by_date"],
        trading_weeks=shared_data["trading_weeks"],
        prior_split=shared_data["prior_split"],
        recent_split=shared_data["recent_split"],
        stop_loss_pcnt=config["stop_loss_pcnt"],
        profit_exit_pcnt=config["profit_exit_pcnt"],
        max_hold_weeks=config["max_hold_weeks"],
        boost_direction=config["boost_direction"],
        min_loss_pcnt=config["min_loss_pcnt"],
    )
    
    tune.report(metrics=result)


# =============================================================================
# Visualization
# =============================================================================

def create_heatmap_dashboard(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create heatmap dashboard for best configuration slice."""
    # Find best config
    best_row = results_df.loc[results_df["sortino_overall"].idxmax()]
    best_max_hold = int(best_row["max_hold_weeks"])
    best_boost = best_row["boost_direction"]
    best_min_loss = best_row["min_loss_pcnt"]
    
    # Slice for best non-grid params
    df_slice = results_df[
        (results_df["max_hold_weeks"] == best_max_hold) &
        (results_df["boost_direction"] == best_boost) &
        (results_df["min_loss_pcnt"] == best_min_loss)
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Sortino Ratio",
            f"Total Return %",
            f"Top-5 Concentration %",
            f"Recent max_hold P&L",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )
    
    for metric, colorscale, row, col, reverse in [
        ("sortino_overall", "RdYlGn", 1, 1, False),
        ("total_return_pcnt", "RdYlGn", 1, 2, False),
        ("top5_concentration", "RdYlGn", 2, 1, True),  # Lower is better
        ("recent_max_hold_pnl", "RdYlGn", 2, 2, False),
    ]:
        pivot = df_slice.pivot(
            index="stop_loss_pcnt",
            columns="profit_exit_pcnt",
            values=metric,
        )
        
        z_values = pivot.values
        if reverse:
            z_values = -z_values  # Flip for colorscale
        
        fig.add_trace(
            go.Heatmap(
                z=z_values if not reverse else pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=colorscale,
                reversescale=reverse,
                showscale=True,
                hovertemplate=f"SL: %{{y}}%<br>TP: %{{x}}%<br>{metric}: %{{z:.2f}}<extra></extra>",
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        title=f"<b>Grid Search Results: {EXPERIMENT_NAME}</b><br>"
              f"<sub>Slice: MaxHold={best_max_hold}w | Boost={best_boost} | MinLoss={best_min_loss}%</sub>",
        height=800,
    )
    
    for i in range(1, 5):
        fig.update_xaxes(title_text="Profit Exit %", row=(i-1)//2+1, col=(i-1)%2+1)
        fig.update_yaxes(title_text="Stop Loss %", row=(i-1)//2+1, col=(i-1)%2+1)
    
    output_path = output_dir / "grid_heatmaps.html"
    fig.write_html(str(output_path))
    return output_path


def create_stability_dashboard(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create stability analysis dashboard."""
    # Top configs by Sortino
    top_by_sortino = results_df.nlargest(10, "sortino_overall")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Top 10 Configs: Sortino Ratio (Overall vs Recent)",
            "Top 10 Configs: Top-5 Concentration % (lower = better)",
            "Top 10 Configs: Recent max_hold P&L",
        ),
        vertical_spacing=0.12,
    )
    
    # Labels
    labels = [f"SL{int(r['stop_loss_pcnt'])}_TP{int(r['profit_exit_pcnt'])}_H{int(r['max_hold_weeks'])}_{r['boost_direction'][:1].upper()}_L{int(r['min_loss_pcnt'])}" 
              for _, r in top_by_sortino.iterrows()]
    
    # Sortino comparison
    fig.add_trace(
        go.Bar(name="Overall", x=labels, y=top_by_sortino["sortino_overall"].tolist(),
               marker_color="#4CAF50"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="Recent", x=labels, y=top_by_sortino["sortino_recent"].tolist(),
               marker_color="#2196F3"),
        row=1, col=1,
    )
    
    # Top-5 concentration
    fig.add_trace(
        go.Bar(name="Top-5 Concentration", x=labels, 
               y=top_by_sortino["top5_concentration"].tolist(),
               marker_color="#FF9800", showlegend=False),
        row=2, col=1,
    )
    
    # Recent max_hold P&L
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in top_by_sortino["recent_max_hold_pnl"]]
    fig.add_trace(
        go.Bar(name="Recent max_hold P&L", x=labels, 
               y=top_by_sortino["recent_max_hold_pnl"].tolist(),
               marker_color=colors, showlegend=False),
        row=3, col=1,
    )
    
    fig.update_layout(
        title="<b>Period Stability Analysis (Top 10 by Sortino)</b>",
        height=900,
        barmode="group",
    )
    
    output_path = output_dir / "stability_analysis.html"
    fig.write_html(str(output_path))
    return output_path


# =============================================================================
# Main
# =============================================================================

@workflow_script("19.1b-grid-search")
def main():
    """Run grid search with expanded parameters and risk metrics."""
    
    total_combos = (len(STOP_LOSS_RANGE) * len(PROFIT_EXIT_RANGE) * 
                    len(MAX_HOLD_RANGE) * len(BOOST_DIRECTION_RANGE) * 
                    len(MIN_LOSS_PCNT_RANGE))
    
    print("=" * 70)
    print("19.1b GRID SEARCH (EXPANDED)")
    print("=" * 70)
    print()
    print(f"Grid Parameters:")
    print(f"  Stop Loss:      {STOP_LOSS_RANGE} ({len(STOP_LOSS_RANGE)} values)")
    print(f"  Profit Exit:    {PROFIT_EXIT_RANGE} ({len(PROFIT_EXIT_RANGE)} values)")
    print(f"  Max Hold:       {MAX_HOLD_RANGE} ({len(MAX_HOLD_RANGE)} values)")
    print(f"  Boost Dir:      {BOOST_DIRECTION_RANGE} ({len(BOOST_DIRECTION_RANGE)} values)")
    print(f"  Min Loss:       {MIN_LOSS_PCNT_RANGE} ({len(MIN_LOSS_PCNT_RANGE)} values)")
    print()
    print(f"Total combinations: {total_combos:,}")
    print(f"Regime: {REGIME_SYMBOL} MA{REGIME_MA_PERIOD}")
    print(f"Boost Multiplier: {BOOST_MULTIPLIER}x")
    print()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
    
    try:
        # Load data once
        print("Loading shared data...")
        data_ref = ray.get(load_shared_data.remote())
        shared_data_ref = ray.put(data_ref)
        print()
        
        # Build search space
        search_space = {
            "shared_data_ref": shared_data_ref,
            "stop_loss_pcnt": tune.grid_search(STOP_LOSS_RANGE),
            "profit_exit_pcnt": tune.grid_search(PROFIT_EXIT_RANGE),
            "max_hold_weeks": tune.grid_search(MAX_HOLD_RANGE),
            "boost_direction": tune.grid_search(BOOST_DIRECTION_RANGE),
            "min_loss_pcnt": tune.grid_search(MIN_LOSS_PCNT_RANGE),
        }
        
        # Run grid search
        print("Running grid search...")
        tuner = tune.Tuner(
            ray_tune_trainable,
            param_space=search_space,
            run_config=RunConfig(
                name=EXPERIMENT_NAME,
                verbose=1,
            ),
        )
        
        results = tuner.fit()
        
        # Collect results
        print("\nCollecting results...")
        rows = []
        for result in results:
            config = result.config
            metrics = result.metrics
            if metrics:
                rows.append({
                    "stop_loss_pcnt": config["stop_loss_pcnt"],
                    "profit_exit_pcnt": config["profit_exit_pcnt"],
                    "max_hold_weeks": config["max_hold_weeks"],
                    "boost_direction": config["boost_direction"],
                    "min_loss_pcnt": config["min_loss_pcnt"],
                    **metrics,
                })
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
        
        # Find best configs by different criteria
        best_return = results_df.loc[results_df["total_return_pcnt"].idxmax()]
        best_sortino = results_df.loc[results_df["sortino_overall"].idxmax()]
        best_sharpe = results_df.loc[results_df["sharpe_overall"].idxmax()]
        
        # Best stable (not degraded, highest sortino)
        stable_df = results_df[~results_df["is_degraded"]]
        best_stable = stable_df.nlargest(1, "sortino_overall").iloc[0] if len(stable_df) > 0 else None
        
        # Best low-concentration (diversified gains)
        low_conc = results_df[results_df["top5_concentration"] < 50]
        best_diversified = low_conc.nlargest(1, "sortino_overall").iloc[0] if len(low_conc) > 0 else None
        
        best_configs = {
            "generated_at": datetime.now().isoformat(),
            "total_combinations": total_combos,
            "regime": {"symbol": REGIME_SYMBOL, "ma_period": REGIME_MA_PERIOD},
            "boost_multiplier": BOOST_MULTIPLIER,
            "best_by_return": best_return.to_dict(),
            "best_by_sortino": best_sortino.to_dict(),
            "best_by_sharpe": best_sharpe.to_dict(),
            "best_stable": best_stable.to_dict() if best_stable is not None else None,
            "best_diversified": best_diversified.to_dict() if best_diversified is not None else None,
            "summary": {
                "degraded_count": int(results_df["is_degraded"].sum()),
                "stable_count": int((~results_df["is_degraded"]).sum()),
                "low_concentration_count": int((results_df["top5_concentration"] < 50).sum()),
            },
        }
        
        with open(OUTPUT_DIR / "best_configs.json", "w") as f:
            json.dump(best_configs, f, indent=2, default=str)
        
        # Create dashboards
        print("Creating dashboards...")
        heatmap_path = create_heatmap_dashboard(results_df, OUTPUT_DIR)
        stability_path = create_stability_dashboard(results_df, OUTPUT_DIR)
        
        # Print summary
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print()
        
        def print_config(name: str, row):
            print(f"{name}:")
            print(f"  Params: SL={row['stop_loss_pcnt']}%, TP={row['profit_exit_pcnt']}%, "
                  f"MaxHold={row['max_hold_weeks']}w, Boost={row['boost_direction']}, MinLoss={row['min_loss_pcnt']}%")
            print(f"  Return: {row['total_return_pcnt']:.1f}%, Sortino: {row['sortino_overall']:.2f}, "
                  f"Sharpe: {row['sharpe_overall']:.2f}")
            print(f"  Top-5 Concentration: {row['top5_concentration']:.1f}%, "
                  f"Recent max_hold P&L: ${row['recent_max_hold_pnl']:,.2f}")
            print(f"  Degraded: {'YES ⚠️' if row['is_degraded'] else 'No ✅'}")
            print()
        
        print_config("Best by Sortino", best_sortino)
        print_config("Best by Return", best_return)
        if best_stable is not None:
            print_config("Best Stable (not degraded)", best_stable)
        if best_diversified is not None:
            print_config("Best Diversified (top5 < 50%)", best_diversified)
        
        print(f"Stability: {best_configs['summary']['stable_count']} stable, "
              f"{best_configs['summary']['degraded_count']} degraded")
        print(f"Low concentration (< 50%): {best_configs['summary']['low_concentration_count']}")
        print()
        print(f"Outputs:")
        print(f"  Results: {OUTPUT_DIR / 'grid_results.csv'}")
        print(f"  Best Configs: {OUTPUT_DIR / 'best_configs.json'}")
        print(f"  Heatmaps: {heatmap_path}")
        print(f"  Stability: {stability_path}")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

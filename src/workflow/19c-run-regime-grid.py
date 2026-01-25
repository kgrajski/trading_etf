#!/usr/bin/env python3
"""
19c-run-regime-grid.py

Systematic grid search with REGIME-DEPENDENT capital allocation.

Key difference from 19b:
- Detects market regime using SPY 50-day moving average
- Bull market (SPY > 50-day MA): Increase position size by 10%
- Bear/neutral market: Standard position sizing

Grid:
- SL: 4% to 16% in steps of 2%
- TP: 2% to 16% in steps of 2%
- MaxHold: 1 to 8 weeks in steps of 1

Uses Ray for parallel execution with shared data (loaded once, no copies).

Outputs:
- experiments/exp019_grid_regime/grid_results.csv
- experiments/exp019_grid_regime/grid_heatmaps.html
- experiments/exp019_grid_regime/best_configs.json
- experiments/exp019_grid_regime/regime_analysis.html
"""

import json
import logging
import os
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ray
from ray import tune
from ray.tune import RunConfig

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp019_grid_regime"  # Regime-dependent capital allocation

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Grid parameters
STOP_LOSS_RANGE = [4, 6, 8, 10, 12, 14, 16]  # %
PROFIT_EXIT_RANGE = [2, 4, 6, 8, 10, 12, 14, 16]  # %
MAX_HOLD_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]  # weeks

# Fixed parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10
BOTTOM_PERCENTILE = 0.05
MIN_LOSS_PCNT = 2.0

# Regime detection parameters
REGIME_SYMBOL = "SPY"  # Use SPY as market proxy
REGIME_MA_PERIOD = 50  # 50-day moving average
BULL_MARKET_MULTIPLIER = 1.10  # 10% larger positions in bull markets


# =============================================================================
# Data Loading (runs once, shared across all workers)
# =============================================================================

def get_target_symbols() -> List[str]:
    """Get list of target ETF symbols."""
    filtered_etfs_path = METADATA_DIR / "filtered_etfs.json"
    
    if not filtered_etfs_path.exists():
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    
    with open(filtered_etfs_path) as f:
        data = json.load(f)
    
    return [etf["symbol"] for etf in data.get("etfs", [])]


@ray.remote
def load_shared_data():
    """
    Load all daily and weekly data once for sharing across workers.
    
    Returns:
        Dict with 'daily' and 'weekly' DataFrames by symbol, plus regime data
    """
    print("🔄 Loading shared dataset...")
    
    symbols = get_target_symbols()
    
    daily_data = {}
    weekly_data = {}
    
    # Load daily data
    for symbol in symbols:
        daily_path = DAILY_DATA_DIR / f"{symbol}.csv"
        if daily_path.exists():
            df = pd.read_csv(daily_path, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)
            daily_data[symbol] = df
    
    # Load weekly data
    for symbol in symbols:
        weekly_path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if weekly_path.exists():
            df = pd.read_csv(weekly_path, parse_dates=["week_start", "week_end"])
            df = df.sort_values("week_start").reset_index(drop=True)
            weekly_data[symbol] = df
    
    # Calculate regime data using SPY
    regime_by_date = {}
    if REGIME_SYMBOL in daily_data:
        spy_df = daily_data[REGIME_SYMBOL].copy()
        spy_df["ma_50"] = spy_df["close"].rolling(window=REGIME_MA_PERIOD).mean()
        spy_df["is_bull"] = spy_df["close"] > spy_df["ma_50"]
        
        for _, row in spy_df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            regime_by_date[date_str] = {
                "is_bull": bool(row["is_bull"]) if pd.notna(row["is_bull"]) else False,
                "spy_close": row["close"],
                "spy_ma50": row["ma_50"] if pd.notna(row["ma_50"]) else None,
            }
        print(f"📊 Calculated regime for {len(regime_by_date)} trading days")
    else:
        print(f"⚠️ {REGIME_SYMBOL} not found - defaulting to neutral regime")
    
    print(f"✅ Loaded {len(daily_data)} daily, {len(weekly_data)} weekly files")
    
    return {
        "daily": daily_data,
        "weekly": weekly_data,
        "symbols": symbols,
        "regime_by_date": regime_by_date,
    }


# =============================================================================
# Backtest Logic (runs in each worker)
# =============================================================================

def run_backtest_with_data(
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    regime_by_date: Dict[str, Dict],
    stop_loss_pcnt: float,
    profit_exit_pcnt: float,
    max_hold_weeks: int,
) -> Dict[str, Any]:
    """
    Run a single backtest with pre-loaded data and regime-dependent sizing.
    
    In bull markets (SPY > 50-day MA), position sizes are increased by 10%.
    This may result in fewer total trades due to faster capital deployment.
    """
    # Get trading weeks
    all_weeks = set()
    for symbol, df in weekly_data.items():
        for _, row in df.iterrows():
            all_weeks.add((
                row["week_start"].strftime("%Y-%m-%d"),
                row["week_end"].strftime("%Y-%m-%d")
            ))
    trading_weeks = sorted(all_weeks, key=lambda x: x[0])
    
    if not trading_weeks:
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "total_return_pcnt": 0, "max_drawdown_pcnt": 0,
                "bull_weeks": 0, "bear_weeks": 0, "bull_trades": 0, "bear_trades": 0}
    
    # Get all trading days
    sample_daily = next(iter(daily_data.values()))
    all_trading_days = sample_daily["date"].dt.strftime("%Y-%m-%d").tolist()
    
    # State tracking
    trades = []
    next_trade_id = 1
    current_capital = INITIAL_CAPITAL
    
    # Regime tracking
    bull_weeks = 0
    bear_weeks = 0
    bull_trades = 0
    bear_trades = 0
    committed_capital = 0.0
    pending_orders = []
    
    # Process each week
    for week_idx, (week_start, week_end) in enumerate(trading_weeks):
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        # Get trading days for this week
        trading_days = [d for d in all_trading_days if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            continue
        
        first_day = trading_days[0]
        
        # Determine regime for this week based on first trading day
        regime_info = regime_by_date.get(first_day, {"is_bull": False})
        is_bull_week = regime_info.get("is_bull", False)
        
        if is_bull_week:
            bull_weeks += 1
            position_multiplier = BULL_MARKET_MULTIPLIER  # 1.10x in bull markets
        else:
            bear_weeks += 1
            position_multiplier = 1.0  # Standard sizing in bear/neutral
        
        # 1. Try to fill pending orders from last week
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
            
            # Check if limit hit
            if row["low"] <= limit_price:
                fill_price = min(limit_price, row["open"])
                
                # Calculate position size with regime adjustment
                available = current_capital - committed_capital
                if available <= 0:
                    continue
                
                # Apply regime multiplier to position size
                base_target_size = INITIAL_CAPITAL / MAX_ACTIVE_TRADES
                target_size = base_target_size * position_multiplier
                position_budget = min(available, target_size)
                shares = int(position_budget / fill_price)
                
                if shares > 0:
                    position_value = shares * fill_price
                    committed_capital += position_value
                    
                    # Track regime for this trade
                    if is_bull_week:
                        bull_trades += 1
                    else:
                        bear_trades += 1
                    
                    trades.append({
                        "trade_id": next_trade_id,
                        "symbol": symbol,
                        "entry_date": first_day,
                        "entry_week_idx": week_idx,  # Track entry week for Friday exit
                        "entry_price": fill_price,
                        "shares": shares,
                        "position_value": position_value,
                        "is_bull_entry": is_bull_week,  # Track regime at entry
                        "is_open": True,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "pnl_dollars": 0,
                        "pnl_pcnt": 0,
                    })
                    next_trade_id += 1
        
        pending_orders = []
        
        # 2. Check exits for open trades
        # Determine last trading day of this week for Friday exit logic
        last_day_of_week = trading_days[-1] if trading_days else None
        
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
            
            # Calculate which week triggers max_hold exit (Friday of that week)
            # max_hold_weeks=1 means exit Friday of entry week
            # max_hold_weeks=2 means exit Friday of week after entry, etc.
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
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Check max hold - exit on last day (Friday) of the max_hold week
                # Use OPEN price (conservative: we decide overnight to exit, place MOO order)
                is_last_day = (day == last_day_of_week)
                is_max_hold_week = (week_idx >= max_hold_exit_week)
                
                if is_last_day and is_max_hold_week:
                    exit_price = row["open"]  # Conservative: exit at open, not close
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "max_hold"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
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
        
        # Create pending orders
        active_count = sum(1 for t in trades if t["is_open"])
        available_slots = MAX_ACTIVE_TRADES - active_count
        
        for _, candidate in qualified.head(available_slots).iterrows():
            pending_orders.append({
                "symbol": candidate["symbol"],
                "limit_price": candidate["close"],
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
    
    # Calculate metrics
    closed_trades = [t for t in trades if t["exit_date"] is not None]
    total_trades = len(closed_trades)
    
    if total_trades == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_return_pcnt": 0,
            "max_drawdown_pcnt": 0,
            "bull_weeks": bull_weeks,
            "bear_weeks": bear_weeks,
            "bull_trades": bull_trades,
            "bear_trades": bear_trades,
        }
    
    winning_trades = sum(1 for t in closed_trades if t["pnl_dollars"] > 0)
    losing_trades = sum(1 for t in closed_trades if t["pnl_dollars"] <= 0)
    win_rate = (winning_trades / total_trades) * 100
    total_pnl = sum(t["pnl_dollars"] for t in closed_trades)
    total_return_pcnt = (total_pnl / INITIAL_CAPITAL) * 100
    
    # Calculate max drawdown
    sorted_trades = sorted(closed_trades, key=lambda t: t["exit_date"])
    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0
    
    for trade in sorted_trades:
        capital += trade["pnl_dollars"]
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    # Calculate regime-specific metrics
    bull_trade_pnl = sum(t["pnl_dollars"] for t in closed_trades if t.get("is_bull_entry", False))
    bear_trade_pnl = sum(t["pnl_dollars"] for t in closed_trades if not t.get("is_bull_entry", False))
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_return_pcnt": total_return_pcnt,
        "max_drawdown_pcnt": max_dd,
        "bull_weeks": bull_weeks,
        "bear_weeks": bear_weeks,
        "bull_trades": bull_trades,
        "bear_trades": bear_trades,
        "bull_trade_pnl": bull_trade_pnl,
        "bear_trade_pnl": bear_trade_pnl,
    }


def ray_tune_trainable(config: Dict[str, Any]) -> None:
    """
    Ray Tune trainable function.
    
    Receives config with shared data reference and hyperparameters.
    """
    # Get shared data from Ray Object Store (no copy!)
    shared_data = ray.get(config["shared_data_ref"])
    
    # Run backtest with regime data
    result = run_backtest_with_data(
        daily_data=shared_data["daily"],
        weekly_data=shared_data["weekly"],
        regime_by_date=shared_data.get("regime_by_date", {}),
        stop_loss_pcnt=config["stop_loss_pcnt"],
        profit_exit_pcnt=config["profit_exit_pcnt"],
        max_hold_weeks=config["max_hold_weeks"],
    )
    
    # Report metrics to Ray Tune (new API uses metrics dict)
    tune.report(
        metrics={
            "total_trades": result["total_trades"],
            "winning_trades": result["winning_trades"],
            "losing_trades": result["losing_trades"],
            "win_rate": result["win_rate"],
            "total_pnl": result["total_pnl"],
            "total_return_pcnt": result["total_return_pcnt"],
            "max_drawdown_pcnt": result["max_drawdown_pcnt"],
            "bull_weeks": result["bull_weeks"],
            "bear_weeks": result["bear_weeks"],
            "bull_trades": result["bull_trades"],
            "bear_trades": result["bear_trades"],
            "bull_trade_pnl": result["bull_trade_pnl"],
            "bear_trade_pnl": result["bear_trade_pnl"],
        }
    )


# =============================================================================
# Visualization
# =============================================================================

def create_heatmap_dashboard(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create interactive heatmap dashboard."""
    
    # Find best max_hold for the heatmaps
    best_row = results_df.loc[results_df["total_return_pcnt"].idxmax()]
    best_max_hold = int(best_row["max_hold_weeks"])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Return % (MaxHold={best_max_hold}w)",
            f"Win Rate % (MaxHold={best_max_hold}w)",
            f"Max Drawdown % (MaxHold={best_max_hold}w)",
            f"Total Trades (MaxHold={best_max_hold}w)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )
    
    df_slice = results_df[results_df["max_hold_weeks"] == best_max_hold]
    
    for metric, colorscale, row, col in [
        ("total_return_pcnt", "RdYlGn", 1, 1),
        ("win_rate", "Blues", 1, 2),
        ("max_drawdown_pcnt", "Reds_r", 2, 1),
        ("total_trades", "Purples", 2, 2),
    ]:
        pivot = df_slice.pivot(
            index="stop_loss_pcnt",
            columns="profit_exit_pcnt",
            values=metric
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(len=0.4, y=0.8 if row == 1 else 0.2),
                hovertemplate=(
                    f"SL: %{{y}}%<br>"
                    f"TP: %{{x}}%<br>"
                    f"{metric}: %{{z:.1f}}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
    
    best_sl = best_row["stop_loss_pcnt"]
    best_tp = best_row["profit_exit_pcnt"]
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Grid Search Results: {len(results_df)} Configurations</b><br>"
                f"<sup>Best: SL={best_sl}%, TP={best_tp}%, MaxHold={best_max_hold}w → "
                f"Return={best_row['total_return_pcnt']:.1f}%, "
                f"WinRate={best_row['win_rate']:.1f}%, "
                f"MaxDD={best_row['max_drawdown_pcnt']:.1f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=900,
        template="plotly_white",
    )
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Profit Target (%)", row=i, col=j)
            fig.update_yaxes(title_text="Stop Loss (%)", row=i, col=j)
    
    output_path = output_dir / "grid_heatmaps.html"
    fig.write_html(output_path)
    
    return output_path


def create_max_hold_comparison(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create comparison across max hold values."""
    
    agg = results_df.groupby("max_hold_weeks").agg({
        "total_return_pcnt": ["mean", "max", "std"],
        "win_rate": "mean",
        "max_drawdown_pcnt": "mean",
        "total_trades": "mean",
    }).round(2)
    
    agg.columns = ["_".join(col) for col in agg.columns]
    agg = agg.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg Return by Max Hold",
            "Best Return by Max Hold",
            "Avg Win Rate by Max Hold",
            "Avg Max Drawdown by Max Hold",
        ),
    )
    
    fig.add_trace(
        go.Bar(x=agg["max_hold_weeks"], y=agg["total_return_pcnt_mean"],
               marker_color="#2E86AB", name="Avg Return"),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Bar(x=agg["max_hold_weeks"], y=agg["total_return_pcnt_max"],
               marker_color="#28A745", name="Best Return"),
        row=1, col=2,
    )
    
    fig.add_trace(
        go.Bar(x=agg["max_hold_weeks"], y=agg["win_rate_mean"],
               marker_color="#6F42C1", name="Avg Win Rate"),
        row=2, col=1,
    )
    
    fig.add_trace(
        go.Bar(x=agg["max_hold_weeks"], y=agg["max_drawdown_pcnt_mean"],
               marker_color="#DC3545", name="Avg Max DD"),
        row=2, col=2,
    )
    
    fig.update_layout(
        title="Performance by Max Hold Period",
        height=600,
        showlegend=False,
        template="plotly_white",
    )
    
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Max Hold (weeks)", row=i, col=j)
    
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Return %", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate %", row=2, col=1)
    fig.update_yaxes(title_text="Max DD %", row=2, col=2)
    
    output_path = output_dir / "grid_max_hold_comparison.html"
    fig.write_html(output_path)
    
    return output_path


def create_regime_analysis(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create regime-specific performance analysis."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Bull vs Bear Trade Distribution",
            "P&L by Regime",
            "Bull Trade % by Configuration",
            "Return Contribution by Regime",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]],
    )
    
    # 1. Trade count by regime (aggregated)
    total_bull = results_df["bull_trades"].sum()
    total_bear = results_df["bear_trades"].sum()
    
    fig.add_trace(
        go.Bar(
            x=["Bull Market", "Bear/Neutral"],
            y=[total_bull, total_bear],
            marker_color=["#28A745", "#DC3545"],
            text=[f"{total_bull:,.0f}", f"{total_bear:,.0f}"],
            textposition="auto",
        ),
        row=1, col=1,
    )
    
    # 2. P&L by regime
    total_bull_pnl = results_df["bull_trade_pnl"].sum()
    total_bear_pnl = results_df["bear_trade_pnl"].sum()
    
    fig.add_trace(
        go.Bar(
            x=["Bull Market", "Bear/Neutral"],
            y=[total_bull_pnl, total_bear_pnl],
            marker_color=["#28A745", "#DC3545"],
            text=[f"${total_bull_pnl:,.0f}", f"${total_bear_pnl:,.0f}"],
            textposition="auto",
        ),
        row=1, col=2,
    )
    
    # 3. Bull trade percentage by return (scatter)
    results_df["bull_trade_pct"] = results_df["bull_trades"] / (results_df["bull_trades"] + results_df["bear_trades"] + 0.001) * 100
    
    fig.add_trace(
        go.Scatter(
            x=results_df["total_return_pcnt"],
            y=results_df["bull_trade_pct"],
            mode="markers",
            marker=dict(
                size=8,
                color=results_df["total_return_pcnt"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Return %", x=0.45),
            ),
            hovertemplate=(
                "Return: %{x:.1f}%<br>"
                "Bull Trade %: %{y:.1f}%<br>"
                "<extra></extra>"
            ),
        ),
        row=2, col=1,
    )
    
    # 4. Return contribution pie
    bull_contrib = max(0, total_bull_pnl)
    bear_contrib = max(0, total_bear_pnl)
    
    if bull_contrib + bear_contrib > 0:
        fig.add_trace(
            go.Pie(
                labels=["Bull Markets", "Bear/Neutral"],
                values=[bull_contrib, bear_contrib],
                marker_colors=["#28A745", "#DC3545"],
                hole=0.4,
                textinfo="label+percent",
            ),
            row=2, col=2,
        )
    
    # Calculate summary stats
    best_row = results_df.loc[results_df["total_return_pcnt"].idxmax()]
    bull_weeks = int(best_row.get("bull_weeks", 0))
    bear_weeks = int(best_row.get("bear_weeks", 0))
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Regime Analysis: Bull Market Position Sizing +10%</b><br>"
                f"<sup>Total Bull Weeks: {bull_weeks} | Bear/Neutral Weeks: {bear_weeks} | "
                f"Bull P&L: ${total_bull_pnl/len(results_df):,.0f} avg | Bear P&L: ${total_bear_pnl/len(results_df):,.0f} avg</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=800,
        template="plotly_white",
        showlegend=False,
    )
    
    fig.update_xaxes(title_text="Regime", row=1, col=1)
    fig.update_xaxes(title_text="Regime", row=1, col=2)
    fig.update_xaxes(title_text="Total Return %", row=2, col=1)
    fig.update_yaxes(title_text="Trade Count", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=1, col=2)
    fig.update_yaxes(title_text="Bull Trade %", row=2, col=1)
    
    output_path = output_dir / "regime_analysis.html"
    fig.write_html(output_path)
    
    return output_path


# =============================================================================
# Main
# =============================================================================

@workflow_script("19c-run-regime-grid")
def main() -> None:
    """Run grid search with regime-dependent capital allocation."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid size
    total_combinations = len(STOP_LOSS_RANGE) * len(PROFIT_EXIT_RANGE) * len(MAX_HOLD_RANGE)
    logging.info(f"Grid: SL={STOP_LOSS_RANGE}, TP={PROFIT_EXIT_RANGE}, MaxHold={MAX_HOLD_RANGE}")
    logging.info(f"Total combinations: {total_combinations}")
    
    # Initialize Ray
    logging.info("🚀 Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    try:
        # Load data once into Ray Object Store
        logging.info("🔄 Loading shared data into Ray Object Store...")
        shared_data_ref = ray.put(ray.get(load_shared_data.remote()))
        logging.info("✅ Data loaded and shared!")
        
        # Run grid search with Ray Tune
        logging.info(f"🔍 Running grid search ({total_combinations} combinations)...")
        
        analysis = tune.run(
            ray_tune_trainable,
            config={
                "shared_data_ref": shared_data_ref,
                "stop_loss_pcnt": tune.grid_search(STOP_LOSS_RANGE),
                "profit_exit_pcnt": tune.grid_search(PROFIT_EXIT_RANGE),
                "max_hold_weeks": tune.grid_search(MAX_HOLD_RANGE),
            },
            num_samples=1,
            verbose=1,
            raise_on_failed_trial=False,
        )
        
        # Extract results
        logging.info("📊 Processing results...")
        results_list = []
        
        for trial in analysis.trials:
            if trial.last_result:
                results_list.append({
                    "stop_loss_pcnt": trial.config["stop_loss_pcnt"],
                    "profit_exit_pcnt": trial.config["profit_exit_pcnt"],
                    "max_hold_weeks": trial.config["max_hold_weeks"],
                    "total_trades": trial.last_result.get("total_trades", 0),
                    "winning_trades": trial.last_result.get("winning_trades", 0),
                    "losing_trades": trial.last_result.get("losing_trades", 0),
                    "win_rate": trial.last_result.get("win_rate", 0),
                    "total_pnl": trial.last_result.get("total_pnl", 0),
                    "total_return_pcnt": trial.last_result.get("total_return_pcnt", 0),
                    "max_drawdown_pcnt": trial.last_result.get("max_drawdown_pcnt", 0),
                    "bull_weeks": trial.last_result.get("bull_weeks", 0),
                    "bear_weeks": trial.last_result.get("bear_weeks", 0),
                    "bull_trades": trial.last_result.get("bull_trades", 0),
                    "bear_trades": trial.last_result.get("bear_trades", 0),
                    "bull_trade_pnl": trial.last_result.get("bull_trade_pnl", 0),
                    "bear_trade_pnl": trial.last_result.get("bear_trade_pnl", 0),
                })
        
        results_df = pd.DataFrame(results_list)
        
        # Save results
        results_df.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
        logging.info(f"Results saved: {OUTPUT_DIR / 'grid_results.csv'}")
        
        # Find best configurations
        if len(results_df) > 0:
            best_return = results_df.loc[results_df["total_return_pcnt"].idxmax()]
            best_winrate = results_df.loc[results_df["win_rate"].idxmax()]
            best_drawdown = results_df.loc[results_df["max_drawdown_pcnt"].idxmin()]
            
            results_df["return_per_dd"] = results_df["total_return_pcnt"] / (results_df["max_drawdown_pcnt"] + 1)
            best_risk_adj = results_df.loc[results_df["return_per_dd"].idxmax()]
            
            best_configs = {
                "generated_at": datetime.now().isoformat(),
                "total_combinations": total_combinations,
                "best_by_return": best_return.to_dict(),
                "best_by_winrate": best_winrate.to_dict(),
                "best_by_drawdown": best_drawdown.to_dict(),
                "best_risk_adjusted": best_risk_adj.to_dict(),
            }
            
            with open(OUTPUT_DIR / "best_configs.json", "w") as f:
                json.dump(best_configs, f, indent=2)
            
            # Create visualizations
            logging.info("📈 Creating visualizations...")
            heatmap_path = create_heatmap_dashboard(results_df, OUTPUT_DIR)
            maxhold_path = create_max_hold_comparison(results_df, OUTPUT_DIR)
            regime_path = create_regime_analysis(results_df, OUTPUT_DIR)
            
            # Print summary
            print("\n" + "=" * 70)
            print(f"GRID SEARCH COMPLETE: {EXPERIMENT_NAME}")
            print("=" * 70)
            
            print(f"\nGrid: {len(results_df)} of {total_combinations} combinations completed")
            
            print("\n" + "-" * 70)
            print("BEST CONFIGURATIONS:")
            print("-" * 70)
            
            print(f"\n📈 Best Return: {best_return['total_return_pcnt']:.1f}%")
            print(f"   SL={best_return['stop_loss_pcnt']}%, TP={best_return['profit_exit_pcnt']}%, "
                  f"MaxHold={best_return['max_hold_weeks']}w")
            print(f"   Win Rate: {best_return['win_rate']:.1f}%, Max DD: {best_return['max_drawdown_pcnt']:.1f}%")
            
            print(f"\n🎯 Best Win Rate: {best_winrate['win_rate']:.1f}%")
            print(f"   SL={best_winrate['stop_loss_pcnt']}%, TP={best_winrate['profit_exit_pcnt']}%, "
                  f"MaxHold={best_winrate['max_hold_weeks']}w")
            
            print(f"\n🛡️ Best Drawdown: {best_drawdown['max_drawdown_pcnt']:.1f}%")
            print(f"   SL={best_drawdown['stop_loss_pcnt']}%, TP={best_drawdown['profit_exit_pcnt']}%, "
                  f"MaxHold={best_drawdown['max_hold_weeks']}w")
            
            print(f"\n⚖️ Best Risk-Adjusted: {best_risk_adj['return_per_dd']:.2f}")
            print(f"   SL={best_risk_adj['stop_loss_pcnt']}%, TP={best_risk_adj['profit_exit_pcnt']}%, "
                  f"MaxHold={best_risk_adj['max_hold_weeks']}w")
            
            print("\n" + "-" * 70)
            print("Output Files:")
            print(f"  Results CSV:    {OUTPUT_DIR / 'grid_results.csv'}")
            print(f"  Best Configs:   {OUTPUT_DIR / 'best_configs.json'}")
            print(f"  Heatmaps:       {heatmap_path}")
            print(f"  MaxHold Chart:  {maxhold_path}")
            print(f"  Regime Analysis:{regime_path}")
            print("=" * 70)
            
            # Print regime summary
            print("\n" + "-" * 70)
            print("REGIME SUMMARY:")
            print("-" * 70)
            avg_bull_weeks = results_df["bull_weeks"].mean()
            avg_bear_weeks = results_df["bear_weeks"].mean()
            total_bull_pnl = results_df["bull_trade_pnl"].sum() / len(results_df)
            total_bear_pnl = results_df["bear_trade_pnl"].sum() / len(results_df)
            print(f"  Avg Bull Weeks: {avg_bull_weeks:.0f}")
            print(f"  Avg Bear Weeks: {avg_bear_weeks:.0f}")
            print(f"  Avg Bull P&L: ${total_bull_pnl:,.0f}")
            print(f"  Avg Bear P&L: ${total_bear_pnl:,.0f}")
            print(f"  Position Multiplier in Bull: {BULL_MARKET_MULTIPLIER}x")
            print("=" * 70)
        else:
            print("❌ No results collected!")
    
    finally:
        ray.shutdown()
    
    logging.info("Grid search complete")


if __name__ == "__main__":
    main()

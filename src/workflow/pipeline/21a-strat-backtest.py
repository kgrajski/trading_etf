#!/usr/bin/env python3
"""
21a-strat-backtest.py

Champion Strategy Backtest with Period Analysis.

Runs the optimized mean-reversion strategy on full historical data,
then analyzes performance by period to detect model slippage.

Champion Parameters (from grid search 19d + robustness 20):
- Stop Loss: 16%
- Profit Target: 10%
- Max Hold: 1 week
- Regime: Bear market boost (1.10x when SPY < MA50)

Outputs:
- experiments/exp021_champion_backtest/
  - config.json
  - trades.csv
  - summary.json
  - dashboard.html
"""

import json
import logging
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
# Champion Configuration (Optimized from 19d Grid Search)
# =============================================================================

EXPERIMENT_NAME = "exp021_champion_backtest"

# Capital parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10

# Entry parameters
BOTTOM_PERCENTILE = 0.05  # Bottom 5% losers
MIN_LOSS_PCNT = 2.0  # Minimum weekly loss to qualify

# Exit parameters (CHAMPION - optimized)
STOP_LOSS_PCNT = 16.0
PROFIT_EXIT_PCNT = 10.0
MAX_HOLD_WEEKS = 1

# Regime parameters (CHAMPION - bear boost)
USE_REGIME = True
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50
BEAR_MARKET_MULTIPLIER = 1.10  # 10% larger positions in bear markets

# Analysis parameters
RECENT_WEEKS = 52  # Split point for historical vs recent
SLIPPAGE_ALARM_THRESHOLD = 5.0  # Warn if win rate drops > 5%

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME


# =============================================================================
# Data Loading
# =============================================================================

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


def get_target_symbols() -> List[str]:
    """Get list of target ETF symbols."""
    path = METADATA_DIR / "filtered_etfs.json"
    if not path.exists():
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    with open(path) as f:
        data = json.load(f)
    return [etf["symbol"] for etf in data.get("etfs", [])]


def calculate_regime(daily_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Calculate market regime for each trading day."""
    if REGIME_SYMBOL not in daily_data:
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
# Backtest Engine
# =============================================================================

def run_champion_backtest(
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    regime_by_date: Dict[str, Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Run champion strategy backtest.
    
    Returns:
        trades: List of trade dictionaries with full metadata
        summary: Aggregate metrics
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
            position_multiplier = 1.0
        else:
            bear_weeks += 1
            position_multiplier = BEAR_MARKET_MULTIPLIER
        
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
            
            stop_price = entry_price * (1 - STOP_LOSS_PCNT / 100)
            profit_price = entry_price * (1 + PROFIT_EXIT_PCNT / 100)
            max_hold_exit_week = entry_week_idx + MAX_HOLD_WEEKS - 1
            
            for day in trading_days:
                if pd.Timestamp(day) <= pd.Timestamp(entry_date):
                    continue
                
                day_df = daily_data[symbol]
                day_row = day_df[day_df["date"] == pd.Timestamp(day)]
                
                if len(day_row) == 0:
                    continue
                
                row = day_row.iloc[0]
                
                # Check stop loss
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
                
                # Check max hold (Friday exit)
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
    """Compute metrics for a set of trades."""
    # Filter to filled trades only
    filled = [t for t in trades if t["entry_date"] is not None]
    expired = [t for t in trades if t["exit_reason"] == "expired"]
    
    if not filled:
        return {
            "period": period_name,
            "total_signals": len(trades),
            "filled_trades": 0,
            "expired_orders": len(expired),
            "win_rate": 0,
        }
    
    # Exit reason breakdown
    exit_reasons = {}
    for t in filled:
        reason = t["exit_reason"] or "open"
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    
    # Win/loss
    winners = [t for t in filled if t["pnl_dollars"] > 0]
    losers = [t for t in filled if t["pnl_dollars"] <= 0]
    
    # P&L by exit reason
    pnl_by_reason = {}
    count_by_reason = {}
    for t in filled:
        reason = t["exit_reason"] or "open"
        pnl_by_reason[reason] = pnl_by_reason.get(reason, 0) + t["pnl_dollars"]
        count_by_reason[reason] = count_by_reason.get(reason, 0) + 1
    
    avg_pnl_by_reason = {r: pnl_by_reason[r] / count_by_reason[r] 
                         for r in pnl_by_reason}
    
    # Cumulative P&L by exit reason (this is what matters!)
    cumulative_pnl_by_reason = pnl_by_reason.copy()
    
    # Regime breakdown
    bull_trades = [t for t in filled if t.get("is_bull_entry") is True]
    bear_trades = [t for t in filled if t.get("is_bull_entry") is False]
    
    bull_pnl = sum(t["pnl_dollars"] for t in bull_trades)
    bear_pnl = sum(t["pnl_dollars"] for t in bear_trades)
    
    bull_winners = sum(1 for t in bull_trades if t["pnl_dollars"] > 0)
    bear_winners = sum(1 for t in bear_trades if t["pnl_dollars"] > 0)
    
    total_pnl = sum(t["pnl_dollars"] for t in filled)
    
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
        "cumulative_pnl_by_reason": cumulative_pnl_by_reason,
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_pnl": bull_pnl,
        "bear_pnl": bear_pnl,
        "bull_win_rate": bull_winners / len(bull_trades) * 100 if bull_trades else 0,
        "bear_win_rate": bear_winners / len(bear_trades) * 100 if bear_trades else 0,
    }


def detect_slippage(historical: Dict, recent: Dict) -> Dict:
    """Compare historical vs recent performance for slippage."""
    alarms = []
    
    win_rate_delta = recent["win_rate"] - historical["win_rate"]
    if win_rate_delta < -SLIPPAGE_ALARM_THRESHOLD:
        alarms.append(f"⚠️ WIN RATE SLIPPAGE: {win_rate_delta:+.1f}% "
                     f"(Historical: {historical['win_rate']:.1f}%, "
                     f"Recent: {recent['win_rate']:.1f}%)")
    
    # Compare avg P&L
    hist_avg = historical.get("avg_pnl_per_trade", 0)
    recent_avg = recent.get("avg_pnl_per_trade", 0)
    if hist_avg > 0 and recent_avg < hist_avg * 0.5:
        alarms.append(f"⚠️ AVG P&L SLIPPAGE: Dropped from ${hist_avg:.2f} to ${recent_avg:.2f}")
    
    return {
        "win_rate_delta": win_rate_delta,
        "avg_pnl_delta": recent_avg - hist_avg,
        "alarms": alarms,
    }


# =============================================================================
# Dashboard
# =============================================================================

def create_dashboard(
    trades: List[Dict],
    historical_stats: Dict,
    recent_stats: Dict,
    slippage: Dict,
    output_dir: Path,
) -> Path:
    """Create interactive dashboard with period comparison."""
    
    # Filter to filled trades with exit dates
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
    
    # Find split point
    split_idx = len(sorted_trades) - min(RECENT_WEEKS * 5, len(sorted_trades) // 2)
    if split_idx > 0:
        split_date = sorted_trades[split_idx]["exit_date"]
    else:
        split_date = dates[len(dates) // 2]
    
    # Create figure
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "pie"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "Equity Curve (Champion Strategy)",
            f"Exit Reasons - Historical", f"Exit Reasons - Recent {RECENT_WEEKS}w",
            "P&L by Regime - Historical", f"P&L by Regime - Recent {RECENT_WEEKS}w",
        ),
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.1,
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity,
            mode="lines",
            name="Equity",
            line=dict(color="#2E86AB", width=2),
            hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Add split line (shape instead of vline to avoid annotation issues)
    fig.add_shape(
        type="line",
        x0=split_date, x1=split_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash"),
        row=1, col=1,
    )
    fig.add_annotation(
        x=split_date, y=1.05, yref="paper",
        text=f"← Historical | Recent {RECENT_WEEKS}w →",
        showarrow=False,
        font=dict(size=10, color="gray"),
        row=1, col=1,
    )
    
    # Exit reason pie charts
    for stats, col in [(historical_stats, 1), (recent_stats, 2)]:
        reasons = stats.get("exit_reasons", {})
        if reasons:
            labels = list(reasons.keys())
            values = list(reasons.values())
            colors = {
                "max_hold": "#4CAF50",
                "profit_exit": "#2196F3", 
                "stop_loss": "#F44336",
                "end_of_data": "#9E9E9E",
            }
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=[colors.get(l, "#666") for l in labels],
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value} trades<extra></extra>",
                ),
                row=2, col=col,
            )
    
    # Regime P&L bars
    for stats, col in [(historical_stats, 1), (recent_stats, 2)]:
        fig.add_trace(
            go.Bar(
                x=["Bull Entry", "Bear Entry"],
                y=[stats.get("bull_pnl", 0), stats.get("bear_pnl", 0)],
                marker_color=["#4CAF50", "#F44336"],
                text=[f"${stats.get('bull_pnl', 0):,.0f}", 
                      f"${stats.get('bear_pnl', 0):,.0f}"],
                textposition="outside",
                hovertemplate="%{x}: $%{y:,.2f}<extra></extra>",
            ),
            row=3, col=col,
        )
    
    # Build summary table HTML
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
            <th style="padding:10px;text-align:right;">Recent {RECENT_WEEKS}w</th>
            <th style="padding:10px;text-align:right;">Delta</th>
        </tr>
        <tr>
            <td style="padding:8px;">Filled Trades</td>
            <td style="padding:8px;text-align:right;">{historical_stats['filled_trades']}</td>
            <td style="padding:8px;text-align:right;">{recent_stats['filled_trades']}</td>
            <td style="padding:8px;text-align:right;">-</td>
        </tr>
        <tr style="background:#f9f9f9;">
            <td style="padding:8px;">Win Rate</td>
            <td style="padding:8px;text-align:right;">{historical_stats['win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{recent_stats['win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;color:{'red' if slippage['win_rate_delta'] < 0 else 'green'};">{slippage['win_rate_delta']:+.1f}%</td>
        </tr>
        <tr>
            <td style="padding:8px;">Total P&L</td>
            <td style="padding:8px;text-align:right;">${historical_stats['total_pnl']:,.2f}</td>
            <td style="padding:8px;text-align:right;">${recent_stats['total_pnl']:,.2f}</td>
            <td style="padding:8px;text-align:right;">-</td>
        </tr>
        <tr style="background:#f9f9f9;">
            <td style="padding:8px;">Avg P&L/Trade</td>
            <td style="padding:8px;text-align:right;">${historical_stats['avg_pnl_per_trade']:.2f}</td>
            <td style="padding:8px;text-align:right;">${recent_stats['avg_pnl_per_trade']:.2f}</td>
            <td style="padding:8px;text-align:right;color:{'red' if slippage['avg_pnl_delta'] < 0 else 'green'};">${slippage['avg_pnl_delta']:+.2f}</td>
        </tr>
        <tr>
            <td style="padding:8px;">Bull Entry Win Rate</td>
            <td style="padding:8px;text-align:right;">{historical_stats['bull_win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{recent_stats['bull_win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">-</td>
        </tr>
        <tr style="background:#f9f9f9;">
            <td style="padding:8px;">Bear Entry Win Rate</td>
            <td style="padding:8px;text-align:right;">{historical_stats['bear_win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">{recent_stats['bear_win_rate']:.1f}%</td>
            <td style="padding:8px;text-align:right;">-</td>
        </tr>
    </table>
    """
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>Champion Strategy Backtest</b><br>"
                 f"<sub>SL={STOP_LOSS_PCNT}% | TP={PROFIT_EXIT_PCNT}% | MaxHold={MAX_HOLD_WEEKS}w | "
                 f"Bear Boost={BEAR_MARKET_MULTIPLIER}x</sub>",
            x=0.5,
        ),
        height=900,
        showlegend=False,
    )
    
    # Save
    output_path = output_dir / "dashboard.html"
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Champion Strategy Backtest</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .config {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Champion Strategy Backtest</h1>
    <div class="config">
        <strong>Champion Parameters:</strong> 
        SL={STOP_LOSS_PCNT}% | TP={PROFIT_EXIT_PCNT}% | MaxHold={MAX_HOLD_WEEKS}w | 
        Bear Boost={BEAR_MARKET_MULTIPLIER}x | Regime={REGIME_SYMBOL} vs MA{REGIME_MA_PERIOD}
    </div>
    {alarm_html}
    <h2>Period Comparison</h2>
    {summary_table}
    {fig.to_html(full_html=False, include_plotlyjs='cdn')}
</body>
</html>
"""
    
    output_path.write_text(html_content)
    return output_path


# =============================================================================
# Main
# =============================================================================

@workflow_script("21a-strat-backtest")
def main():
    """Run champion strategy backtest with period analysis."""
    
    print("=" * 70)
    print("CHAMPION STRATEGY BACKTEST")
    print("=" * 70)
    print()
    print("Configuration:")
    print(f"  Stop Loss:        {STOP_LOSS_PCNT}%")
    print(f"  Profit Target:    {PROFIT_EXIT_PCNT}%")
    print(f"  Max Hold:         {MAX_HOLD_WEEKS} week(s)")
    print(f"  Bear Boost:       {BEAR_MARKET_MULTIPLIER}x (when SPY < MA{REGIME_MA_PERIOD})")
    print(f"  Recent Period:    {RECENT_WEEKS} weeks")
    print(f"  Slippage Alarm:   >{SLIPPAGE_ALARM_THRESHOLD}% win rate drop")
    print()
    
    # Create output dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    symbols = get_target_symbols()
    symbols.append(REGIME_SYMBOL)
    symbols = list(set(symbols))
    
    daily_data = load_daily_data(symbols)
    weekly_data = load_weekly_data(symbols)
    regime_by_date = calculate_regime(daily_data)
    
    print(f"  Symbols: {len(daily_data)}")
    print(f"  Regime data points: {len(regime_by_date)}")
    print()
    
    # Run backtest
    print("Running champion backtest...")
    trades, summary = run_champion_backtest(daily_data, weekly_data, regime_by_date)
    print(f"  Total signals: {len(trades)}")
    print(f"  Bull weeks: {summary['bull_weeks']}, Bear weeks: {summary['bear_weeks']}")
    print()
    
    # Split trades by period (3-way: Historical, Prior 52w, Recent 52w)
    filled_trades = [t for t in trades if t["entry_date"] is not None]
    sorted_filled = sorted(filled_trades, key=lambda t: t["entry_date"])
    
    trading_weeks = get_trading_weeks(weekly_data)
    total_weeks = len(trading_weeks)
    
    # Recent = last 52 weeks
    recent_split_idx = max(0, total_weeks - RECENT_WEEKS)
    # Prior = 52 weeks before recent
    prior_split_idx = max(0, total_weeks - 2 * RECENT_WEEKS)
    
    if recent_split_idx < len(trading_weeks):
        recent_split_date = trading_weeks[recent_split_idx][0]
    else:
        recent_split_date = sorted_filled[len(sorted_filled) * 2 // 3]["entry_date"] if sorted_filled else None
    
    if prior_split_idx < len(trading_weeks):
        prior_split_date = trading_weeks[prior_split_idx][0]
    else:
        prior_split_date = sorted_filled[len(sorted_filled) // 3]["entry_date"] if sorted_filled else None
    
    # Split trades
    historical_trades = [t for t in trades if t["entry_date"] is None or t["entry_date"] < prior_split_date]
    prior_trades = [t for t in trades if t["entry_date"] and prior_split_date <= t["entry_date"] < recent_split_date]
    recent_trades = [t for t in trades if t["entry_date"] and t["entry_date"] >= recent_split_date]
    
    # Analyze periods
    print("Analyzing periods...")
    print(f"  Historical: before {prior_split_date}")
    print(f"  Prior {RECENT_WEEKS}w: {prior_split_date} to {recent_split_date}")
    print(f"  Recent {RECENT_WEEKS}w: {recent_split_date} onwards")
    
    all_stats = analyze_trades(trades, "All")
    historical_stats = analyze_trades(historical_trades, "Historical")
    prior_stats = analyze_trades(prior_trades, f"Prior {RECENT_WEEKS}w")
    recent_stats = analyze_trades(recent_trades, f"Recent {RECENT_WEEKS}w")
    
    # Detect slippage (compare prior period to recent for fair comparison)
    slippage = detect_slippage(prior_stats, recent_stats)
    
    # Print results
    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print()
    print(f"{'Metric':<20} {'Historical':>15} {'Prior 52w':>15} {'Recent 52w':>15} {'Trend':>15}")
    print("-" * 90)
    print(f"{'Filled Trades':<20} {historical_stats['filled_trades']:>15} {prior_stats['filled_trades']:>15} {recent_stats['filled_trades']:>15}")
    print(f"{'Win Rate':<20} {historical_stats['win_rate']:>14.1f}% {prior_stats['win_rate']:>14.1f}% {recent_stats['win_rate']:>14.1f}% {slippage['win_rate_delta']:>+14.1f}%")
    print(f"{'Total P&L':<20} ${historical_stats['total_pnl']:>13,.2f} ${prior_stats['total_pnl']:>13,.2f} ${recent_stats['total_pnl']:>13,.2f}")
    print(f"{'Avg P&L/Trade':<20} ${historical_stats['avg_pnl_per_trade']:>13,.2f} ${prior_stats['avg_pnl_per_trade']:>13,.2f} ${recent_stats['avg_pnl_per_trade']:>13,.2f}")
    print()
    
    # Exit reasons for all three periods
    for period_name, stats in [("Historical", historical_stats), 
                                (f"Prior {RECENT_WEEKS}w", prior_stats),
                                (f"Recent {RECENT_WEEKS}w", recent_stats)]:
        print(f"Exit Reasons ({period_name}):")
        for reason, count in stats.get("exit_reasons", {}).items():
            avg = stats.get("avg_pnl_by_reason", {}).get(reason, 0)
            cumul = stats.get("cumulative_pnl_by_reason", {}).get(reason, 0)
            print(f"  {reason:<15}: {count:>5} trades, avg ${avg:>8.2f}, CUMULATIVE ${cumul:>10,.2f}")
        print()
    
    # Check for concerning pattern: max_hold cumulative trend
    hist_max_hold_pnl = historical_stats.get("cumulative_pnl_by_reason", {}).get("max_hold", 0)
    prior_max_hold_pnl = prior_stats.get("cumulative_pnl_by_reason", {}).get("max_hold", 0)
    recent_max_hold_pnl = recent_stats.get("cumulative_pnl_by_reason", {}).get("max_hold", 0)
    
    print("=" * 90)
    print("MAX_HOLD CUMULATIVE P&L TREND (Core Strategy Health)")
    print("=" * 90)
    print(f"  Historical:   ${hist_max_hold_pnl:>10,.2f}")
    print(f"  Prior 52w:    ${prior_max_hold_pnl:>10,.2f}")
    print(f"  Recent 52w:   ${recent_max_hold_pnl:>10,.2f}")
    print()
    
    if recent_max_hold_pnl < 0 and prior_max_hold_pnl < 0:
        print("🚨 ALARM: max_hold exits NEGATIVE for both prior AND recent periods!")
        print("    The core mean-reversion thesis may no longer be working.")
        print()
    elif recent_max_hold_pnl < 0:
        print("⚠️  WARNING: Recent max_hold exits have NEGATIVE cumulative P&L!")
        print("    Strategy currently relies on rare profit_exit wins.")
        print()
    elif recent_max_hold_pnl < prior_max_hold_pnl * 0.5 and prior_max_hold_pnl > 0:
        print("⚠️  WARNING: Recent max_hold P&L dropped significantly from prior period.")
        print()
    
    # Alarms
    if slippage["alarms"]:
        print("=" * 70)
        print("⚠️  SLIPPAGE ALARMS")
        print("=" * 70)
        for alarm in slippage["alarms"]:
            print(alarm)
        print()
    
    # Save outputs
    print("Saving outputs...")
    
    # Config
    config = {
        "champion_parameters": {
            "stop_loss_pcnt": STOP_LOSS_PCNT,
            "profit_exit_pcnt": PROFIT_EXIT_PCNT,
            "max_hold_weeks": MAX_HOLD_WEEKS,
            "min_loss_pcnt": MIN_LOSS_PCNT,
            "bottom_percentile": BOTTOM_PERCENTILE,
            "initial_capital": INITIAL_CAPITAL,
            "max_active_trades": MAX_ACTIVE_TRADES,
        },
        "regime": {
            "enabled": USE_REGIME,
            "symbol": REGIME_SYMBOL,
            "ma_period": REGIME_MA_PERIOD,
            "bear_multiplier": BEAR_MARKET_MULTIPLIER,
        },
        "analysis": {
            "recent_weeks": RECENT_WEEKS,
            "slippage_threshold": SLIPPAGE_ALARM_THRESHOLD,
        },
        "generated_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Trades CSV
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    
    # Summary JSON
    summary_out = {
        "all": all_stats,
        "historical": historical_stats,
        "prior_52w": prior_stats,
        "recent_52w": recent_stats,
        "slippage": slippage,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    
    # Dashboard
    dashboard_path = create_dashboard(
        trades, historical_stats, recent_stats, slippage, OUTPUT_DIR
    )
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Dashboard: {dashboard_path}")
    print(f"  Trades:    {OUTPUT_DIR / 'trades.csv'}")
    print(f"  Summary:   {OUTPUT_DIR / 'summary.json'}")
    print(f"  Config:    {OUTPUT_DIR / 'config.json'}")


if __name__ == "__main__":
    main()

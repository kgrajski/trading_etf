#!/usr/bin/env python3
"""
19.1b-grid-search.py

Grid Search with All Features + Period Stability Analysis.

Consolidates all bells & whistles from 19b, 19d, and adds:
- Regime awareness (SPY vs MA50)
- Configurable boost direction (bull/bear/none)
- Period stability analysis per config
- Flags for configs that degraded recently

The core backtest logic MUST match 19.1-full-backtest.py exactly.

Grid:
- SL: 4% to 16% in steps of 2%
- TP: 2% to 16% in steps of 2%
- MaxHold: 1 to 8 weeks

Outputs:
- experiments/exp019_1b_grid/
  - grid_results.csv (includes period stability metrics)
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

# Grid parameters
STOP_LOSS_RANGE = [4, 6, 8, 10, 12, 14, 16]  # %
PROFIT_EXIT_RANGE = [2, 4, 6, 8, 10, 12, 14, 16]  # %
MAX_HOLD_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]  # weeks

# Fixed parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10
BOTTOM_PERCENTILE = 0.05
MIN_LOSS_PCNT = 2.0

# Regime parameters
USE_REGIME = True
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50

# Boost parameters
BOOST_DIRECTION = "bear"  # Options: "bull", "bear", "none"
BOOST_MULTIPLIER = 1.10

# Period analysis
RECENT_WEEKS = 52
STABILITY_THRESHOLD = 5.0  # Flag if win rate drops > this %


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
# Core Backtest Logic (MUST MATCH 19.1)
# =============================================================================

def get_position_multiplier(is_bull_week: bool) -> float:
    """Get position size multiplier based on regime and boost config."""
    if not USE_REGIME or BOOST_DIRECTION == "none":
        return 1.0
    
    if BOOST_DIRECTION == "bear" and not is_bull_week:
        return BOOST_MULTIPLIER
    elif BOOST_DIRECTION == "bull" and is_bull_week:
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
) -> Dict[str, Any]:
    """
    Run backtest and return metrics including period stability.
    
    This logic MUST match 19.1-full-backtest.py exactly.
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
        
        position_multiplier = get_position_multiplier(is_bull_week)
        
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
                        "entry_price": fill_price,
                        "shares": shares,
                        "position_value": position_value,
                        "is_bull_entry": is_bull_week,
                        "is_open": True,
                        "exit_date": None,
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
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "stop_loss"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Profit target
                if row["high"] >= profit_price:
                    exit_price = max(profit_price, row["open"])
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "profit_exit"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Max hold (Friday exit)
                is_last_day = (day == last_day)
                is_max_hold_week = (week_idx >= max_hold_exit_week)
                
                if is_last_day and is_max_hold_week:
                    exit_price = row["open"]
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "max_hold"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
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
        qualified = bottom_df[bottom_df["pct_return"] <= -MIN_LOSS_PCNT]
        
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
        return {"total_trades": 0, "win_rate": 0, "total_pnl": 0}
    
    # Overall metrics
    total_trades = len(closed_trades)
    winning = sum(1 for t in closed_trades if t["pnl_dollars"] > 0)
    total_pnl = sum(t["pnl_dollars"] for t in closed_trades)
    
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
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": total_trades - winning,
        "win_rate": winning / total_trades * 100,
        "total_pnl": total_pnl,
        "total_return_pcnt": total_pnl / INITIAL_CAPITAL * 100,
        "max_drawdown_pcnt": max_dd,
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
    )
    
    tune.report(metrics=result)


# =============================================================================
# Visualization
# =============================================================================

def create_heatmap_dashboard(results_df: pd.DataFrame, output_dir: Path) -> Path:
    """Create heatmap dashboard."""
    best_row = results_df.loc[results_df["total_return_pcnt"].idxmax()]
    best_max_hold = int(best_row["max_hold_weeks"])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Return % (MaxHold={best_max_hold}w)",
            f"Win Rate % (MaxHold={best_max_hold}w)",
            f"Recent Win Rate Delta (MaxHold={best_max_hold}w)",
            f"Recent max_hold P&L (MaxHold={best_max_hold}w)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )
    
    df_slice = results_df[results_df["max_hold_weeks"] == best_max_hold]
    
    for metric, colorscale, row, col in [
        ("total_return_pcnt", "RdYlGn", 1, 1),
        ("win_rate", "Blues", 1, 2),
        ("win_rate_delta", "RdYlGn", 2, 1),
        ("recent_max_hold_pnl", "RdYlGn", 2, 2),
    ]:
        pivot = df_slice.pivot(
            index="stop_loss_pcnt",
            columns="profit_exit_pcnt",
            values=metric,
        )
        
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=colorscale,
                showscale=True,
                hovertemplate=f"SL: %{{y}}%<br>TP: %{{x}}%<br>{metric}: %{{z:.2f}}<extra></extra>",
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        title=f"<b>Grid Search Results: {EXPERIMENT_NAME}</b><br>"
              f"<sub>Boost={BOOST_DIRECTION} {BOOST_MULTIPLIER}x | Regime={REGIME_SYMBOL} MA{REGIME_MA_PERIOD}</sub>",
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
    # Top configs
    top_by_return = results_df.nlargest(10, "total_return_pcnt")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Top 10 Configs: Prior vs Recent Win Rate",
            "Top 10 Configs: Recent max_hold P&L",
        ),
        vertical_spacing=0.15,
    )
    
    # Win rate comparison
    labels = [f"SL{int(r['stop_loss_pcnt'])}_TP{int(r['profit_exit_pcnt'])}_H{int(r['max_hold_weeks'])}" 
              for _, r in top_by_return.iterrows()]
    
    fig.add_trace(
        go.Bar(name="Prior 52w", x=labels, y=top_by_return["prior_win_rate"].tolist(),
               marker_color="#4CAF50"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(name="Recent 52w", x=labels, y=top_by_return["recent_win_rate"].tolist(),
               marker_color="#2196F3"),
        row=1, col=1,
    )
    
    # Recent max_hold P&L
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in top_by_return["recent_max_hold_pnl"]]
    fig.add_trace(
        go.Bar(name="Recent max_hold P&L", x=labels, 
               y=top_by_return["recent_max_hold_pnl"].tolist(),
               marker_color=colors, showlegend=False),
        row=2, col=1,
    )
    
    fig.update_layout(
        title="<b>Period Stability Analysis</b>",
        height=700,
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
    """Run grid search with period stability analysis."""
    
    print("=" * 70)
    print("19.1b GRID SEARCH")
    print("=" * 70)
    print()
    print(f"Grid: SL={STOP_LOSS_RANGE}, TP={PROFIT_EXIT_RANGE}, MaxHold={MAX_HOLD_RANGE}")
    print(f"Total combinations: {len(STOP_LOSS_RANGE) * len(PROFIT_EXIT_RANGE) * len(MAX_HOLD_RANGE)}")
    print(f"Regime: {REGIME_SYMBOL} MA{REGIME_MA_PERIOD}")
    print(f"Boost: {BOOST_DIRECTION} {BOOST_MULTIPLIER}x")
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
                    **metrics,
                })
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
        
        # Find best configs
        best_return = results_df.loc[results_df["total_return_pcnt"].idxmax()]
        best_stable = results_df[~results_df["is_degraded"]].nlargest(1, "total_return_pcnt")
        best_risk_adj = results_df.loc[(results_df["total_return_pcnt"] / results_df["max_drawdown_pcnt"].clip(lower=1)).idxmax()]
        
        best_configs = {
            "generated_at": datetime.now().isoformat(),
            "regime": {"symbol": REGIME_SYMBOL, "ma_period": REGIME_MA_PERIOD},
            "boost": {"direction": BOOST_DIRECTION, "multiplier": BOOST_MULTIPLIER},
            "best_by_return": best_return.to_dict(),
            "best_stable": best_stable.iloc[0].to_dict() if len(best_stable) > 0 else None,
            "best_risk_adjusted": best_risk_adj.to_dict(),
            "degraded_count": int(results_df["is_degraded"].sum()),
            "stable_count": int((~results_df["is_degraded"]).sum()),
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
        print(f"Best by Return: SL={best_return['stop_loss_pcnt']}%, TP={best_return['profit_exit_pcnt']}%, MaxHold={best_return['max_hold_weeks']}w")
        print(f"  Return: {best_return['total_return_pcnt']:.1f}%, Win Rate: {best_return['win_rate']:.1f}%")
        print(f"  Recent Win Rate Delta: {best_return['win_rate_delta']:+.1f}%")
        print(f"  Recent max_hold P&L: ${best_return['recent_max_hold_pnl']:,.2f}")
        print(f"  Degraded: {'YES ⚠️' if best_return['is_degraded'] else 'No ✅'}")
        print()
        print(f"Stability Summary: {best_configs['stable_count']} stable, {best_configs['degraded_count']} degraded")
        print()
        print(f"Outputs:")
        print(f"  Results: {OUTPUT_DIR / 'grid_results.csv'}")
        print(f"  Heatmaps: {heatmap_path}")
        print(f"  Stability: {stability_path}")
        
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

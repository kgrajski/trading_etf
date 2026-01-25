#!/usr/bin/env python3
"""
20-run-robustness-check.py

Robustness check for mean-reversion backtest hyperparameters.

Tests the top configurations from 19b or 19c across different starting weeks
to validate that performance is consistent across market conditions.

Approach:
- Stratified sampling of start dates by market regime (bull/bear)
- Run each config for a fixed test period (26 weeks = 6 months)
- Record performance metrics for each (config, start_date) pair

Inputs:
- robustness_configs.json from source experiment (19b or 19c)
- Daily and weekly price data

Outputs:
- experiments/exp020_robustness_{source}/robustness_results.csv
- experiments/exp020_robustness_{source}/equity_curves.csv
- experiments/exp020_robustness_{source}/run_metadata.json

Usage:
    python 20-run-robustness-check.py
"""
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import ray
from ray import tune
from ray.tune import RunConfig

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

# Source experiment to test (change this to test different experiments)
SOURCE_EXPERIMENT = "exp019_grid_regime"  # Use regime scaling version

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Robustness check parameters
TEST_PERIOD_WEEKS = 26  # 6 months
SAMPLES_PER_REGIME = 10  # Number of start dates per regime
RANDOM_SEED = 42

# Backtest parameters (fixed)
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10
BOTTOM_PERCENTILE = 0.05
MIN_LOSS_PCNT = 2.0

# Regime detection parameters (must match 19c for consistency)
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50
BULL_MARKET_MULTIPLIER = 1.10


# =============================================================================
# Data Loading
# =============================================================================

def get_target_symbols() -> List[str]:
    """Get list of target ETF symbols."""
    filtered_etfs_path = METADATA_DIR / "filtered_etfs.json"
    
    if not filtered_etfs_path.exists():
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    
    with open(filtered_etfs_path) as f:
        data = json.load(f)
    
    return [etf["symbol"] for etf in data.get("etfs", [])]


def load_shared_data() -> Dict[str, Any]:
    """Load all data for backtesting."""
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
    
    print(f"✅ Loaded {len(daily_data)} daily, {len(weekly_data)} weekly files")
    
    return {
        "daily": daily_data,
        "weekly": weekly_data,
        "symbols": symbols,
        "regime_by_date": regime_by_date,
    }


def get_trading_weeks(weekly_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
    """Get sorted list of (week_start, week_end) tuples."""
    all_weeks = set()
    for symbol, df in weekly_data.items():
        for _, row in df.iterrows():
            all_weeks.add((
                row["week_start"].strftime("%Y-%m-%d"),
                row["week_end"].strftime("%Y-%m-%d")
            ))
    return sorted(all_weeks, key=lambda x: x[0])


def classify_weeks_by_regime(
    trading_weeks: List[Tuple[str, str]],
    regime_by_date: Dict[str, Dict],
    daily_data: Dict[str, pd.DataFrame],
) -> Dict[str, List[int]]:
    """
    Classify each week by regime based on the first trading day.
    
    Returns:
        Dict with 'bull' and 'bear' lists of week indices
    """
    # Get all trading days
    sample_daily = next(iter(daily_data.values()))
    all_trading_days = sample_daily["date"].dt.strftime("%Y-%m-%d").tolist()
    
    bull_weeks = []
    bear_weeks = []
    
    for week_idx, (week_start, week_end) in enumerate(trading_weeks):
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        # Get first trading day of this week
        trading_days = [d for d in all_trading_days if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            continue
        
        first_day = trading_days[0]
        regime_info = regime_by_date.get(first_day, {"is_bull": False})
        
        if regime_info.get("is_bull", False):
            bull_weeks.append(week_idx)
        else:
            bear_weeks.append(week_idx)
    
    return {"bull": bull_weeks, "bear": bear_weeks}


def sample_start_weeks(
    week_regimes: Dict[str, List[int]],
    total_weeks: int,
    test_period: int,
    samples_per_regime: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Stratified sample of start weeks.
    
    Only considers weeks that have enough subsequent data for the test period.
    """
    random.seed(seed)
    
    # Valid start weeks must have test_period weeks of data following
    max_start_idx = total_weeks - test_period
    
    samples = []
    
    for regime, week_indices in week_regimes.items():
        # Filter to valid start weeks
        valid_starts = [idx for idx in week_indices if idx <= max_start_idx]
        
        if len(valid_starts) == 0:
            print(f"⚠️ No valid start weeks for {regime} regime")
            continue
        
        # Sample (or take all if fewer than requested)
        n_samples = min(samples_per_regime, len(valid_starts))
        selected = random.sample(valid_starts, n_samples)
        
        for week_idx in selected:
            samples.append({
                "week_idx": week_idx,
                "start_regime": regime,
            })
        
        print(f"📌 Sampled {n_samples} start weeks from {regime} regime (of {len(valid_starts)} valid)")
    
    return samples


# =============================================================================
# Backtest Logic (adapted from 19c)
# =============================================================================

def run_backtest_for_period(
    daily_data: Dict[str, pd.DataFrame],
    weekly_data: Dict[str, pd.DataFrame],
    regime_by_date: Dict[str, Dict],
    trading_weeks: List[Tuple[str, str]],
    start_week_idx: int,
    end_week_idx: int,
    stop_loss_pcnt: float,
    profit_exit_pcnt: float,
    max_hold_weeks: int,
    use_regime_scaling: bool = False,
) -> Dict[str, Any]:
    """
    Run backtest for a specific period (start to end week indices).
    
    Args:
        start_week_idx: Index into trading_weeks for first week
        end_week_idx: Index into trading_weeks for last week (exclusive)
        use_regime_scaling: If True, apply 10% larger positions in bull markets
    
    Returns:
        Dict with metrics and trade details
    """
    # Get subset of weeks
    period_weeks = trading_weeks[start_week_idx:end_week_idx]
    
    if not period_weeks:
        return {
            "total_trades": 0, "win_rate": 0, "total_pnl": 0, 
            "total_return_pcnt": 0, "max_drawdown_pcnt": 0,
            "bull_weeks": 0, "bear_weeks": 0,
        }
    
    # Get all trading days
    sample_daily = next(iter(daily_data.values()))
    all_trading_days = sample_daily["date"].dt.strftime("%Y-%m-%d").tolist()
    
    # State tracking
    trades = []
    next_trade_id = 1
    current_capital = INITIAL_CAPITAL
    committed_capital = 0.0
    pending_orders = []
    
    # Regime tracking
    bull_weeks = 0
    bear_weeks = 0
    bull_trades = 0
    bear_trades = 0
    
    # Weekly equity for curve
    weekly_equity = []
    
    # Process each week
    for rel_week_idx, (week_start, week_end) in enumerate(period_weeks):
        week_start_dt = pd.Timestamp(week_start)
        week_end_dt = pd.Timestamp(week_end)
        
        # Get trading days for this week
        trading_days = [d for d in all_trading_days if week_start_dt <= pd.Timestamp(d) <= week_end_dt]
        
        if not trading_days:
            continue
        
        first_day = trading_days[0]
        last_day_of_week = trading_days[-1]
        
        # Determine regime
        regime_info = regime_by_date.get(first_day, {"is_bull": False})
        is_bull_week = regime_info.get("is_bull", False)
        
        if is_bull_week:
            bull_weeks += 1
            position_multiplier = BULL_MARKET_MULTIPLIER if use_regime_scaling else 1.0
        else:
            bear_weeks += 1
            position_multiplier = 1.0
        
        # 1. Try to fill pending orders
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
                    
                    if is_bull_week:
                        bull_trades += 1
                    else:
                        bear_trades += 1
                    
                    trades.append({
                        "trade_id": next_trade_id,
                        "symbol": symbol,
                        "entry_date": first_day,
                        "entry_week_idx": rel_week_idx,
                        "entry_price": fill_price,
                        "shares": shares,
                        "position_value": position_value,
                        "is_bull_entry": is_bull_week,
                        "is_open": True,
                        "exit_date": None,
                        "exit_price": None,
                        "exit_reason": None,
                        "pnl_dollars": 0,
                        "pnl_pcnt": 0,
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
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
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
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
                    committed_capital -= trade["position_value"]
                    current_capital += trade["pnl_dollars"]
                    break
                
                # Max hold (Friday exit at open)
                is_last_day = (day == last_day_of_week)
                is_max_hold_week = (rel_week_idx >= max_hold_exit_week)
                
                if is_last_day and is_max_hold_week:
                    exit_price = row["open"]
                    trade["exit_date"] = day
                    trade["exit_price"] = exit_price
                    trade["exit_reason"] = "max_hold"
                    trade["is_open"] = False
                    trade["pnl_dollars"] = (exit_price - entry_price) * trade["shares"]
                    trade["pnl_pcnt"] = ((exit_price / entry_price) - 1) * 100
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
        
        if returns_data:
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
        
        # Record weekly equity
        weekly_equity.append({
            "week": rel_week_idx,
            "week_start": week_start,
            "capital": current_capital,
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
                    current_capital += trade["pnl_dollars"]
    
    # Calculate metrics
    closed_trades = [t for t in trades if t["exit_date"] is not None]
    total_trades = len(closed_trades)
    
    if total_trades == 0:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "win_rate": 0, "total_pnl": 0, "total_return_pcnt": 0, 
            "max_drawdown_pcnt": 0, "bull_weeks": bull_weeks, "bear_weeks": bear_weeks,
            "bull_trades": bull_trades, "bear_trades": bear_trades,
            "weekly_equity": weekly_equity,
        }
    
    winning_trades = sum(1 for t in closed_trades if t["pnl_dollars"] > 0)
    losing_trades = sum(1 for t in closed_trades if t["pnl_dollars"] <= 0)
    win_rate = (winning_trades / total_trades) * 100
    total_pnl = sum(t["pnl_dollars"] for t in closed_trades)
    total_return_pcnt = (total_pnl / INITIAL_CAPITAL) * 100
    
    # Max drawdown
    sorted_trades = sorted(closed_trades, key=lambda t: t["exit_date"])
    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0
    
    for trade in sorted_trades:
        capital += trade["pnl_dollars"]
        peak = max(peak, capital)
        dd = (peak - capital) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    # Regime-specific P&L
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
        "final_capital": current_capital,
        "weekly_equity": weekly_equity,
    }


# =============================================================================
# Main
# =============================================================================

def main(source_experiment: str):
    """Run robustness check for configs from the specified source experiment."""
    
    # Setup
    experiment_name = f"exp020_robustness_{source_experiment.replace('exp019_', '')}"
    output_dir = PROJECT_ROOT / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK: {experiment_name}")
    print(f"{'='*60}")
    print(f"Source: {source_experiment}")
    print(f"Test Period: {TEST_PERIOD_WEEKS} weeks")
    print(f"Samples per Regime: {SAMPLES_PER_REGIME}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load configs
    config_path = PROJECT_ROOT / "experiments" / source_experiment / "robustness_configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config_data = json.load(f)
    
    configs = config_data["configs"]
    use_regime_scaling = config_data.get("use_regime_scaling", False)
    
    print(f"📋 Loaded {len(configs)} configurations:")
    for cfg in configs:
        print(f"   - {cfg['name']}: SL={cfg['stop_loss_pcnt']}%, TP={cfg['profit_exit_pcnt']}%, MaxHold={cfg['max_hold_weeks']}w")
    print(f"   Regime scaling: {'Yes (1.10x bull)' if use_regime_scaling else 'No'}\n")
    
    # Load data
    shared_data = load_shared_data()
    daily_data = shared_data["daily"]
    weekly_data = shared_data["weekly"]
    regime_by_date = shared_data["regime_by_date"]
    
    # Get trading weeks and classify by regime
    trading_weeks = get_trading_weeks(weekly_data)
    print(f"\n📅 Total trading weeks: {len(trading_weeks)}")
    print(f"   Date range: {trading_weeks[0][0]} to {trading_weeks[-1][1]}")
    
    week_regimes = classify_weeks_by_regime(trading_weeks, regime_by_date, daily_data)
    print(f"\n📊 Week classification:")
    print(f"   Bull weeks: {len(week_regimes['bull'])}")
    print(f"   Bear weeks: {len(week_regimes['bear'])}")
    
    # Sample start weeks
    start_samples = sample_start_weeks(
        week_regimes,
        len(trading_weeks),
        TEST_PERIOD_WEEKS,
        SAMPLES_PER_REGIME,
        RANDOM_SEED,
    )
    print(f"\n🎯 Total start samples: {len(start_samples)}")
    
    # Run backtests
    results = []
    equity_curves = []
    
    total_runs = len(configs) * len(start_samples)
    run_count = 0
    
    print(f"\n🚀 Running {total_runs} backtests...\n")
    
    for cfg in configs:
        for sample in start_samples:
            run_count += 1
            start_week_idx = sample["week_idx"]
            end_week_idx = start_week_idx + TEST_PERIOD_WEEKS
            start_date = trading_weeks[start_week_idx][0]
            
            # Run backtest
            result = run_backtest_for_period(
                daily_data=daily_data,
                weekly_data=weekly_data,
                regime_by_date=regime_by_date,
                trading_weeks=trading_weeks,
                start_week_idx=start_week_idx,
                end_week_idx=end_week_idx,
                stop_loss_pcnt=cfg["stop_loss_pcnt"],
                profit_exit_pcnt=cfg["profit_exit_pcnt"],
                max_hold_weeks=cfg["max_hold_weeks"],
                use_regime_scaling=use_regime_scaling,
            )
            
            # Store result
            run_id = f"{cfg['name']}_{start_date}"
            
            results.append({
                "run_id": run_id,
                "config_name": cfg["name"],
                "stop_loss_pcnt": cfg["stop_loss_pcnt"],
                "profit_exit_pcnt": cfg["profit_exit_pcnt"],
                "max_hold_weeks": cfg["max_hold_weeks"],
                "start_week_idx": start_week_idx,
                "start_date": start_date,
                "start_regime": sample["start_regime"],
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
                "bull_trade_pnl": result.get("bull_trade_pnl", 0),
                "bear_trade_pnl": result.get("bear_trade_pnl", 0),
                "final_capital": result.get("final_capital", INITIAL_CAPITAL),
            })
            
            # Store equity curve
            for eq in result.get("weekly_equity", []):
                equity_curves.append({
                    "run_id": run_id,
                    "config_name": cfg["name"],
                    "start_regime": sample["start_regime"],
                    "week": eq["week"],
                    "week_start": eq["week_start"],
                    "capital": eq["capital"],
                })
            
            if run_count % 10 == 0:
                print(f"   Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.0f}%)")
    
    print(f"\n✅ Completed all {total_runs} runs")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "robustness_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n📊 Results saved to: {results_path}")
    
    equity_df = pd.DataFrame(equity_curves)
    equity_path = output_dir / "equity_curves.csv"
    equity_df.to_csv(equity_path, index=False)
    print(f"📈 Equity curves saved to: {equity_path}")
    
    # Save metadata
    metadata = {
        "experiment_name": experiment_name,
        "source_experiment": source_experiment,
        "generated_at": datetime.now().isoformat(),
        "test_period_weeks": TEST_PERIOD_WEEKS,
        "samples_per_regime": SAMPLES_PER_REGIME,
        "total_runs": total_runs,
        "configs_tested": [cfg["name"] for cfg in configs],
        "use_regime_scaling": use_regime_scaling,
        "date_range": {
            "first_week": trading_weeks[0][0],
            "last_week": trading_weeks[-1][1],
        },
    }
    
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Metadata saved to: {metadata_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY BY CONFIGURATION")
    print(f"{'='*60}")
    
    for cfg_name in results_df["config_name"].unique():
        cfg_results = results_df[results_df["config_name"] == cfg_name]
        
        print(f"\n{cfg_name}:")
        print(f"   Avg Return: {cfg_results['total_return_pcnt'].mean():.1f}% "
              f"(std: {cfg_results['total_return_pcnt'].std():.1f}%)")
        print(f"   Return Range: {cfg_results['total_return_pcnt'].min():.1f}% to "
              f"{cfg_results['total_return_pcnt'].max():.1f}%")
        print(f"   Avg Drawdown: {cfg_results['max_drawdown_pcnt'].mean():.1f}%")
        print(f"   Avg Win Rate: {cfg_results['win_rate'].mean():.1f}%")
        
        # By starting regime
        bull_starts = cfg_results[cfg_results["start_regime"] == "bull"]
        bear_starts = cfg_results[cfg_results["start_regime"] == "bear"]
        
        print(f"   Bull starts ({len(bull_starts)}): avg {bull_starts['total_return_pcnt'].mean():.1f}%")
        print(f"   Bear starts ({len(bear_starts)}): avg {bear_starts['total_return_pcnt'].mean():.1f}%")
    
    print(f"\n{'='*60}")
    print(f"Run 20b-analyze-robustness.py --source {source_experiment} for visualization")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main(SOURCE_EXPERIMENT)

#!/usr/bin/env python3
"""
16-run-exit-rule-backtest.py

Portfolio backtest with exit rules instead of fixed holding periods:
- Entry: Buy top-K losers from previous week (mean-reversion signal)
- Exit: Take-profit OR stop-loss OR max-hold timeout
- Max 20 concurrent positions
- Zero transaction costs (for now)

This tests whether letting trades run to targets improves over forced rebalancing.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp016_exit_rules"
MIN_HISTORY_WEEKS = 52

# Feature matrix path
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Position limits
MAX_POSITIONS = 20
NEW_ENTRIES_PER_WEEK = 5  # Max new positions to consider each week

# Exit rule configurations to test
EXIT_CONFIGS = [
    {"take_profit": 0.03, "stop_loss": -0.03, "max_hold_weeks": 52, "name": "TP3_SL3"},
    {"take_profit": 0.05, "stop_loss": -0.05, "max_hold_weeks": 52, "name": "TP5_SL5"},
    {"take_profit": 0.10, "stop_loss": -0.05, "max_hold_weeks": 52, "name": "TP10_SL5"},
    {"take_profit": 0.10, "stop_loss": -0.10, "max_hold_weeks": 52, "name": "TP10_SL10"},
    {"take_profit": 0.05, "stop_loss": -0.03, "max_hold_weeks": 12, "name": "TP5_SL3_12w"},
    {"take_profit": None, "stop_loss": -0.05, "max_hold_weeks": 8, "name": "SL5_8w"},  # No TP, just stop
]


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    entry_week_idx: int
    entry_price: float  # Actually log of price, but we track returns
    cumulative_return: float = 0.0
    weeks_held: int = 0


def load_data() -> pd.DataFrame:
    """Load feature matrix."""
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[df["category"] == "target"].copy()
    return df


def get_entry_candidates(
    df: pd.DataFrame,
    week_idx: int,
    n_candidates: int,
    existing_symbols: set,
) -> list[str]:
    """
    Get entry candidates for a given week.
    
    Returns symbols with worst returns from previous week,
    excluding those already in portfolio.
    """
    # Get prior week's returns
    prior_week = df[df["week_idx"] == week_idx - 1][["symbol", "log_return"]].copy()
    
    # Exclude existing positions
    prior_week = prior_week[~prior_week["symbol"].isin(existing_symbols)]
    
    # Sort by return (ascending = worst first)
    prior_week = prior_week.sort_values("log_return").head(n_candidates)
    
    return prior_week["symbol"].tolist()


def run_backtest_with_exit_rules(
    df: pd.DataFrame,
    take_profit: Optional[float],
    stop_loss: Optional[float],
    max_hold_weeks: int,
) -> pd.DataFrame:
    """
    Run backtest with exit rules.
    
    Args:
        take_profit: Exit when cumulative return >= this (None = no TP)
        stop_loss: Exit when cumulative return <= this (None = no SL)
        max_hold_weeks: Force exit after this many weeks
    
    Returns:
        DataFrame of all trades
    """
    week_indices = sorted(df["week_idx"].unique())
    prediction_weeks = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    # Build price lookup (using log_return to track returns)
    returns_by_week = {}
    for week_idx in week_indices:
        week_data = df[df["week_idx"] == week_idx][["symbol", "log_return"]]
        returns_by_week[week_idx] = dict(zip(week_data["symbol"], week_data["log_return"]))
    
    # Track state
    open_positions: list[Position] = []
    closed_trades = []
    
    for week_idx in prediction_weeks:
        # 1. Update existing positions
        positions_to_close = []
        
        for pos in open_positions:
            # Get this week's return for the symbol
            week_return = returns_by_week.get(week_idx, {}).get(pos.symbol, 0.0)
            pos.cumulative_return += week_return
            pos.weeks_held += 1
            
            # Check exit conditions
            exit_reason = None
            
            if take_profit is not None and pos.cumulative_return >= take_profit:
                exit_reason = "take_profit"
            elif stop_loss is not None and pos.cumulative_return <= stop_loss:
                exit_reason = "stop_loss"
            elif pos.weeks_held >= max_hold_weeks:
                exit_reason = "max_hold"
            
            if exit_reason:
                positions_to_close.append((pos, exit_reason, week_idx))
        
        # Close positions
        for pos, reason, exit_week in positions_to_close:
            closed_trades.append({
                "symbol": pos.symbol,
                "entry_week_idx": pos.entry_week_idx,
                "exit_week_idx": exit_week,
                "weeks_held": pos.weeks_held,
                "return": pos.cumulative_return,
                "exit_reason": reason,
            })
            open_positions.remove(pos)
        
        # 2. Open new positions if we have capacity
        capacity = MAX_POSITIONS - len(open_positions)
        if capacity > 0:
            existing_symbols = {p.symbol for p in open_positions}
            candidates = get_entry_candidates(
                df, week_idx, 
                min(NEW_ENTRIES_PER_WEEK, capacity),
                existing_symbols
            )
            
            for symbol in candidates:
                if len(open_positions) >= MAX_POSITIONS:
                    break
                
                open_positions.append(Position(
                    symbol=symbol,
                    entry_week_idx=week_idx,
                    entry_price=0.0,  # Not tracking actual price
                ))
    
    # Close any remaining open positions at end
    final_week = prediction_weeks[-1]
    for pos in open_positions:
        closed_trades.append({
            "symbol": pos.symbol,
            "entry_week_idx": pos.entry_week_idx,
            "exit_week_idx": final_week,
            "weeks_held": pos.weeks_held,
            "return": pos.cumulative_return,
            "exit_reason": "end_of_data",
        })
    
    return pd.DataFrame(closed_trades)


def analyze_trades(trades_df: pd.DataFrame, config_name: str) -> dict:
    """Analyze trade statistics."""
    if len(trades_df) == 0:
        return {"config": config_name, "n_trades": 0}
    
    # Basic stats
    n_trades = len(trades_df)
    win_rate = (trades_df["return"] > 0).mean()
    avg_return = trades_df["return"].mean()
    median_return = trades_df["return"].median()
    avg_hold = trades_df["weeks_held"].mean()
    
    # Return distribution
    total_return = trades_df["return"].sum()
    
    # By exit reason
    exit_counts = trades_df["exit_reason"].value_counts().to_dict()
    
    # Annualized (rough estimate)
    # Total weeks in backtest
    total_weeks = trades_df["exit_week_idx"].max() - trades_df["entry_week_idx"].min()
    years = total_weeks / 52 if total_weeks > 0 else 1
    ann_return = total_return / years if years > 0 else 0
    
    # Trades per year
    trades_per_year = n_trades / years if years > 0 else 0
    
    return {
        "config": config_name,
        "n_trades": n_trades,
        "trades_per_year": trades_per_year,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "total_return": total_return,
        "ann_return": ann_return,
        "avg_hold_weeks": avg_hold,
        **{f"exit_{k}": v for k, v in exit_counts.items()},
    }


def generate_report(
    all_results: dict,
    all_trades: dict,
    output_dir: Path,
) -> None:
    """Generate experiment report."""
    report_lines = [
        f"# Exit Rule Backtest Report: {EXPERIMENT_NAME}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Strategy",
        "",
        "**Mean-Reversion with Exit Rules:**",
        f"- Entry: Buy top-{NEW_ENTRIES_PER_WEEK} losers from prior week",
        f"- Max concurrent positions: {MAX_POSITIONS}",
        "- Exit: Take-profit OR stop-loss OR max-hold timeout",
        "- Transaction costs: **ZERO** (for this analysis)",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Config | Trades | Trades/Yr | Win Rate | Avg Return | Ann. Return | Avg Hold |",
        "|--------|--------|-----------|----------|------------|-------------|----------|",
    ]
    
    for config in EXIT_CONFIGS:
        name = config["name"]
        if name in all_results:
            r = all_results[name]
            report_lines.append(
                f"| {name} | {r['n_trades']} | {r['trades_per_year']:.0f} | "
                f"{r['win_rate']:.1%} | {r['avg_return']:.2%} | {r['ann_return']:.1%} | "
                f"{r['avg_hold_weeks']:.1f}w |"
            )
    
    # Exit reason breakdown
    report_lines.extend([
        "",
        "---",
        "",
        "## Exit Reason Breakdown",
        "",
        "| Config | Take Profit | Stop Loss | Max Hold | End of Data |",
        "|--------|-------------|-----------|----------|-------------|",
    ])
    
    for config in EXIT_CONFIGS:
        name = config["name"]
        if name in all_results:
            r = all_results[name]
            tp = r.get("exit_take_profit", 0)
            sl = r.get("exit_stop_loss", 0)
            mh = r.get("exit_max_hold", 0)
            eod = r.get("exit_end_of_data", 0)
            total = tp + sl + mh + eod
            
            if total > 0:
                report_lines.append(
                    f"| {name} | {tp} ({tp/total:.0%}) | {sl} ({sl/total:.0%}) | "
                    f"{mh} ({mh/total:.0%}) | {eod} ({eod/total:.0%}) |"
                )
    
    # Return distribution by exit type
    report_lines.extend([
        "",
        "---",
        "",
        "## Return by Exit Type",
        "",
        "| Config | TP Avg | SL Avg | Max Hold Avg |",
        "|--------|--------|--------|--------------|",
    ])
    
    for config in EXIT_CONFIGS:
        name = config["name"]
        if name in all_trades:
            trades = all_trades[name]
            if len(trades) > 0:
                tp_trades = trades[trades["exit_reason"] == "take_profit"]
                sl_trades = trades[trades["exit_reason"] == "stop_loss"]
                mh_trades = trades[trades["exit_reason"] == "max_hold"]
                
                tp_avg = tp_trades["return"].mean() if len(tp_trades) > 0 else 0
                sl_avg = sl_trades["return"].mean() if len(sl_trades) > 0 else 0
                mh_avg = mh_trades["return"].mean() if len(mh_trades) > 0 else 0
                
                report_lines.append(
                    f"| {name} | {tp_avg:.2%} | {sl_avg:.2%} | {mh_avg:.2%} |"
                )
    
    # Best configuration
    report_lines.extend([
        "",
        "---",
        "",
        "## Conclusions",
        "",
    ])
    
    # Find best by annualized return
    best_config = max(all_results.values(), key=lambda x: x.get("ann_return", -999))
    
    report_lines.append(f"**Best Configuration:** `{best_config['config']}`")
    report_lines.append(f"- Annualized Return: {best_config['ann_return']:.1%}")
    report_lines.append(f"- Win Rate: {best_config['win_rate']:.1%}")
    report_lines.append(f"- Trades/Year: {best_config['trades_per_year']:.0f}")
    
    # Comparison to fixed holding
    report_lines.extend([
        "",
        "### Comparison to Fixed Holding (exp015)",
        "",
        "| Approach | Ann. Return (gross) | Trades/Yr |",
        "|----------|---------------------|-----------|",
        "| Fixed 1-week, K=5 | +11.7% | 520 |",
        "| Fixed 4-week, K=5 | +11.1% | 130 |",
        f"| Best Exit Rule | {best_config['ann_return']:.1%} | {best_config['trades_per_year']:.0f} |",
    ])
    
    report_lines.append("")
    report_lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    logging.info(f"Report saved: {report_path}")


@workflow_script("16-run-exit-rule-backtest")
def main() -> None:
    """Run exit rule backtest."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    df = load_data()
    logging.info(f"Feature matrix: {len(df)} rows, {df['week_idx'].nunique()} weeks")
    
    # Run all configurations
    all_results = {}
    all_trades = {}
    
    for config in EXIT_CONFIGS:
        name = config["name"]
        logging.info(f"Running {name}...")
        
        trades_df = run_backtest_with_exit_rules(
            df,
            take_profit=config["take_profit"],
            stop_loss=config["stop_loss"],
            max_hold_weeks=config["max_hold_weeks"],
        )
        
        # Save trades
        trades_df.to_csv(OUTPUT_DIR / f"trades_{name}.csv", index=False)
        all_trades[name] = trades_df
        
        # Analyze
        results = analyze_trades(trades_df, name)
        all_results[name] = results
        
        logging.info(f"  {name}: {results['n_trades']} trades, "
                    f"win={results['win_rate']:.1%}, ann={results['ann_return']:.1%}")
    
    # Save summary
    summary_df = pd.DataFrame(list(all_results.values()))
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    
    # Save config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "max_positions": MAX_POSITIONS,
        "new_entries_per_week": NEW_ENTRIES_PER_WEEK,
        "exit_configs": EXIT_CONFIGS,
        "generated": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate report
    generate_report(all_results, all_trades, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"EXIT RULE BACKTEST: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    print("\nResults by Configuration:")
    print("-" * 70)
    print(f"{'Config':<15} {'Trades':>7} {'Win%':>7} {'AvgRet':>8} {'AnnRet':>8} {'Hold':>6}")
    print("-" * 70)
    
    for config in EXIT_CONFIGS:
        name = config["name"]
        if name in all_results:
            r = all_results[name]
            print(f"{name:<15} {r['n_trades']:>7} {r['win_rate']:>6.1%} "
                  f"{r['avg_return']:>7.2%} {r['ann_return']:>7.1%} {r['avg_hold_weeks']:>5.1f}w")
    
    print("=" * 70)
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()

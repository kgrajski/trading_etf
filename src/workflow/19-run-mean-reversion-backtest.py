#!/usr/bin/env python3
"""
19-run-mean-reversion-backtest.py

Phase II: Mean-reversion backtesting engine.

Simulates a mean-reversion strategy:
- Entry: Buy bottom 5% weekly losers with loss >= MIN_LOSS_PCNT
- Limit order at prior week's close (day-only on Monday)
- Exit: Stop-loss, profit-target, or max-hold timeout
- Capital: Fixed pool, losses reduce capacity

Reads from:
- data/historical/{tier}/daily/*.csv (from 02-fetch-daily-data.py)
- data/historical/{tier}/weekly/*.csv (from 03-generate-features.py)

Outputs:
- experiments/exp019_*/trades.csv
- experiments/exp019_*/weekly_summary.csv
- experiments/exp019_*/dashboard.html
- experiments/exp019_*/config.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.backtesting.mean_reversion_backtester import (
    BacktestConfig,
    BacktestResult,
    MeanReversionBacktester,
    Trade,
)
from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp019_mean_reversion_backtest"

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Default hyperparameters (optimized from grid search 19d)
DEFAULT_CONFIG = BacktestConfig(
    initial_capital=10_000.0,
    max_active_trades=10,
    bottom_percentile=0.05,
    min_loss_pcnt=2.0,
    stop_loss_pcnt=16.0,  # Optimized: wide stop, let trades breathe
    profit_exit_pcnt=10.0,  # Optimized: take 10% profits
    max_hold_weeks=1,  # Optimized: quick exit Friday of entry week
)

# Hyperparameter grid for comparison (optional)
PARAM_GRID = {
    "stop_loss_pcnt": [5.0, 10.0, 15.0],
    "profit_exit_pcnt": [5.0, 10.0, 15.0, 20.0],
    "max_hold_weeks": [4, 8, 12],
}


def get_target_symbols() -> List[str]:
    """Get list of target ETF symbols (exclude macro symbols)."""
    filtered_etfs_path = METADATA_DIR / "filtered_etfs.json"
    
    if not filtered_etfs_path.exists():
        # Fallback: use all symbols in weekly dir
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    
    with open(filtered_etfs_path) as f:
        data = json.load(f)
    
    return [etf["symbol"] for etf in data.get("etfs", [])]


def create_dashboard(result: BacktestResult, output_dir: Path) -> Path:
    """
    Create interactive dashboard for backtest results.
    
    Includes:
    - Equity curve with drawdown
    - Trade log with hover data
    - Weekly summary
    - Performance metrics
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.35, 0.20, 0.25, 0.20],
        subplot_titles=(
            "Equity Curve",
            "Drawdown",
            "Weekly P&L",
            "Active Positions",
        ),
    )
    
    equity_df = result.equity_curve
    
    if len(equity_df) > 0:
        # Row 1: Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df["date"],
                y=equity_df["capital"],
                mode="lines+markers",
                name="Capital",
                line=dict(color="#2E86AB", width=2),
                marker=dict(size=6),
                customdata=np.column_stack([
                    equity_df["symbol"],
                    equity_df["pnl_dollars"],
                    equity_df["pnl_pcnt"],
                    equity_df["cumulative_return_pcnt"],
                ]),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Capital: $%{y:,.2f}<br>"
                    "Trade: %{customdata[0]}<br>"
                    "P&L: $%{customdata[1]:.2f} (%{customdata[2]:.1f}%)<br>"
                    "Cumulative: %{customdata[3]:.1f}%<br>"
                    "<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        
        # Starting capital line
        fig.add_hline(
            y=result.config.initial_capital,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=1, col=1,
        )
        
        # Row 2: Drawdown
        fig.add_trace(
            go.Scatter(
                x=equity_df["date"],
                y=-equity_df["drawdown_pcnt"],
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#DC3545", width=1),
                fillcolor="rgba(220, 53, 69, 0.3)",
                hovertemplate="Drawdown: %{y:.1f}%<extra></extra>",
            ),
            row=2, col=1,
        )
        
        # Row 3: Weekly P&L bars
        weekly_pnl = equity_df.groupby("date")["pnl_dollars"].sum().reset_index()
        colors = ["#28A745" if v > 0 else "#DC3545" for v in weekly_pnl["pnl_dollars"]]
        
        fig.add_trace(
            go.Bar(
                x=weekly_pnl["date"],
                y=weekly_pnl["pnl_dollars"],
                name="Weekly P&L",
                marker_color=colors,
                hovertemplate="P&L: $%{y:.2f}<extra></extra>",
            ),
            row=3, col=1,
        )
    
    # Row 4: Active positions from weekly summaries
    if result.weekly_summaries:
        summary_df = pd.DataFrame([{
            "week_start": s.week_start,
            "active_positions": s.active_positions,
            "new_entries": s.new_entries,
            "exits": s.exits_stop_loss + s.exits_profit + s.exits_timeout,
        } for s in result.weekly_summaries])
        
        fig.add_trace(
            go.Scatter(
                x=summary_df["week_start"],
                y=summary_df["active_positions"],
                mode="lines+markers",
                name="Active Positions",
                line=dict(color="#6F42C1", width=2),
                marker=dict(size=4),
                hovertemplate="Positions: %{y}<extra></extra>",
            ),
            row=4, col=1,
        )
    
    # Build title with metrics
    config = result.config
    title_text = (
        f"<b>Mean-Reversion Backtest: {EXPERIMENT_NAME}</b><br>"
        f"<sup>Capital: ${config.initial_capital:,.0f} | "
        f"Trades: {result.total_trades} | "
        f"Win Rate: {result.win_rate:.1f}% | "
        f"Total P&L: ${result.total_pnl:,.2f} ({result.total_return_pcnt:.1f}%) | "
        f"Max DD: {result.max_drawdown_pcnt:.1f}%</sup><br>"
        f"<sup>SL: {config.stop_loss_pcnt}% | TP: {config.profit_exit_pcnt}% | "
        f"MaxHold: {config.max_hold_weeks}w | MinLoss: {config.min_loss_pcnt}%</sup>"
    )
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor="center"),
        height=1000,
        showlegend=False,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=120, b=40, l=60, r=60),
    )
    
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_yaxes(title_text="Positions", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    output_path = output_dir / "dashboard.html"
    fig.write_html(output_path)
    
    return output_path


def create_trade_inspector(result: BacktestResult, output_dir: Path) -> Path:
    """Create HTML table of all trades with details."""
    trades = [t for t in result.trades if t.entry_date is not None]
    
    if not trades:
        return None
    
    # Build table rows
    table_rows = []
    for trade in sorted(trades, key=lambda t: t.entry_date or ""):
        pnl_class = "positive" if trade.pnl_dollars > 0 else "negative"
        reason_class = {
            "stop_loss": "stop",
            "profit_exit": "profit",
            "max_hold": "timeout",
            "end_of_data": "end",
        }.get(trade.exit_reason, "")
        
        entry_price_str = f"${trade.entry_price:.2f}" if trade.entry_price else "N/A"
        exit_price_str = f"${trade.exit_price:.2f}" if trade.exit_price else "N/A"
        
        table_rows.append(f"""
        <tr class="{pnl_class}">
            <td>{trade.trade_id}</td>
            <td><strong>{trade.symbol}</strong></td>
            <td>{trade.signal_week_start}</td>
            <td>{trade.weekly_return_pcnt:.2f}%</td>
            <td>{trade.entry_date or 'N/A'}</td>
            <td>{entry_price_str}</td>
            <td>{trade.shares}</td>
            <td>{trade.exit_date or 'N/A'}</td>
            <td>{exit_price_str}</td>
            <td class="{reason_class}">{trade.exit_reason or 'N/A'}</td>
            <td class="{pnl_class}">${trade.pnl_dollars:.2f}</td>
            <td class="{pnl_class}">{trade.pnl_pcnt:.1f}%</td>
            <td>{trade.hold_days}</td>
        </tr>
        """)
    
    config = result.config
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Trade Log - {EXPERIMENT_NAME}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; }}
        .summary {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric {{
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2E86AB;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: #2E86AB;
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-size: 12px;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
            font-size: 13px;
        }}
        tr:hover {{
            background: #f0f8ff;
        }}
        .positive {{ color: #28A745; }}
        .negative {{ color: #DC3545; }}
        .stop {{ background: #ffebee; }}
        .profit {{ background: #e8f5e9; }}
        .timeout {{ background: #fff3e0; }}
        .end {{ background: #f3e5f5; }}
        a {{ color: #2E86AB; }}
    </style>
</head>
<body>
    <h1>Trade Log: {EXPERIMENT_NAME}</h1>
    
    <div class="summary">
        <div class="summary-grid">
            <div class="metric">
                <div class="metric-value">{result.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if result.total_pnl > 0 else 'negative'}">${result.total_pnl:,.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.total_return_pcnt:.1f}%</div>
                <div class="metric-label">Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.max_drawdown_pcnt:.1f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
        <p style="margin-top:15px;font-size:12px;color:#666;">
            Config: SL={config.stop_loss_pcnt}% | TP={config.profit_exit_pcnt}% | 
            MaxHold={config.max_hold_weeks}w | MinLoss={config.min_loss_pcnt}%
        </p>
    </div>
    
    <p><a href="dashboard.html">← Back to Dashboard</a></p>
    
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Symbol</th>
                <th>Signal Week</th>
                <th>Week Return</th>
                <th>Entry Date</th>
                <th>Entry Price</th>
                <th>Shares</th>
                <th>Exit Date</th>
                <th>Exit Price</th>
                <th>Exit Reason</th>
                <th>P&L $</th>
                <th>P&L %</th>
                <th>Days</th>
            </tr>
        </thead>
        <tbody>
            {"".join(table_rows)}
        </tbody>
    </table>
</body>
</html>
"""
    
    output_path = output_dir / "trades.html"
    output_path.write_text(html_content)
    
    return output_path


def save_results(result: BacktestResult, output_dir: Path) -> None:
    """Save backtest results to files."""
    
    # Trades CSV
    trades_data = []
    for t in result.trades:
        if t.entry_date is None:
            continue
        trades_data.append({
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "signal_week": t.signal_week_start,
            "weekly_return_pcnt": t.weekly_return_pcnt,
            "entry_date": t.entry_date,
            "entry_price": t.entry_price,
            "shares": t.shares,
            "position_value": t.position_value,
            "exit_date": t.exit_date,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason,
            "pnl_dollars": t.pnl_dollars,
            "pnl_pcnt": t.pnl_pcnt,
            "hold_days": t.hold_days,
        })
    
    trades_df = pd.DataFrame(trades_data)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    
    # Weekly summary CSV
    summary_data = []
    for s in result.weekly_summaries:
        summary_data.append({
            "week_start": s.week_start,
            "week_end": s.week_end,
            "n_candidates": s.n_candidates,
            "n_qualified": s.n_qualified,
            "new_entries": s.new_entries,
            "exits_stop_loss": s.exits_stop_loss,
            "exits_profit": s.exits_profit,
            "exits_timeout": s.exits_timeout,
            "active_positions": s.active_positions,
            "committed_capital": s.committed_capital,
            "available_capital": s.available_capital,
            "total_capital": s.total_capital,
            "realized_pnl_week": s.realized_pnl_week,
            "cumulative_realized_pnl": s.cumulative_realized_pnl,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "weekly_summary.csv", index=False)
    
    # Equity curve CSV
    if len(result.equity_curve) > 0:
        result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    
    # Config JSON
    config_dict = {
        "experiment_name": EXPERIMENT_NAME,
        "generated_at": datetime.now().isoformat(),
        "config": {
            "initial_capital": result.config.initial_capital,
            "max_active_trades": result.config.max_active_trades,
            "bottom_percentile": result.config.bottom_percentile,
            "min_loss_pcnt": result.config.min_loss_pcnt,
            "stop_loss_pcnt": result.config.stop_loss_pcnt,
            "profit_exit_pcnt": result.config.profit_exit_pcnt,
            "max_hold_weeks": result.config.max_hold_weeks,
        },
        "metrics": {
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "total_return_pcnt": result.total_return_pcnt,
            "max_drawdown_pcnt": result.max_drawdown_pcnt,
        },
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


@workflow_script("19-run-mean-reversion-backtest")
def main() -> None:
    """Run mean-reversion backtest."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get target symbols
    symbols = get_target_symbols()
    logging.info(f"Found {len(symbols)} target symbols")
    
    # Create and configure backtester
    config = DEFAULT_CONFIG
    backtester = MeanReversionBacktester(config)
    
    # Load data
    logging.info("Loading data...")
    backtester.load_data(DAILY_DATA_DIR, WEEKLY_DATA_DIR, symbols)
    
    # Run backtest
    logging.info("Running backtest...")
    result = backtester.run()
    
    # Save results
    logging.info("Saving results...")
    save_results(result, OUTPUT_DIR)
    
    # Create dashboard
    logging.info("Creating dashboard...")
    dashboard_path = create_dashboard(result, OUTPUT_DIR)
    
    # Create trade inspector
    trade_inspector_path = create_trade_inspector(result, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"MEAN-REVERSION BACKTEST: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Initial Capital:    ${config.initial_capital:,.0f}")
    print(f"  Max Active Trades:  {config.max_active_trades}")
    print(f"  Stop Loss:          {config.stop_loss_pcnt}%")
    print(f"  Profit Target:      {config.profit_exit_pcnt}%")
    print(f"  Max Hold:           {config.max_hold_weeks} weeks")
    print(f"  Min Loss Threshold: {config.min_loss_pcnt}%")
    
    print(f"\nResults:")
    print(f"  Total Trades:       {result.total_trades}")
    print(f"  Winning Trades:     {result.winning_trades}")
    print(f"  Losing Trades:      {result.losing_trades}")
    print(f"  Win Rate:           {result.win_rate:.1f}%")
    print(f"  Total P&L:          ${result.total_pnl:,.2f}")
    print(f"  Return:             {result.total_return_pcnt:.1f}%")
    print(f"  Max Drawdown:       {result.max_drawdown_pcnt:.1f}%")
    
    # Exit reason breakdown
    if result.trades:
        exit_counts = {}
        for t in result.trades:
            if t.exit_reason:
                exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
        
        print(f"\nExit Reasons:")
        for reason, count in sorted(exit_counts.items()):
            print(f"  {reason}: {count}")
    
    print(f"\nOutput Files:")
    print(f"  Dashboard:     {dashboard_path}")
    print(f"  Trade Log:     {trade_inspector_path}")
    print(f"  Trades CSV:    {OUTPUT_DIR / 'trades.csv'}")
    print(f"  Weekly CSV:    {OUTPUT_DIR / 'weekly_summary.csv'}")
    
    print("=" * 70)
    logging.info("Backtest complete")


if __name__ == "__main__":
    main()

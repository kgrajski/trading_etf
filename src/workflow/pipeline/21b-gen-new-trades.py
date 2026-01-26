#!/usr/bin/env python3
"""
21-generate-trade-candidates.py

Generate actionable trade candidates for the upcoming week.

Uses optimized mean-reversion strategy parameters from config.
Produces inspector HTML with trade specs (shares, limit price, stop, target).

Outputs to pre_production/{date}/:
- _inspector.html (main view - opens automatically)
- candidates.csv
- summary.json
- charts/{SYMBOL}.html
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, workflow_script

# =============================================================================
# Configuration
# =============================================================================

# Data paths
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# Output
PRE_PRODUCTION_DIR = PROJECT_ROOT / "pre_production"
CONFIG_PATH = PRE_PRODUCTION_DIR / "config.json"

# Chart parameters
BB_PERIOD = 20
BB_STD = 2
CHART_LOOKBACK_DAYS = 252

# Default strategy parameters (used if config.json doesn't exist)
DEFAULT_CONFIG = {
    "strategy_version": "v1.0",
    "stop_loss_pcnt": 16.0,
    "profit_exit_pcnt": 10.0,
    "max_hold_weeks": 1,
    "min_loss_pcnt": 2.0,
    "bottom_percentile": 0.05,
    "initial_capital": 10000.0,
    "max_active_trades": 10,
    "use_bear_boost": True,
    "bear_market_multiplier": 1.10,
    "regime_symbol": "SPY",
    "regime_ma_period": 50,
}


def load_config() -> Dict:
    """Load strategy config, creating default if needed."""
    PRE_PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    else:
        # Create default config
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"Created default config: {CONFIG_PATH}")
        return DEFAULT_CONFIG


def get_current_regime(config: Dict) -> Dict:
    """Determine current market regime."""
    symbol = config.get("regime_symbol", "SPY")
    ma_period = config.get("regime_ma_period", 50)
    
    daily_path = DAILY_DATA_DIR / f"{symbol}.csv"
    
    if not daily_path.exists():
        return {"is_bull": True, "regime": "unknown"}
    
    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.sort_values("date").tail(ma_period + 10)
    df["ma"] = df["close"].rolling(window=ma_period).mean()
    
    latest = df.iloc[-1]
    is_bull = latest["close"] > latest["ma"]
    
    return {
        "is_bull": bool(is_bull),
        "regime": "bull" if is_bull else "bear",
        "close": float(latest["close"]),
        "ma": float(latest["ma"]),
        "date": latest["date"].strftime("%Y-%m-%d"),
    }


def get_latest_week_data() -> Tuple[pd.DataFrame, str]:
    """Get latest week returns with close prices."""
    # Load feature matrix
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    target_df = df[df["category"] == "target"].copy()
    
    # Latest week
    latest_week_idx = target_df["week_idx"].max()
    latest_week = target_df[target_df["week_idx"] == latest_week_idx].copy()
    
    # Calculate pct return
    latest_week["pct_return"] = (np.exp(latest_week["log_return"]) - 1) * 100
    
    # Get week start
    week_start = latest_week["week_start"].iloc[0]
    week_str = week_start.strftime("%Y-%m-%d") if hasattr(week_start, "strftime") else str(week_start)[:10]
    
    # Get close prices from weekly data
    close_prices = {}
    for symbol in latest_week["symbol"].unique():
        weekly_path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if weekly_path.exists():
            wdf = pd.read_csv(weekly_path, parse_dates=["week_start"])
            row = wdf[wdf["week_start"] == week_start]
            if len(row) > 0:
                close_prices[symbol] = row["close"].iloc[0]
    
    latest_week["close"] = latest_week["symbol"].map(close_prices)
    latest_week = latest_week.dropna(subset=["close"])
    
    return latest_week, week_str


def get_etf_names() -> Dict[str, str]:
    """Load ETF names from metadata."""
    path = METADATA_DIR / "filtered_etfs.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {etf["symbol"]: etf.get("name", etf["symbol"]) for etf in data.get("etfs", [])}


def identify_candidates(df: pd.DataFrame, config: Dict, regime: Dict) -> pd.DataFrame:
    """Identify and size trade candidates."""
    # Sort by worst return
    sorted_df = df.sort_values("pct_return")
    
    # Bottom percentile
    n = max(1, int(len(sorted_df) * config["bottom_percentile"]))
    bottom = sorted_df.head(n)
    
    # Filter by min loss
    candidates = bottom[bottom["pct_return"] <= -config["min_loss_pcnt"]].copy()
    
    if len(candidates) == 0:
        return candidates
    
    # Add names
    names = get_etf_names()
    candidates["etf_name"] = candidates["symbol"].map(lambda s: names.get(s, s))
    
    # Position sizing
    is_bear = not regime.get("is_bull", True)
    use_boost = config.get("use_bear_boost", True)
    multiplier = config.get("bear_market_multiplier", 1.10) if (is_bear and use_boost) else 1.0
    
    base_size = config["initial_capital"] / config["max_active_trades"]
    target_size = base_size * multiplier
    
    candidates["multiplier"] = multiplier
    candidates["target_usd"] = target_size
    
    # Trade specs
    sl = config["stop_loss_pcnt"]
    tp = config["profit_exit_pcnt"]
    
    candidates["limit_price"] = candidates["close"].round(2)
    candidates["stop_price"] = (candidates["close"] * (1 - sl / 100)).round(2)
    candidates["target_price"] = (candidates["close"] * (1 + tp / 100)).round(2)
    candidates["shares"] = (target_size / candidates["close"]).astype(int)
    candidates["position_usd"] = (candidates["shares"] * candidates["close"]).round(2)
    
    return candidates


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add Bollinger Band columns."""
    df = df.copy()
    df["bb_middle"] = df["close"].rolling(window=BB_PERIOD).mean()
    df["bb_std"] = df["close"].rolling(window=BB_PERIOD).std()
    df["bb_upper"] = df["bb_middle"] + BB_STD * df["bb_std"]
    df["bb_lower"] = df["bb_middle"] - BB_STD * df["bb_std"]
    return df


def create_chart(
    symbol: str,
    name: str,
    trade: pd.Series,
    output_dir: Path,
) -> Path:
    """Create candlestick + Bollinger chart with trade levels."""
    daily_path = DAILY_DATA_DIR / f"{symbol}.csv"
    
    if not daily_path.exists():
        return None
    
    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.sort_values("date").tail(CHART_LOOKBACK_DAYS)
    df = compute_bollinger_bands(df)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    
    # Bollinger Bands fill
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["bb_upper"], df["bb_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(128,128,128,0.4)", width=1),
            name="BB",
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    
    # BB middle
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["bb_middle"],
            line=dict(color="gray", width=1, dash="dash"),
            name="MA20",
        ),
        row=1, col=1,
    )
    
    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price",
            increasing_line_color="#28A745",
            decreasing_line_color="#DC3545",
        ),
        row=1, col=1,
    )
    
    # Trade levels
    fig.add_hline(y=trade["limit_price"], line_color="#FFD700", line_width=2,
                  annotation_text=f"Limit: ${trade['limit_price']:.2f}", row=1, col=1)
    fig.add_hline(y=trade["stop_price"], line_color="#DC3545", line_width=1, line_dash="dot",
                  annotation_text=f"Stop: ${trade['stop_price']:.2f}", row=1, col=1)
    fig.add_hline(y=trade["target_price"], line_color="#28A745", line_width=1, line_dash="dot",
                  annotation_text=f"Target: ${trade['target_price']:.2f}", row=1, col=1)
    
    # Volume
    colors = ["#28A745" if df["close"].iloc[i] >= df["open"].iloc[i] else "#DC3545" for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], marker_color=colors, opacity=0.7, name="Volume"),
        row=2, col=1,
    )
    
    fig.update_layout(
        title=f"<b>{symbol}</b> - {name[:50]}<br>"
              f"<sub>Shares: {trade['shares']} | Limit: ${trade['limit_price']:.2f} | "
              f"Stop: ${trade['stop_price']:.2f} | Target: ${trade['target_price']:.2f}</sub>",
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    out_path = output_dir / f"{symbol}.html"
    fig.write_html(str(out_path))
    return out_path


def create_inspector(
    candidates: pd.DataFrame,
    config: Dict,
    regime: Dict,
    week_str: str,
    output_dir: Path,
) -> Path:
    """Create inspector HTML with trade specs."""
    
    # Build table rows
    rows_html = []
    for _, t in candidates.iterrows():
        color = "negative" if t["pct_return"] < 0 else "positive"
        rows_html.append(f"""
        <tr onclick="loadChart('{t['symbol']}')" class="clickable">
            <td><strong>{t['symbol']}</strong></td>
            <td>{str(t.get('etf_name', ''))[:25]}</td>
            <td class="{color}">{t['pct_return']:.1f}%</td>
            <td><strong>{t['shares']}</strong></td>
            <td>${t['limit_price']:.2f}</td>
            <td class="negative">${t['stop_price']:.2f}</td>
            <td class="positive">${t['target_price']:.2f}</td>
        </tr>
        """)
    
    first_symbol = candidates.iloc[0]["symbol"] if len(candidates) > 0 else ""
    regime_str = f"{'🐻 BEAR' if not regime.get('is_bull') else '🐂 BULL'}"
    mult_str = f"{config.get('bear_market_multiplier', 1.0):.0%}" if not regime.get('is_bull') and config.get('use_bear_boost') else "1x"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trade Candidates - {week_str}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        h1 {{ margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 15px; }}
        .params {{ background: #e8f4f8; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; }}
        .container {{ display: flex; gap: 20px; height: calc(100vh - 180px); }}
        .sidebar {{ width: 550px; flex-shrink: 0; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 12px; background: #2E86AB; color: white; font-weight: bold; }}
        .table-container {{ overflow-y: auto; flex: 1; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f0f0f0; padding: 8px 6px; text-align: left; font-size: 11px; position: sticky; top: 0; }}
        td {{ padding: 6px; border-bottom: 1px solid #eee; font-size: 12px; }}
        tr.clickable {{ cursor: pointer; }}
        tr.clickable:hover {{ background: #e8f4f8; }}
        tr.active {{ background: #d0e8f0 !important; }}
        .negative {{ color: #DC3545; }}
        .positive {{ color: #28A745; }}
        .chart-container {{ flex: 1; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        iframe {{ width: 100%; height: 100%; border: none; }}
    </style>
</head>
<body>
    <h1>📈 Trade Candidates</h1>
    <div class="subtitle">Week of {week_str} | {len(candidates)} candidates | Regime: {regime_str} | Size: {mult_str}</div>
    <div class="params">
        <strong>Strategy:</strong> SL={config['stop_loss_pcnt']}% | TP={config['profit_exit_pcnt']}% | MaxHold={config['max_hold_weeks']}w | 
        Capital=${config['initial_capital']:,.0f} | Max Trades={config['max_active_trades']}
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">Click symbol to view chart (↑↓ to navigate)</div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Name</th>
                            <th>Return</th>
                            <th>Shares</th>
                            <th>Limit</th>
                            <th>Stop</th>
                            <th>Target</th>
                        </tr>
                    </thead>
                    <tbody id="tbl">{"".join(rows_html)}</tbody>
                </table>
            </div>
        </div>
        <div class="chart-container">
            <iframe id="chart" src="charts/{first_symbol}.html"></iframe>
        </div>
    </div>
    <script>
        function loadChart(s) {{
            document.getElementById('chart').src = 'charts/' + s + '.html';
            document.querySelectorAll('#tbl tr').forEach(r => r.classList.remove('active'));
            event.currentTarget.classList.add('active');
        }}
        document.querySelector('#tbl tr')?.classList.add('active');
        document.addEventListener('keydown', e => {{
            const rows = Array.from(document.querySelectorAll('#tbl tr'));
            const idx = rows.findIndex(r => r.classList.contains('active'));
            if (e.key === 'ArrowDown' && idx < rows.length - 1) {{ e.preventDefault(); rows[idx+1].click(); rows[idx+1].scrollIntoView({{block:'nearest'}}); }}
            if (e.key === 'ArrowUp' && idx > 0) {{ e.preventDefault(); rows[idx-1].click(); rows[idx-1].scrollIntoView({{block:'nearest'}}); }}
        }});
    </script>
</body>
</html>"""
    
    out_path = output_dir / "_inspector.html"
    out_path.write_text(html)
    return out_path


@workflow_script("21b-gen-new-trades")
def main():
    """Generate trade candidates for upcoming week."""
    
    # Load config
    config = load_config()
    print(f"Config: {CONFIG_PATH}")
    print(f"  Strategy: SL={config['stop_loss_pcnt']}%, TP={config['profit_exit_pcnt']}%, MaxHold={config['max_hold_weeks']}w")
    print(f"  Bear boost: {config.get('use_bear_boost', False)} ({config.get('bear_market_multiplier', 1.0)}x)")
    print()
    
    # Get regime
    regime = get_current_regime(config)
    print(f"Regime: {regime['regime'].upper()}")
    if "close" in regime:
        print(f"  {config['regime_symbol']}: ${regime['close']:.2f} vs MA{config['regime_ma_period']}: ${regime['ma']:.2f}")
    print()
    
    # Get latest week data
    print("Loading latest week data...")
    latest_week, week_str = get_latest_week_data()
    print(f"  Week: {week_str}")
    print(f"  ETFs: {len(latest_week)}")
    print()
    
    # Create output dir
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = PRE_PRODUCTION_DIR / today
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Identify candidates
    candidates = identify_candidates(latest_week, config, regime)
    print(f"Candidates: {len(candidates)}")
    
    if len(candidates) == 0:
        print("⚠️  No candidates meet criteria this week.")
        return
    
    # Print trade specs
    print()
    print("=" * 70)
    print("TRADE SPECIFICATIONS")
    print("=" * 70)
    total_capital = 0
    for i, (_, t) in enumerate(candidates.iterrows(), 1):
        print(f"\n{i}. {t['symbol']} - {t.get('etf_name', '')[:40]}")
        print(f"   Return: {t['pct_return']:.2f}%")
        print(f"   >>> SHARES: {t['shares']}  |  LIMIT: ${t['limit_price']:.2f}")
        print(f"   Stop: ${t['stop_price']:.2f} ({config['stop_loss_pcnt']}%)")
        print(f"   Target: ${t['target_price']:.2f} ({config['profit_exit_pcnt']}%)")
        total_capital += t["position_usd"]
    
    print()
    print(f"Total Capital: ${total_capital:,.2f} / ${config['initial_capital']:,.2f}")
    print()
    
    # Create charts
    print("Creating charts...")
    for _, t in candidates.iterrows():
        create_chart(t["symbol"], t.get("etf_name", ""), t, charts_dir)
    
    # Create inspector
    inspector_path = create_inspector(candidates, config, regime, week_str, output_dir)
    print(f"Inspector: {inspector_path}")
    
    # Save CSV
    csv_path = output_dir / "candidates.csv"
    candidates.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")
    
    # Save summary
    summary = {
        "generated": today,
        "week": week_str,
        "regime": regime,
        "config": config,
        "n_candidates": len(candidates),
        "symbols": candidates["symbol"].tolist(),
        "total_capital": total_capital,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    
    # Open inspector
    if sys.platform == "darwin":
        subprocess.run(["open", str(inspector_path)])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", str(inspector_path)])


if __name__ == "__main__":
    main()

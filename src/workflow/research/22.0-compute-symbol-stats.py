#!/usr/bin/env python3
"""
22.0-compute-symbol-stats.py

Compute Symbol-Specific Volatility Statistics for Adaptive Strategy.

For each symbol, computes rolling:
- σ (standard deviation of weekly returns)
- β (beta vs SPY)
- ATR (Average True Range - daily volatility)

Outputs:
- experiments/exp022_symbol_stats/
  - symbol_stats.parquet (time-series of stats per symbol)
  - latest_stats.csv (most recent stats snapshot)
  - spaghetti_plots.html (all symbols overlaid)
  - _inspector.html (individual symbol explorer)
  - charts/{SYMBOL}.html

Visualizations:
1. Spaghetti plots: See distribution of σ, β, ATR across all symbols
2. Inspector: Deep-dive into individual symbols
3. Terminal: Highlights of interesting findings
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp022_symbol_stats"

# Rolling window parameters
SIGMA_LOOKBACK_WEEKS = 26   # ~6 months for volatility calculation
BETA_LOOKBACK_WEEKS = 52    # 1 year for beta calculation
ATR_LOOKBACK_DAYS = 14      # Standard ATR period

# Reference symbol for beta calculation
MARKET_SYMBOL = "SPY"

# Data paths
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Visualization
CHART_LOOKBACK_WEEKS = 104  # 2 years of history in charts


# =============================================================================
# Data Loading
# =============================================================================

def get_symbols() -> List[str]:
    """Get list of symbols from filtered ETFs."""
    path = METADATA_DIR / "filtered_etfs.json"
    if not path.exists():
        # Fallback: list all weekly data files
        return [f.stem for f in WEEKLY_DATA_DIR.glob("*.csv")]
    
    with open(path) as f:
        data = json.load(f)
    return [etf["symbol"] for etf in data.get("etfs", [])]


def get_etf_names() -> Dict[str, str]:
    """Load ETF names from metadata."""
    path = METADATA_DIR / "filtered_etfs.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {etf["symbol"]: etf.get("name", etf["symbol"]) for etf in data.get("etfs", [])}


def load_weekly_returns(symbol: str) -> pd.DataFrame:
    """Load weekly data and compute log returns."""
    path = WEEKLY_DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path, parse_dates=["week_start"])
    df = df.sort_values("week_start").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change() * 100
    return df


def load_daily_data(symbol: str) -> pd.DataFrame:
    """Load daily OHLC data."""
    path = DAILY_DATA_DIR / f"{symbol}.csv"
    if not path.exists():
        return None
    
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =============================================================================
# Statistics Computation
# =============================================================================

def compute_rolling_sigma(df: pd.DataFrame, lookback: int = SIGMA_LOOKBACK_WEEKS) -> pd.Series:
    """Compute rolling standard deviation of weekly returns."""
    return df["pct_return"].rolling(window=lookback, min_periods=lookback // 2).std()


def compute_rolling_beta(
    symbol_returns: pd.Series,
    market_returns: pd.Series,
    lookback: int = BETA_LOOKBACK_WEEKS
) -> pd.Series:
    """Compute rolling beta vs market."""
    # Align series
    aligned = pd.DataFrame({
        "symbol": symbol_returns,
        "market": market_returns
    }).dropna()
    
    if len(aligned) < lookback // 2:
        return pd.Series(index=symbol_returns.index, dtype=float)
    
    # Rolling covariance / variance
    cov = aligned["symbol"].rolling(window=lookback, min_periods=lookback // 2).cov(aligned["market"])
    var = aligned["market"].rolling(window=lookback, min_periods=lookback // 2).var()
    
    beta = cov / var
    return beta.reindex(symbol_returns.index)


def compute_atr(df: pd.DataFrame, period: int = ATR_LOOKBACK_DAYS) -> pd.Series:
    """Compute Average True Range (daily volatility measure)."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period // 2).mean()
    
    # Normalize as percentage of close
    atr_pct = (atr / df["close"]) * 100
    
    return atr_pct


def compute_symbol_stats(symbol: str, market_weekly: pd.DataFrame) -> Dict:
    """Compute all stats for a single symbol."""
    weekly_df = load_weekly_returns(symbol)
    daily_df = load_daily_data(symbol)
    
    if weekly_df is None or len(weekly_df) < SIGMA_LOOKBACK_WEEKS // 2:
        return None
    
    # Rolling sigma
    weekly_df["sigma"] = compute_rolling_sigma(weekly_df)
    
    # Rolling beta
    if market_weekly is not None:
        # Merge market returns
        merged = weekly_df.merge(
            market_weekly[["week_start", "pct_return"]].rename(columns={"pct_return": "market_return"}),
            on="week_start",
            how="left"
        )
        weekly_df["beta"] = compute_rolling_beta(merged["pct_return"], merged["market_return"])
    else:
        weekly_df["beta"] = np.nan
    
    # ATR (from daily data)
    if daily_df is not None and len(daily_df) >= ATR_LOOKBACK_DAYS:
        atr_daily = compute_atr(daily_df)
        # Map daily ATR to weekly (take week-end value)
        daily_df["atr_pct"] = atr_daily
        daily_df["week_start"] = daily_df["date"] - pd.to_timedelta(daily_df["date"].dt.dayofweek, unit="D")
        weekly_atr = daily_df.groupby("week_start")["atr_pct"].last()
        weekly_df = weekly_df.merge(weekly_atr.reset_index(), on="week_start", how="left")
    else:
        weekly_df["atr_pct"] = np.nan
    
    weekly_df["symbol"] = symbol
    
    return weekly_df


# =============================================================================
# Visualization: Spaghetti Plots
# =============================================================================

def create_spaghetti_plots(all_stats: pd.DataFrame) -> go.Figure:
    """Create spaghetti plots with all symbols overlaid."""
    
    # Filter to recent data for readability
    min_date = all_stats["week_start"].max() - pd.Timedelta(weeks=CHART_LOOKBACK_WEEKS)
    recent = all_stats[all_stats["week_start"] >= min_date].copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>Rolling Volatility (σ)</b> - Weekly Return Std Dev",
            "<b>Rolling Beta (β)</b> - Sensitivity to SPY",
            "<b>ATR %</b> - Daily Volatility (as % of price)"
        ],
        row_heights=[0.33, 0.33, 0.33],
    )
    
    symbols = recent["symbol"].unique()
    n_symbols = len(symbols)
    
    # Better contrast colors for spaghetti effect
    for i, symbol in enumerate(symbols):
        sym_data = recent[recent["symbol"] == symbol].sort_values("week_start")
        
        # Sigma - darker blue with more opacity
        fig.add_trace(
            go.Scatter(
                x=sym_data["week_start"],
                y=sym_data["sigma"],
                mode="lines",
                line=dict(width=0.8, color="rgba(30,80,180,0.25)"),
                name=symbol,
                legendgroup=symbol,
                showlegend=False,
                hovertemplate=f"{symbol}<br>σ: %{{y:.2f}}%<br>%{{x}}<extra></extra>",
            ),
            row=1, col=1,
        )
        
        # Beta - darker orange
        fig.add_trace(
            go.Scatter(
                x=sym_data["week_start"],
                y=sym_data["beta"],
                mode="lines",
                line=dict(width=0.8, color="rgba(200,80,20,0.25)"),
                name=symbol,
                legendgroup=symbol,
                showlegend=False,
                hovertemplate=f"{symbol}<br>β: %{{y:.2f}}<br>%{{x}}<extra></extra>",
            ),
            row=2, col=1,
        )
        
        # ATR - darker green
        fig.add_trace(
            go.Scatter(
                x=sym_data["week_start"],
                y=sym_data["atr_pct"],
                mode="lines",
                line=dict(width=0.8, color="rgba(20,120,80,0.25)"),
                name=symbol,
                legendgroup=symbol,
                showlegend=False,
                hovertemplate=f"{symbol}<br>ATR: %{{y:.2f}}%<br>%{{x}}<extra></extra>",
            ),
            row=3, col=1,
        )
    
    # Add percentile bands (10th-90th) for context
    pct_by_week = recent.groupby("week_start").agg({
        "sigma": ["median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)],
        "beta": ["median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)], 
        "atr_pct": ["median", lambda x: x.quantile(0.1), lambda x: x.quantile(0.9)]
    }).reset_index()
    pct_by_week.columns = ["week_start", "sigma_med", "sigma_p10", "sigma_p90",
                           "beta_med", "beta_p10", "beta_p90",
                           "atr_med", "atr_p10", "atr_p90"]
    
    # Add median lines (bold, on top)
    fig.add_trace(
        go.Scatter(
            x=pct_by_week["week_start"], y=pct_by_week["sigma_med"],
            mode="lines", line=dict(width=3, color="#1E3A8A"),
            name="Median σ", showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pct_by_week["week_start"], y=pct_by_week["beta_med"],
            mode="lines", line=dict(width=3, color="#C2410C"),
            name="Median β", showlegend=True,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pct_by_week["week_start"], y=pct_by_week["atr_med"],
            mode="lines", line=dict(width=3, color="#166534"),
            name="Median ATR", showlegend=True,
        ),
        row=3, col=1,
    )
    
    # Add reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(100,100,100,0.5)", 
                  annotation_text="β=1", annotation_position="right", row=2, col=1)
    
    fig.update_layout(
        height=900,
        title=f"<b>Symbol Volatility Spaghetti Plots</b><br>"
              f"<sub>{n_symbols} symbols | σ lookback={SIGMA_LOOKBACK_WEEKS}w | "
              f"β lookback={BETA_LOOKBACK_WEEKS}w | ATR period={ATR_LOOKBACK_DAYS}d</sub>",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # White background with light gridlines
    for row in [1, 2, 3]:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.5)", row=row, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200,200,200,0.5)", row=row, col=1)
    
    fig.update_yaxes(title_text="σ (%)", row=1, col=1)
    fig.update_yaxes(title_text="β", row=2, col=1)
    fig.update_yaxes(title_text="ATR (%)", row=3, col=1)
    
    return fig


def create_distribution_snapshot(all_stats: pd.DataFrame) -> go.Figure:
    """Create distribution plots of latest stats."""
    
    # Get latest values per symbol
    latest = all_stats.sort_values("week_start").groupby("symbol").last().reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"<b>Volatility (σ) Distribution</b><br><sub>n={len(latest)}</sub>",
            f"<b>Beta (β) Distribution</b>",
            f"<b>ATR % Distribution</b>",
            f"<b>σ vs β Scatter</b>",
        ],
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "scatter"}]],
    )
    
    # Histograms
    fig.add_trace(
        go.Histogram(x=latest["sigma"].dropna(), nbinsx=40, marker_color="blue", opacity=0.7),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=latest["beta"].dropna(), nbinsx=40, marker_color="orange", opacity=0.7),
        row=1, col=2,
    )
    fig.add_trace(
        go.Histogram(x=latest["atr_pct"].dropna(), nbinsx=40, marker_color="green", opacity=0.7),
        row=2, col=1,
    )
    
    # Scatter: sigma vs beta
    fig.add_trace(
        go.Scatter(
            x=latest["sigma"], y=latest["beta"],
            mode="markers",
            marker=dict(size=5, color=latest["atr_pct"], colorscale="Viridis", showscale=True,
                       colorbar=dict(title="ATR%", x=1.02)),
            text=latest["symbol"],
            hovertemplate="%{text}<br>σ: %{x:.2f}%<br>β: %{y:.2f}<extra></extra>",
        ),
        row=2, col=2,
    )
    
    # Reference lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", row=1, col=2)  # Beta = 1
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=2)  # Beta = 1
    
    fig.update_layout(
        height=700,
        title="<b>Latest Symbol Statistics Distribution</b>",
        showlegend=False,
    )
    
    fig.update_xaxes(title_text="σ (%)", row=1, col=1)
    fig.update_xaxes(title_text="β", row=1, col=2)
    fig.update_xaxes(title_text="ATR (%)", row=2, col=1)
    fig.update_xaxes(title_text="σ (%)", row=2, col=2)
    fig.update_yaxes(title_text="β", row=2, col=2)
    
    return fig


# =============================================================================
# Visualization: Individual Symbol Charts
# =============================================================================

def create_symbol_chart(symbol: str, df: pd.DataFrame, name: str, output_dir: Path) -> Path:
    """Create detailed chart for a single symbol."""
    
    # Filter to recent history
    min_date = df["week_start"].max() - pd.Timedelta(weeks=CHART_LOOKBACK_WEEKS)
    df = df[df["week_start"] >= min_date].copy()
    
    if len(df) < 10:
        return None
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            f"Weekly Returns",
            f"Rolling Volatility (σ) - {SIGMA_LOOKBACK_WEEKS}w",
            f"Rolling Beta (β) - {BETA_LOOKBACK_WEEKS}w",
            f"ATR % - {ATR_LOOKBACK_DAYS}d",
        ],
        row_heights=[0.25, 0.25, 0.25, 0.25],
    )
    
    # Returns as bar chart
    colors = ["#28A745" if r >= 0 else "#DC3545" for r in df["pct_return"]]
    fig.add_trace(
        go.Bar(
            x=df["week_start"], y=df["pct_return"],
            marker_color=colors, opacity=0.7,
            name="Return",
            hovertemplate="%{y:.2f}%<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Sigma
    fig.add_trace(
        go.Scatter(
            x=df["week_start"], y=df["sigma"],
            mode="lines", line=dict(color="blue", width=2),
            name="σ",
            hovertemplate="σ: %{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    
    # Beta with reference line
    fig.add_trace(
        go.Scatter(
            x=df["week_start"], y=df["beta"],
            mode="lines", line=dict(color="orange", width=2),
            name="β",
            hovertemplate="β: %{y:.2f}<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=3, col=1)
    
    # ATR
    fig.add_trace(
        go.Scatter(
            x=df["week_start"], y=df["atr_pct"],
            mode="lines", line=dict(color="green", width=2),
            name="ATR",
            hovertemplate="ATR: %{y:.2f}%<extra></extra>",
        ),
        row=4, col=1,
    )
    
    # Latest values annotation
    latest = df.iloc[-1]
    stats_text = (
        f"Latest: σ={latest['sigma']:.2f}% | "
        f"β={latest['beta']:.2f} | "
        f"ATR={latest['atr_pct']:.2f}%"
    )
    
    fig.update_layout(
        height=800,
        title=f"<b>{symbol}</b> - {name[:50]}<br><sub>{stats_text}</sub>",
        showlegend=False,
        hovermode="x unified",
    )
    
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="σ %", row=2, col=1)
    fig.update_yaxes(title_text="β", row=3, col=1)
    fig.update_yaxes(title_text="ATR %", row=4, col=1)
    
    out_path = output_dir / f"{symbol}.html"
    fig.write_html(str(out_path))
    return out_path


def create_inspector(
    latest_stats: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Create inspector HTML for browsing individual symbols."""
    
    # Sort by sigma (most volatile first)
    sorted_stats = latest_stats.sort_values("sigma", ascending=False)
    
    names = get_etf_names()
    
    rows_html = []
    for _, row in sorted_stats.iterrows():
        symbol = row["symbol"]
        name = names.get(symbol, symbol)
        
        # Color code beta
        beta = row.get("beta", np.nan)
        if pd.notna(beta):
            if beta > 1.2:
                beta_class = "high-beta"
            elif beta < 0.8:
                beta_class = "low-beta"
            else:
                beta_class = ""
            beta_str = f"{beta:.2f}"
        else:
            beta_class = ""
            beta_str = "N/A"
        
        sigma = row.get("sigma", np.nan)
        sigma_str = f"{sigma:.2f}%" if pd.notna(sigma) else "N/A"
        
        atr = row.get("atr_pct", np.nan)
        atr_str = f"{atr:.2f}%" if pd.notna(atr) else "N/A"
        
        rows_html.append(f"""
        <tr onclick="loadChart('{symbol}')" class="clickable">
            <td><strong>{symbol}</strong></td>
            <td>{name[:30]}</td>
            <td>{sigma_str}</td>
            <td class="{beta_class}">{beta_str}</td>
            <td>{atr_str}</td>
        </tr>
        """)
    
    first_symbol = sorted_stats.iloc[0]["symbol"] if len(sorted_stats) > 0 else ""
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Symbol Stats Inspector</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        h1 {{ margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 15px; }}
        .info-box {{ background: #e8f4f8; padding: 12px; border-radius: 4px; margin-bottom: 15px; font-size: 13px; }}
        .container {{ display: flex; gap: 20px; height: calc(100vh - 200px); }}
        .sidebar {{ width: 480px; flex-shrink: 0; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; }}
        .sidebar-header {{ padding: 12px; background: #2E86AB; color: white; font-weight: bold; }}
        .table-container {{ overflow-y: auto; flex: 1; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ background: #f0f0f0; padding: 8px 6px; text-align: left; font-size: 11px; position: sticky; top: 0; cursor: pointer; }}
        th:hover {{ background: #e0e0e0; }}
        td {{ padding: 6px; border-bottom: 1px solid #eee; font-size: 12px; }}
        tr.clickable {{ cursor: pointer; }}
        tr.clickable:hover {{ background: #e8f4f8; }}
        tr.active {{ background: #d0e8f0 !important; }}
        .high-beta {{ color: #DC3545; font-weight: bold; }}
        .low-beta {{ color: #28A745; }}
        .chart-container {{ flex: 1; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        iframe {{ width: 100%; height: 100%; border: none; }}
    </style>
</head>
<body>
    <h1>📊 Symbol Statistics Inspector</h1>
    <div class="subtitle">{len(sorted_stats)} symbols | Sorted by volatility (σ) descending</div>
    
    <div class="info-box">
        <strong>Metrics:</strong><br>
        • <b>σ (Sigma)</b>: Rolling {SIGMA_LOOKBACK_WEEKS}-week std dev of weekly returns. Higher = more volatile.<br>
        • <b>β (Beta)</b>: Sensitivity to SPY. β>1 = amplifies market moves. β<1 = dampens. β<0 = inverse.<br>
        • <b>ATR %</b>: {ATR_LOOKBACK_DAYS}-day Average True Range as % of price. Daily volatility measure.
    </div>
    
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">Click symbol to view chart (↑↓ to navigate)</div>
            <div class="table-container">
                <table id="statsTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Symbol</th>
                            <th onclick="sortTable(1)">Name</th>
                            <th onclick="sortTable(2)">σ ▼</th>
                            <th onclick="sortTable(3)">β</th>
                            <th onclick="sortTable(4)">ATR</th>
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
        
        let sortDir = {{}};
        function sortTable(col) {{
            const table = document.getElementById('statsTable');
            const tbody = document.getElementById('tbl');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            sortDir[col] = !sortDir[col];
            rows.sort((a, b) => {{
                let aVal = a.cells[col].textContent.replace('%', '').trim();
                let bVal = b.cells[col].textContent.replace('%', '').trim();
                let aNum = parseFloat(aVal);
                let bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return sortDir[col] ? aNum - bNum : bNum - aNum;
                }}
                return sortDir[col] ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});
            rows.forEach(r => tbody.appendChild(r));
        }}
    </script>
</body>
</html>"""
    
    out_path = output_dir / "_inspector.html"
    out_path.write_text(html)
    return out_path


# =============================================================================
# Interesting Findings
# =============================================================================

def report_interesting_findings(latest: pd.DataFrame, all_stats: pd.DataFrame):
    """Print interesting findings to terminal."""
    
    print()
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    
    names = get_etf_names()
    
    # Check for likely split artifacts (returns > 30%)
    split_artifacts = all_stats[abs(all_stats["pct_return"]) > 30].copy()
    if len(split_artifacts) > 0:
        print("\n⚠️  LIKELY SPLIT/DATA ARTIFACTS (|return| > 30%):")
        # Group by symbol and date
        artifact_summary = split_artifacts.groupby("symbol").agg({
            "week_start": lambda x: x.iloc[0],
            "pct_return": lambda x: x.iloc[0]
        }).reset_index()
        artifact_summary = artifact_summary.sort_values("pct_return")
        for _, row in artifact_summary.head(15).iterrows():
            name = names.get(row["symbol"], "")[:30]
            print(f"   {row['symbol']:6s} {row['week_start'].strftime('%Y-%m-%d')} "
                  f"return={row['pct_return']:+.1f}%  {name}")
        if len(artifact_summary) > 15:
            print(f"   ... and {len(artifact_summary) - 15} more")
        print(f"\n   RECOMMENDATION: Re-fetch data with adjustment=Adjustment.ALL")
        print(f"   Run: python src/workflow/pipeline/02-fetch-daily-data.py --force")
    else:
        print("\n✅ No obvious split artifacts detected (all |returns| < 30%)")
    
    print()
    print("=" * 70)
    print("INTERESTING FINDINGS")
    print("=" * 70)
    
    names = get_etf_names()
    
    # Most volatile
    print("\n📈 HIGHEST VOLATILITY (σ > 8%):")
    high_vol = latest[latest["sigma"] > 8].sort_values("sigma", ascending=False)
    if len(high_vol) > 0:
        for _, row in high_vol.head(10).iterrows():
            name = names.get(row["symbol"], "")[:30]
            print(f"   {row['symbol']:6s} σ={row['sigma']:5.1f}%  β={row['beta']:5.2f}  {name}")
    else:
        print("   None found")
    
    # Lowest volatility
    print("\n📉 LOWEST VOLATILITY (σ < 1%):")
    low_vol = latest[latest["sigma"] < 1].sort_values("sigma")
    if len(low_vol) > 0:
        for _, row in low_vol.head(10).iterrows():
            name = names.get(row["symbol"], "")[:30]
            print(f"   {row['symbol']:6s} σ={row['sigma']:5.2f}%  β={row['beta']:5.2f}  {name}")
    else:
        print("   None found")
    
    # High beta
    print("\n🔥 HIGH BETA (β > 1.5):")
    high_beta = latest[latest["beta"] > 1.5].sort_values("beta", ascending=False)
    if len(high_beta) > 0:
        for _, row in high_beta.head(10).iterrows():
            name = names.get(row["symbol"], "")[:30]
            print(f"   {row['symbol']:6s} β={row['beta']:5.2f}  σ={row['sigma']:5.1f}%  {name}")
    else:
        print("   None found")
    
    # Negative/inverse beta
    print("\n🔄 INVERSE/LOW BETA (β < 0.3):")
    low_beta = latest[latest["beta"] < 0.3].sort_values("beta")
    if len(low_beta) > 0:
        for _, row in low_beta.head(10).iterrows():
            name = names.get(row["symbol"], "")[:30]
            print(f"   {row['symbol']:6s} β={row['beta']:5.2f}  σ={row['sigma']:5.1f}%  {name}")
    else:
        print("   None found")
    
    # Volatility regime changes (symbols where sigma has changed significantly recently)
    print("\n⚡ VOLATILITY REGIME CHANGES (σ changed >50% in last 13w):")
    recent_cutoff = all_stats["week_start"].max() - pd.Timedelta(weeks=13)
    regime_changes = []
    
    for symbol in latest["symbol"].unique():
        sym_data = all_stats[all_stats["symbol"] == symbol].sort_values("week_start")
        if len(sym_data) < 13:
            continue
        
        recent = sym_data[sym_data["week_start"] >= recent_cutoff]
        older = sym_data[sym_data["week_start"] < recent_cutoff].tail(13)
        
        if len(recent) > 0 and len(older) > 0:
            recent_sigma = recent["sigma"].mean()
            older_sigma = older["sigma"].mean()
            
            if older_sigma > 0:
                change_pct = (recent_sigma - older_sigma) / older_sigma * 100
                if abs(change_pct) > 50:
                    regime_changes.append({
                        "symbol": symbol,
                        "old_sigma": older_sigma,
                        "new_sigma": recent_sigma,
                        "change_pct": change_pct,
                    })
    
    if regime_changes:
        regime_df = pd.DataFrame(regime_changes).sort_values("change_pct", key=abs, ascending=False)
        for _, row in regime_df.head(10).iterrows():
            direction = "↑" if row["change_pct"] > 0 else "↓"
            name = names.get(row["symbol"], "")[:25]
            print(f"   {row['symbol']:6s} σ: {row['old_sigma']:4.1f}% → {row['new_sigma']:4.1f}% ({direction}{abs(row['change_pct']):4.0f}%)  {name}")
    else:
        print("   None found")
    
    # Summary statistics
    print("\n📊 SUMMARY STATISTICS:")
    print(f"   Volatility (σ):  median={latest['sigma'].median():.2f}%  "
          f"mean={latest['sigma'].mean():.2f}%  range=[{latest['sigma'].min():.2f}, {latest['sigma'].max():.2f}]")
    print(f"   Beta (β):        median={latest['beta'].median():.2f}  "
          f"mean={latest['beta'].mean():.2f}  range=[{latest['beta'].min():.2f}, {latest['beta'].max():.2f}]")
    print(f"   ATR %:           median={latest['atr_pct'].median():.2f}%  "
          f"mean={latest['atr_pct'].mean():.2f}%  range=[{latest['atr_pct'].min():.2f}, {latest['atr_pct'].max():.2f}]")


# =============================================================================
# Main
# =============================================================================

@workflow_script("22.0-compute-symbol-stats")
def main():
    """Compute symbol statistics and create visualizations."""
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charts_dir = OUTPUT_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Get symbols
    symbols = get_symbols()
    print(f"Symbols: {len(symbols)}")
    
    # Load market data for beta calculation
    print(f"Loading {MARKET_SYMBOL} for beta calculation...")
    market_weekly = load_weekly_returns(MARKET_SYMBOL)
    if market_weekly is not None:
        print(f"  {MARKET_SYMBOL} weeks: {len(market_weekly)}")
    else:
        print(f"  Warning: {MARKET_SYMBOL} not found, beta will be NaN")
    
    # Compute stats for all symbols
    print("\nComputing symbol statistics...")
    all_stats = []
    names = get_etf_names()
    
    for i, symbol in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(symbols)}...")
        
        stats = compute_symbol_stats(symbol, market_weekly)
        if stats is not None:
            all_stats.append(stats)
    
    if not all_stats:
        print("No symbols with sufficient data!")
        return
    
    # Combine all stats
    all_stats_df = pd.concat(all_stats, ignore_index=True)
    print(f"\nTotal data points: {len(all_stats_df)}")
    print(f"Symbols with data: {all_stats_df['symbol'].nunique()}")
    
    # Get latest snapshot
    latest = all_stats_df.sort_values("week_start").groupby("symbol").last().reset_index()
    print(f"Latest snapshot: {len(latest)} symbols")
    
    # Save data
    print("\nSaving data...")
    all_stats_df.to_parquet(OUTPUT_DIR / "symbol_stats.parquet", index=False)
    latest.to_csv(OUTPUT_DIR / "latest_stats.csv", index=False)
    
    # Create visualizations
    print("\nCreating spaghetti plots...")
    spaghetti_fig = create_spaghetti_plots(all_stats_df)
    spaghetti_fig.write_html(str(OUTPUT_DIR / "spaghetti_plots.html"))
    
    print("Creating distribution snapshot...")
    dist_fig = create_distribution_snapshot(all_stats_df)
    dist_fig.write_html(str(OUTPUT_DIR / "distributions.html"))
    
    print("Creating individual symbol charts...")
    chart_count = 0
    for symbol in latest["symbol"].unique():
        sym_data = all_stats_df[all_stats_df["symbol"] == symbol]
        name = names.get(symbol, symbol)
        result = create_symbol_chart(symbol, sym_data, name, charts_dir)
        if result:
            chart_count += 1
    print(f"  Created {chart_count} charts")
    
    print("Creating inspector...")
    inspector_path = create_inspector(latest, OUTPUT_DIR)
    
    # Report interesting findings
    report_interesting_findings(latest, all_stats_df)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Spaghetti: {OUTPUT_DIR / 'spaghetti_plots.html'}")
    print(f"  Distributions: {OUTPUT_DIR / 'distributions.html'}")
    print(f"  Inspector: {inspector_path}")
    
    # Open visualizations
    import subprocess
    import sys
    if sys.platform == "darwin":
        subprocess.run(["open", str(OUTPUT_DIR / "spaghetti_plots.html")])
        subprocess.run(["open", str(inspector_path)])


if __name__ == "__main__":
    main()

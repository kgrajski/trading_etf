#!/usr/bin/env python3
"""
19.3-gen-trades.py

Research Trade Candidate Generator.

Generates actionable trade candidates for the upcoming week using
the current research champion parameters from 19.1/19.1b.

Champion Parameters (from 19.1b grid search):
- Stop Loss: 4% (tighter)
- Profit Target: 10%
- Max Hold: 3 weeks
- Min Loss: 6% (higher threshold)
- Boost: Bull market (1.10x when SPY > MA50)

Outputs to experiments/exp019_3_trades/{date}/:
- _inspector.html (main view - opens automatically)
- candidates.csv
- summary.json
- charts/{SYMBOL}.html

Note: This is the research version. Once validated, sync to 21b for production.
"""

import json
import subprocess
import sys
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
# Champion Configuration (from 19.1b grid search)
# =============================================================================

EXPERIMENT_NAME = "exp019_3_trades"

# Capital parameters
INITIAL_CAPITAL = 10_000.0
MAX_ACTIVE_TRADES = 10

# Entry parameters (CHAMPION)
BOTTOM_PERCENTILE = 0.05  # Bottom 5% losers
MIN_LOSS_PCNT = 6.0       # Champion: 6% (higher threshold)

# Exit parameters (CHAMPION)
STOP_LOSS_PCNT = 4.0      # Champion: 4% (tighter stop)
PROFIT_EXIT_PCNT = 10.0   # Champion: 10%
MAX_HOLD_WEEKS = 3        # Champion: 3 weeks

# Regime parameters (CHAMPION - bull boost)
USE_REGIME = True
REGIME_SYMBOL = "SPY"
REGIME_MA_PERIOD = 50
BOOST_DIRECTION = "bull"  # Champion: bull (larger positions in bull markets)
BOOST_MULTIPLIER = 1.10   # 10% larger positions when condition met

# Chart parameters
BB_PERIOD = 20
BB_STD = 2
CHART_LOOKBACK_DAYS = 252

# Data paths
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"
DAILY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "daily"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME
SYMBOL_STATS_PATH = PROJECT_ROOT / "experiments" / "exp022_symbol_stats" / "latest_stats.csv"

# Volatility thresholds for risk badges
HIGH_VOLATILITY_THRESHOLD = 4.0    # σ > 4% = high volatility warning
HIGH_BETA_THRESHOLD = 1.5          # β > 1.5 = high market sensitivity
LOW_BETA_THRESHOLD = 0.0           # β < 0 = inverse correlation


# =============================================================================
# Data Loading
# =============================================================================

def get_current_regime() -> Dict:
    """Determine current market regime."""
    daily_path = DAILY_DATA_DIR / f"{REGIME_SYMBOL}.csv"
    
    if not daily_path.exists():
        return {"is_bull": True, "regime": "unknown"}
    
    df = pd.read_csv(daily_path, parse_dates=["date"])
    df = df.sort_values("date").tail(REGIME_MA_PERIOD + 10)
    df["ma"] = df["close"].rolling(window=REGIME_MA_PERIOD).mean()
    
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


def load_symbol_stats() -> pd.DataFrame:
    """Load volatility stats from 22.0 output."""
    if not SYMBOL_STATS_PATH.exists():
        print(f"  Note: Symbol stats not found at {SYMBOL_STATS_PATH}")
        print(f"  Run 22.0-compute-symbol-stats.py first for volatility data")
        return pd.DataFrame()
    
    df = pd.read_csv(SYMBOL_STATS_PATH)
    return df


def get_risk_badge(sigma: float, beta: float) -> str:
    """Generate risk badge based on volatility profile."""
    badges = []
    
    if pd.notna(sigma) and sigma > HIGH_VOLATILITY_THRESHOLD:
        badges.append("🔥")  # High volatility
    
    if pd.notna(beta):
        if beta > HIGH_BETA_THRESHOLD:
            badges.append("⚡")  # High beta / amplifies market
        elif beta < LOW_BETA_THRESHOLD:
            badges.append("🔄")  # Inverse correlation
    
    return "".join(badges) if badges else ""


def get_risk_notes(candidates: pd.DataFrame) -> List[str]:
    """Generate risk notes for the candidate set."""
    notes = []
    
    # High volatility symbols
    high_vol = candidates[candidates["sigma"] > HIGH_VOLATILITY_THRESHOLD]
    if len(high_vol) > 0:
        symbols = ", ".join(high_vol["symbol"].tolist())
        notes.append(f"🔥 <b>High Volatility:</b> {symbols} (σ > {HIGH_VOLATILITY_THRESHOLD}%) - "
                    f"wider price swings, stop may trigger on noise")
    
    # High beta symbols
    high_beta = candidates[candidates["beta"] > HIGH_BETA_THRESHOLD]
    if len(high_beta) > 0:
        symbols = ", ".join(high_beta["symbol"].tolist())
        notes.append(f"⚡ <b>High Beta:</b> {symbols} (β > {HIGH_BETA_THRESHOLD}) - "
                    f"amplifies market moves, watch overall market direction")
    
    # Inverse/low beta symbols
    inv_beta = candidates[candidates["beta"] < LOW_BETA_THRESHOLD]
    if len(inv_beta) > 0:
        symbols = ", ".join(inv_beta["symbol"].tolist())
        notes.append(f"🔄 <b>Inverse Correlation:</b> {symbols} (β < 0) - "
                    f"moves opposite to market, different risk profile")
    
    # Wide σ spread (some symbols much more volatile than others)
    if len(candidates) > 1 and candidates["sigma"].notna().sum() > 1:
        sigma_range = candidates["sigma"].max() - candidates["sigma"].min()
        if sigma_range > 3:
            notes.append(f"📊 <b>Mixed Volatility:</b> σ ranges from "
                        f"{candidates['sigma'].min():.1f}% to {candidates['sigma'].max():.1f}% - "
                        f"consider position sizing by volatility")
    
    return notes


# =============================================================================
# Trade Logic
# =============================================================================

def identify_candidates(df: pd.DataFrame, regime: Dict) -> pd.DataFrame:
    """Identify and size trade candidates using champion parameters."""
    # Sort by worst return
    sorted_df = df.sort_values("pct_return")
    
    # Bottom percentile
    n = max(1, int(len(sorted_df) * BOTTOM_PERCENTILE))
    bottom = sorted_df.head(n)
    
    # Filter by min loss (champion: 6%)
    candidates = bottom[bottom["pct_return"] <= -MIN_LOSS_PCNT].copy()
    
    if len(candidates) == 0:
        return candidates
    
    # Add names
    names = get_etf_names()
    candidates["etf_name"] = candidates["symbol"].map(lambda s: names.get(s, s))
    
    # Load and merge volatility stats
    vol_stats = load_symbol_stats()
    if len(vol_stats) > 0:
        vol_cols = ["symbol", "sigma", "beta", "atr_pct"]
        vol_cols = [c for c in vol_cols if c in vol_stats.columns]
        candidates = candidates.merge(vol_stats[vol_cols], on="symbol", how="left")
        # Add risk badges
        candidates["risk_badge"] = candidates.apply(
            lambda r: get_risk_badge(r.get("sigma", np.nan), r.get("beta", np.nan)), axis=1
        )
    else:
        candidates["sigma"] = np.nan
        candidates["beta"] = np.nan
        candidates["atr_pct"] = np.nan
        candidates["risk_badge"] = ""
    
    # Position sizing with bull boost (champion logic)
    is_bull = regime.get("is_bull", True)
    
    if USE_REGIME and BOOST_DIRECTION == "bull" and is_bull:
        multiplier = BOOST_MULTIPLIER
    elif USE_REGIME and BOOST_DIRECTION == "bear" and not is_bull:
        multiplier = BOOST_MULTIPLIER
    else:
        multiplier = 1.0
    
    base_size = INITIAL_CAPITAL / MAX_ACTIVE_TRADES
    target_size = base_size * multiplier
    
    candidates["multiplier"] = multiplier
    candidates["target_usd"] = target_size
    
    # Trade specs
    candidates["limit_price"] = candidates["close"].round(2)
    candidates["stop_price"] = (candidates["close"] * (1 - STOP_LOSS_PCNT / 100)).round(2)
    candidates["target_price"] = (candidates["close"] * (1 + PROFIT_EXIT_PCNT / 100)).round(2)
    candidates["shares"] = (target_size / candidates["close"]).astype(int)
    candidates["position_usd"] = (candidates["shares"] * candidates["close"]).round(2)
    
    return candidates


# =============================================================================
# Visualization
# =============================================================================

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
    
    # Trade levels - lines without annotations (cleaner look)
    fig.add_hline(y=trade["limit_price"], line_color="#FFD700", line_width=2, row=1, col=1)
    fig.add_hline(y=trade["stop_price"], line_color="#DC3545", line_width=1, line_dash="dot", row=1, col=1)
    fig.add_hline(y=trade["target_price"], line_color="#28A745", line_width=1, line_dash="dot", row=1, col=1)
    
    # Add labels in right margin (outside plot area)
    fig.add_annotation(
        x=1.02, xref="paper", y=trade["limit_price"], yref="y",
        text=f"Limit ${trade['limit_price']:.2f}", showarrow=False,
        font=dict(size=10, color="#FFD700"), bgcolor="rgba(255,255,255,0.8)",
        xanchor="left",
    )
    fig.add_annotation(
        x=1.02, xref="paper", y=trade["stop_price"], yref="y",
        text=f"Stop ${trade['stop_price']:.2f}", showarrow=False,
        font=dict(size=10, color="#DC3545"), bgcolor="rgba(255,255,255,0.8)",
        xanchor="left",
    )
    fig.add_annotation(
        x=1.02, xref="paper", y=trade["target_price"], yref="y",
        text=f"Target ${trade['target_price']:.2f}", showarrow=False,
        font=dict(size=10, color="#28A745"), bgcolor="rgba(255,255,255,0.8)",
        xanchor="left",
    )
    
    # Volume
    colors = ["#28A745" if df["close"].iloc[i] >= df["open"].iloc[i] else "#DC3545" for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df["date"], y=df["volume"], marker_color=colors, opacity=0.7, name="Volume"),
        row=2, col=1,
    )
    
    fig.update_layout(
        title=f"<b>{symbol}</b> - {name[:50]}<br>"
              f"<sub>Shares: {trade['shares']} | Limit: ${trade['limit_price']:.2f} | "
              f"Stop: ${trade['stop_price']:.2f} (-{STOP_LOSS_PCNT}%) | Target: ${trade['target_price']:.2f} (+{PROFIT_EXIT_PCNT}%)</sub>",
        height=700,
        margin=dict(r=100),  # Right margin for price labels
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    out_path = output_dir / f"{symbol}.html"
    fig.write_html(str(out_path))
    return out_path


def load_ai_analysis(output_dir: Path) -> Dict:
    """Load AI analyst report if available."""
    report_path = output_dir / "analyst_report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load AI analysis: {e}")
    return {}


def create_inspector(
    candidates: pd.DataFrame,
    regime: Dict,
    week_str: str,
    output_dir: Path,
) -> Path:
    """Create inspector HTML with trade specs, volatility indicators, and AI analysis."""
    
    # Load AI analysis if available
    ai_analysis = load_ai_analysis(output_dir)
    has_ai = bool(ai_analysis.get("symbol_analyses"))
    
    # Get AI symbol analyses for enriching table
    symbol_analyses = ai_analysis.get("symbol_analyses", {})
    review = ai_analysis.get("review_results", {}) or {}
    top_picks = review.get("top_picks", [])
    avoid_list = review.get("avoid_list", [])
    
    # Build table rows with volatility tooltips and AI badges
    rows_html = []
    for _, t in candidates.iterrows():
        symbol = t["symbol"]
        color = "negative" if t["pct_return"] < 0 else "positive"
        badge = t.get("risk_badge", "")
        
        # Add AI conviction badge if available
        ai_data = symbol_analyses.get(symbol, {})
        conviction = ai_data.get("conviction")
        if conviction is not None:
            if symbol in top_picks:
                badge += f' <span class="ai-badge top">⭐{conviction}</span>'
            elif symbol in avoid_list:
                badge += f' <span class="ai-badge avoid">⛔{conviction}</span>'
            elif conviction >= 7:
                badge += f' <span class="ai-badge high">{conviction}</span>'
            elif conviction <= 3:
                badge += f' <span class="ai-badge low">{conviction}</span>'
            else:
                badge += f' <span class="ai-badge">{conviction}</span>'
        
        # Build tooltip with vol stats
        sigma = t.get("sigma", np.nan)
        beta = t.get("beta", np.nan)
        atr = t.get("atr_pct", np.nan)
        
        tooltip_parts = []
        if pd.notna(sigma):
            tooltip_parts.append(f"σ={sigma:.1f}%")
        if pd.notna(beta):
            tooltip_parts.append(f"β={beta:.2f}")
        if pd.notna(atr):
            tooltip_parts.append(f"ATR={atr:.1f}%")
        tooltip = " | ".join(tooltip_parts) if tooltip_parts else "No vol data"
        
        rows_html.append(f"""
        <tr onclick="loadChart('{symbol}')" class="clickable" title="{tooltip}" data-symbol="{symbol}">
            <td><strong>{symbol}</strong> {badge}</td>
            <td>{str(t.get('etf_name', ''))[:22]}</td>
            <td class="{color}">{t['pct_return']:.1f}%</td>
            <td><strong>{t['shares']}</strong></td>
            <td>${t['limit_price']:.2f}</td>
            <td class="negative">${t['stop_price']:.2f}</td>
            <td class="positive">${t['target_price']:.2f}</td>
        </tr>
        """)
    
    first_symbol = candidates.iloc[0]["symbol"] if len(candidates) > 0 else ""
    regime_str = f"{'🐂 BULL' if regime.get('is_bull') else '🐻 BEAR'}"
    boost_active = (BOOST_DIRECTION == "bull" and regime.get("is_bull")) or \
                   (BOOST_DIRECTION == "bear" and not regime.get("is_bull"))
    mult_str = f"{BOOST_MULTIPLIER:.0%}" if USE_REGIME and boost_active else "1x"
    
    # Generate risk notes
    risk_notes = get_risk_notes(candidates)
    risk_notes_html = ""
    if risk_notes:
        notes_list = "".join([f"<li>{note}</li>" for note in risk_notes])
        risk_notes_html = f"""
    <div class="risk-notes">
        <strong>⚠️ Risk Notes:</strong>
        <ul>{notes_list}</ul>
    </div>"""
    
    # Build AI analysis button and data
    ai_button_html = ""
    ai_data_json = "{}"
    if has_ai:
        thematic = ai_analysis.get("thematic_analysis", {})
        sentiment = thematic.get("overall_sentiment", "unknown").upper()
        ai_button_html = f'''
    <div class="ai-section">
        <button class="ai-btn" onclick="showAIModal()">🤖 AI Market Analysis</button>
        <span class="ai-sentiment {sentiment.lower()}">{sentiment}</span>
        <span class="ai-hint">Click for thematic analysis | Symbol insights appear above chart</span>
    </div>'''
        # Serialize AI data for JavaScript (escape for embedding)
        ai_data_json = json.dumps(ai_analysis, default=str).replace("</", "<\\/")
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Trade Candidates - {week_str}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        h1 {{ margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 15px; }}
        .badge {{ display: inline-block; background: #2E86AB; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 10px; }}
        .params {{ background: #e8f4f8; padding: 10px; border-radius: 4px; margin-bottom: 10px; font-size: 14px; }}
        .champion {{ background: #d4edda; border: 1px solid #28A745; padding: 8px; border-radius: 4px; margin-bottom: 10px; font-size: 13px; }}
        .risk-notes {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin-bottom: 10px; font-size: 13px; }}
        .risk-notes ul {{ margin: 5px 0 0 0; padding-left: 20px; }}
        .risk-notes li {{ margin: 4px 0; }}
        
        /* AI Section Styles */
        .ai-section {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 15px; border-radius: 4px; margin-bottom: 10px; display: flex; align-items: center; gap: 15px; }}
        .ai-btn {{ background: white; color: #667eea; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; }}
        .ai-btn:hover {{ background: #f0f0f0; }}
        .ai-sentiment {{ padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .ai-sentiment.favorable {{ background: #28A745; }}
        .ai-sentiment.neutral {{ background: #FFC107; color: #333; }}
        .ai-sentiment.unfavorable {{ background: #DC3545; }}
        .ai-sentiment.unknown {{ background: #6c757d; }}
        .ai-hint {{ font-size: 11px; opacity: 0.8; }}
        
        /* AI Badge Styles */
        .ai-badge {{ display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; font-weight: bold; margin-left: 4px; background: #6c757d; color: white; }}
        .ai-badge.high {{ background: #28A745; }}
        .ai-badge.low {{ background: #DC3545; }}
        .ai-badge.top {{ background: #FFD700; color: #333; }}
        .ai-badge.avoid {{ background: #DC3545; }}
        
        /* Modal Styles */
        .modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; justify-content: center; align-items: center; }}
        .modal.active {{ display: flex; }}
        .modal-content {{ background: white; border-radius: 8px; max-width: 700px; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .modal-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }}
        .modal-header h2 {{ margin: 0; font-size: 18px; }}
        .modal-close {{ background: none; border: none; color: white; font-size: 24px; cursor: pointer; }}
        .modal-body {{ padding: 20px; }}
        .theme-card {{ background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #667eea; }}
        .theme-card h4 {{ margin: 0 0 5px 0; color: #333; }}
        .theme-symbols {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
        .theme-outlook {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }}
        .theme-outlook.favorable {{ background: #d4edda; color: #155724; }}
        .theme-outlook.neutral {{ background: #fff3cd; color: #856404; }}
        .theme-outlook.unfavorable {{ background: #f8d7da; color: #721c24; }}
        .reviewer-notes {{ background: #e8f4f8; padding: 12px; border-radius: 6px; margin-top: 15px; font-style: italic; }}
        
        /* Symbol Overlay Styles */
        .symbol-overlay {{ display: none; background: rgba(255,255,255,0.97); border-radius: 6px; padding: 12px; margin-bottom: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); border-left: 4px solid #667eea; }}
        .symbol-overlay.active {{ display: block; }}
        .symbol-overlay-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }}
        .symbol-overlay h4 {{ margin: 0; color: #333; }}
        .conviction-badge {{ padding: 4px 12px; border-radius: 4px; font-weight: bold; color: white; }}
        .conviction-badge.high {{ background: #28A745; }}
        .conviction-badge.medium {{ background: #FFC107; color: #333; }}
        .conviction-badge.low {{ background: #DC3545; }}
        .symbol-narrative {{ font-size: 13px; line-height: 1.5; margin-bottom: 8px; }}
        .symbol-pros-cons {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; }}
        .symbol-pros-cons ul {{ margin: 0; padding-left: 18px; }}
        .key-risk {{ background: #fff3cd; padding: 6px 10px; border-radius: 4px; font-size: 12px; margin-top: 8px; }}
        
        .container {{ display: flex; gap: 20px; height: calc(100vh - {'340' if has_ai else '280'}px); }}
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
        .chart-area {{ flex: 1; display: flex; flex-direction: column; }}
        .chart-container {{ flex: 1; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); overflow: hidden; }}
        iframe {{ width: 100%; height: 100%; border: none; }}
        .legend {{ font-size: 11px; color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>📈 Research Trade Candidates <span class="badge">19.3</span></h1>
    <div class="subtitle">Week of {week_str} | {len(candidates)} candidates | Regime: {regime_str} | Size: {mult_str}</div>
    
    <div class="champion">
        <strong>Champion Config (19.1b):</strong> 
        SL={STOP_LOSS_PCNT}% | TP={PROFIT_EXIT_PCNT}% | MaxHold={MAX_HOLD_WEEKS}w | MinLoss={MIN_LOSS_PCNT}% | 
        Boost={BOOST_DIRECTION} {BOOST_MULTIPLIER:.0%}
    </div>
    
    <div class="params">
        <strong>Position Sizing:</strong> 
        Capital=${INITIAL_CAPITAL:,.0f} | Max Trades={MAX_ACTIVE_TRADES} | 
        Base=${INITIAL_CAPITAL/MAX_ACTIVE_TRADES:,.0f}/trade | 
        This Week=${INITIAL_CAPITAL/MAX_ACTIVE_TRADES * (BOOST_MULTIPLIER if boost_active else 1):,.0f}/trade
        <span class="legend">| Badges: 🔥 High Vol (σ>{HIGH_VOLATILITY_THRESHOLD}%) | ⚡ High Beta (β>{HIGH_BETA_THRESHOLD}) | 🔄 Inverse (β<0) | Hover row for stats</span>
    </div>
    {risk_notes_html}
    {ai_button_html}
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
        <div class="chart-area">
            <div class="symbol-overlay" id="symbolOverlay"></div>
            <div class="chart-container">
                <iframe id="chart" src="charts/{first_symbol}.html"></iframe>
            </div>
        </div>
    </div>
    
    <!-- AI Analysis Modal -->
    <div class="modal" id="aiModal" onclick="if(event.target===this)hideAIModal()">
        <div class="modal-content">
            <div class="modal-header">
                <h2>🤖 AI Market Analysis</h2>
                <button class="modal-close" onclick="hideAIModal()">&times;</button>
            </div>
            <div class="modal-body" id="aiModalBody">
                <!-- Populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <script>
        // AI Analysis Data (embedded)
        const aiData = {ai_data_json};
        const hasAI = {'true' if has_ai else 'false'};
        
        function loadChart(s) {{
            document.getElementById('chart').src = 'charts/' + s + '.html';
            document.querySelectorAll('#tbl tr').forEach(r => r.classList.remove('active'));
            event.currentTarget.classList.add('active');
            
            // Show symbol AI overlay if available
            if (hasAI) {{
                showSymbolOverlay(s);
            }}
        }}
        
        function showSymbolOverlay(symbol) {{
            const overlay = document.getElementById('symbolOverlay');
            const analysis = aiData.symbol_analyses?.[symbol];
            
            if (!analysis) {{
                overlay.classList.remove('active');
                return;
            }}
            
            const conviction = analysis.conviction || 5;
            const convClass = conviction >= 7 ? 'high' : conviction <= 3 ? 'low' : 'medium';
            const rec = analysis.recommendation || 'HOLD';
            const narrative = analysis.narrative || '';
            const pros = (analysis.pros || []).map(p => `<li>✅ ${{p}}</li>`).join('');
            const cons = (analysis.cons || []).map(c => `<li>⚠️ ${{c}}</li>`).join('');
            const keyRisk = analysis.key_risk || '';
            const adjusted = analysis.conviction_adjusted_by_reviewer;
            const adjustNote = adjusted ? `<br><small>Adjusted from ${{analysis.conviction_original}} by reviewer: ${{analysis.adjustment_reason}}</small>` : '';
            
            overlay.innerHTML = `
                <div class="symbol-overlay-header">
                    <h4>🤖 AI Analysis: ${{symbol}}</h4>
                    <span class="conviction-badge ${{convClass}}">${{conviction}}/10 ${{rec}}</span>
                </div>
                <div class="symbol-narrative">${{narrative}}${{adjustNote}}</div>
                <div class="symbol-pros-cons">
                    <div><strong>Pros:</strong><ul>${{pros || '<li>None</li>'}}</ul></div>
                    <div><strong>Cons:</strong><ul>${{cons || '<li>None</li>'}}</ul></div>
                </div>
                ${{keyRisk ? `<div class="key-risk"><strong>Key Risk:</strong> ${{keyRisk}}</div>` : ''}}
            `;
            overlay.classList.add('active');
        }}
        
        function showAIModal() {{
            const modal = document.getElementById('aiModal');
            const body = document.getElementById('aiModalBody');
            
            const thematic = aiData.thematic_analysis || {{}};
            const review = aiData.review_results || {{}};
            const themes = thematic.themes || [];
            const summary = thematic.summary || 'No summary available';
            const sentiment = thematic.overall_sentiment || 'unknown';
            const reviewerNotes = review.reviewer_notes || '';
            const topPicks = (review.top_picks || []).join(', ') || 'None';
            const avoidList = (review.avoid_list || []).join(', ') || 'None';
            
            let themesHtml = themes.map(t => `
                <div class="theme-card">
                    <h4>${{t.name}}</h4>
                    <div class="theme-symbols">Symbols: ${{(t.symbols || []).join(', ')}}</div>
                    <p>${{t.narrative || ''}}</p>
                    <span class="theme-outlook ${{t.mean_reversion_outlook || 'neutral'}}">${{(t.mean_reversion_outlook || 'neutral').toUpperCase()}}</span>
                </div>
            `).join('');
            
            body.innerHTML = `
                <p><strong>Summary:</strong> ${{summary}}</p>
                <p><strong>Overall Sentiment:</strong> <span class="theme-outlook ${{sentiment}}">${{sentiment.toUpperCase()}}</span></p>
                <h3>Themes</h3>
                ${{themesHtml || '<p>No themes identified</p>'}}
                <h3>Reviewer Assessment</h3>
                <p><strong>Top Picks:</strong> ${{topPicks}}</p>
                <p><strong>Avoid:</strong> ${{avoidList}}</p>
                ${{reviewerNotes ? `<div class="reviewer-notes"><strong>Notes:</strong> ${{reviewerNotes}}</div>` : ''}}
            `;
            
            modal.classList.add('active');
        }}
        
        function hideAIModal() {{
            document.getElementById('aiModal').classList.remove('active');
        }}
        
        // Initialize
        document.querySelector('#tbl tr')?.classList.add('active');
        if (hasAI) {{
            const firstSymbol = document.querySelector('#tbl tr')?.dataset.symbol;
            if (firstSymbol) showSymbolOverlay(firstSymbol);
        }}
        
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape') {{ hideAIModal(); return; }}
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


# =============================================================================
# Main
# =============================================================================

@workflow_script("19.3-gen-trades")
def main():
    """Generate trade candidates using champion parameters."""
    
    print("=" * 70)
    print("CHAMPION CONFIGURATION (from 19.1b)")
    print("=" * 70)
    print(f"  Stop Loss:     {STOP_LOSS_PCNT}%")
    print(f"  Profit Target: {PROFIT_EXIT_PCNT}%")
    print(f"  Max Hold:      {MAX_HOLD_WEEKS} week(s)")
    print(f"  Min Loss:      {MIN_LOSS_PCNT}%")
    print(f"  Boost:         {BOOST_DIRECTION} ({BOOST_MULTIPLIER:.0%})")
    print()
    
    # Get regime
    regime = get_current_regime()
    print(f"Current Regime: {regime['regime'].upper()}")
    if "close" in regime:
        print(f"  {REGIME_SYMBOL}: ${regime['close']:.2f} vs MA{REGIME_MA_PERIOD}: ${regime['ma']:.2f}")
    
    # Determine if boost applies
    boost_active = (BOOST_DIRECTION == "bull" and regime.get("is_bull")) or \
                   (BOOST_DIRECTION == "bear" and not regime.get("is_bull"))
    if USE_REGIME and boost_active:
        print(f"  >>> BOOST ACTIVE: {BOOST_MULTIPLIER:.0%} position size")
    print()
    
    # Get latest week data
    print("Loading latest week data...")
    latest_week, week_str = get_latest_week_data()
    print(f"  Week: {week_str}")
    print(f"  ETFs: {len(latest_week)}")
    print()
    
    # Create output dir
    today = datetime.now().strftime("%Y-%m-%d")
    output_dir = OUTPUT_DIR / today
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Identify candidates
    candidates = identify_candidates(latest_week, regime)
    print(f"Candidates (bottom {BOTTOM_PERCENTILE*100:.0f}% with >={MIN_LOSS_PCNT}% loss): {len(candidates)}")
    
    if len(candidates) == 0:
        print()
        print("=" * 70)
        print("⚠️  NO CANDIDATES meet criteria this week.")
        print("=" * 70)
        print(f"  Min loss threshold: {MIN_LOSS_PCNT}%")
        print(f"  Bottom percentile: {BOTTOM_PERCENTILE*100:.0f}%")
        
        # Show what we would have had with lower threshold
        sorted_df = latest_week.sort_values("pct_return")
        n = max(1, int(len(sorted_df) * BOTTOM_PERCENTILE))
        bottom = sorted_df.head(n)
        print(f"\n  Worst {n} ETFs this week:")
        for i, (_, row) in enumerate(bottom.head(10).iterrows(), 1):
            print(f"    {i}. {row['symbol']}: {row['pct_return']:.2f}%")
        return
    
    # Print trade specs
    print()
    print("=" * 70)
    print("TRADE SPECIFICATIONS")
    print("=" * 70)
    total_capital = 0
    for i, (_, t) in enumerate(candidates.iterrows(), 1):
        badge = t.get("risk_badge", "")
        badge_str = f" {badge}" if badge else ""
        print(f"\n{i}. {t['symbol']}{badge_str} - {t.get('etf_name', '')[:40]}")
        print(f"   Return: {t['pct_return']:.2f}%")
        print(f"   >>> SHARES: {t['shares']}  |  LIMIT: ${t['limit_price']:.2f}")
        print(f"   Stop: ${t['stop_price']:.2f} (-{STOP_LOSS_PCNT}%)")
        print(f"   Target: ${t['target_price']:.2f} (+{PROFIT_EXIT_PCNT}%)")
        
        # Show volatility profile
        sigma = t.get("sigma", np.nan)
        beta = t.get("beta", np.nan)
        atr = t.get("atr_pct", np.nan)
        if pd.notna(sigma) or pd.notna(beta):
            vol_parts = []
            if pd.notna(sigma):
                vol_parts.append(f"σ={sigma:.1f}%")
            if pd.notna(beta):
                vol_parts.append(f"β={beta:.2f}")
            if pd.notna(atr):
                vol_parts.append(f"ATR={atr:.1f}%")
            print(f"   Vol: {' | '.join(vol_parts)}")
        
        total_capital += t["position_usd"]
    
    print()
    print(f"Total Capital Required: ${total_capital:,.2f} / ${INITIAL_CAPITAL:,.2f}")
    
    # Print risk notes
    risk_notes = get_risk_notes(candidates)
    if risk_notes:
        print()
        print("⚠️  RISK NOTES:")
        for note in risk_notes:
            # Strip HTML tags for terminal
            clean_note = note.replace("<b>", "").replace("</b>", "")
            print(f"   {clean_note}")
    print()
    
    # Create charts
    print("Creating charts...")
    for _, t in candidates.iterrows():
        create_chart(t["symbol"], t.get("etf_name", ""), t, charts_dir)
    
    # Create inspector
    inspector_path = create_inspector(candidates, regime, week_str, output_dir)
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
        "champion_config": {
            "stop_loss_pcnt": STOP_LOSS_PCNT,
            "profit_exit_pcnt": PROFIT_EXIT_PCNT,
            "max_hold_weeks": MAX_HOLD_WEEKS,
            "min_loss_pcnt": MIN_LOSS_PCNT,
            "boost_direction": BOOST_DIRECTION,
            "boost_multiplier": BOOST_MULTIPLIER,
            "bottom_percentile": BOTTOM_PERCENTILE,
            "initial_capital": INITIAL_CAPITAL,
            "max_active_trades": MAX_ACTIVE_TRADES,
        },
        "n_candidates": len(candidates),
        "symbols": candidates["symbol"].tolist(),
        "total_capital": total_capital,
        "boost_active": boost_active,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print()
    print("=" * 70)
    print("✅ COMPLETE")
    print("=" * 70)
    print(f"  Output: {output_dir}")
    
    # Open inspector
    if sys.platform == "darwin":
        subprocess.run(["open", str(inspector_path)])
    elif sys.platform == "linux":
        subprocess.run(["xdg-open", str(inspector_path)])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
15-run-portfolio-backtest.py

Portfolio-level backtest of mean-reversion strategy with:
1. Multiple portfolio sizes (K=5, 10, 20)
2. Multiple weighting schemes (equal, volatility-weighted)
3. Multiple holding periods (1, 2, 4 weeks)
4. Transaction cost modeling
5. Regime filtering (unconditional vs VIX-filtered vs trend-filtered)
6. IC concentration analysis by decile

This is exp015 in the research grid.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.workflow.config import DATA_TIER
from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp015_portfolio_backtest"
MIN_HISTORY_WEEKS = 52

# Feature matrix path
FEATURE_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / DATA_TIER / "feature_matrix.parquet"
WEEKLY_DATA_DIR = PROJECT_ROOT / "data" / "historical" / DATA_TIER / "weekly"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "experiments" / EXPERIMENT_NAME

# Portfolio parameters to test
PORTFOLIO_SIZES = [5, 10, 20]
WEIGHTING_SCHEMES = ["equal", "inverse_volatility"]
HOLDING_PERIODS = [1, 2, 4]  # weeks

# Transaction costs (one-way, in basis points)
TRANSACTION_COST_BPS = 10  # 10 bps = 0.1% per trade

# Regime thresholds (from exp009 regime analysis)
VIX_MEDIUM_LOW = 15  # Below this = low VIX
VIX_MEDIUM_HIGH = 25  # Above this = high VIX
TREND_THRESHOLD = 0.02  # Momentum threshold for flat/up/down


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load feature matrix and macro data."""
    # Feature matrix
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[df["category"] == "target"].copy()
    
    # Macro data for regime
    macro_data = {}
    for symbol in ["VIXY", "SPY"]:
        csv_path = WEEKLY_DATA_DIR / f"{symbol}.csv"
        if csv_path.exists():
            macro_df = pd.read_csv(csv_path, parse_dates=["week_start"])
            macro_df = macro_df.set_index("week_start").sort_index()
            macro_data[symbol] = macro_df
    
    # Build regime dataframe
    weeks = df[["week_start", "week_idx"]].drop_duplicates().sort_values("week_idx")
    regime_records = []
    
    for _, row in weeks.iterrows():
        week = row["week_start"]
        week_idx = row["week_idx"]
        record = {"week_idx": week_idx, "week_start": week}
        
        # VIX level
        if "VIXY" in macro_data:
            vixy = macro_data["VIXY"]
            prior = vixy.index[vixy.index <= week]
            if len(prior) > 0:
                record["vix_level"] = vixy.loc[prior[-1], "close"]
            else:
                record["vix_level"] = np.nan
        else:
            record["vix_level"] = np.nan
        
        # SPY momentum (4-week)
        if "SPY" in macro_data:
            spy = macro_data["SPY"]
            if week in spy.index and "momentum_4w" in spy.columns:
                record["spy_momentum"] = spy.loc[week, "momentum_4w"]
            else:
                prior = spy.index[spy.index <= week]
                if len(prior) > 0 and "momentum_4w" in spy.columns:
                    record["spy_momentum"] = spy.loc[prior[-1], "momentum_4w"]
                else:
                    record["spy_momentum"] = np.nan
        else:
            record["spy_momentum"] = np.nan
        
        regime_records.append(record)
    
    regime_df = pd.DataFrame(regime_records)
    
    # Classify regimes
    regime_df["vix_regime"] = pd.cut(
        regime_df["vix_level"],
        bins=[0, VIX_MEDIUM_LOW, VIX_MEDIUM_HIGH, np.inf],
        labels=["low", "medium", "high"]
    )
    
    regime_df["trend_regime"] = pd.cut(
        regime_df["spy_momentum"],
        bins=[-np.inf, -TREND_THRESHOLD, TREND_THRESHOLD, np.inf],
        labels=["down", "flat", "up"]
    )
    
    return df, regime_df


def compute_volatility(df: pd.DataFrame, week_idx: int, lookback: int = 12) -> pd.Series:
    """Compute trailing volatility for each symbol."""
    # Get prior weeks
    prior_data = df[(df["week_idx"] >= week_idx - lookback) & (df["week_idx"] < week_idx)]
    
    # Compute std of returns per symbol
    vol = prior_data.groupby("symbol")["log_return"].std()
    
    return vol


def select_portfolio(
    df: pd.DataFrame,
    week_idx: int,
    k: int,
    weighting: str,
) -> pd.DataFrame:
    """
    Select top-K mean-reversion portfolio for a given week.
    
    Returns DataFrame with symbol, weight, and expected holding return.
    """
    # Get prior week's returns (signal)
    prior_week = df[df["week_idx"] == week_idx - 1][["symbol", "log_return"]].copy()
    prior_week = prior_week.rename(columns={"log_return": "signal"})
    
    if len(prior_week) < k:
        return pd.DataFrame()
    
    # Mean-reversion: buy losers (lowest prior return)
    prior_week = prior_week.sort_values("signal").head(k)
    
    # Compute weights
    if weighting == "equal":
        prior_week["weight"] = 1.0 / k
    elif weighting == "inverse_volatility":
        vol = compute_volatility(df, week_idx)
        prior_week = prior_week.merge(
            vol.rename("volatility").reset_index(),
            on="symbol",
            how="left"
        )
        # Inverse volatility weight (handle missing/zero vol)
        prior_week["volatility"] = prior_week["volatility"].fillna(prior_week["volatility"].median())
        prior_week["volatility"] = prior_week["volatility"].replace(0, prior_week["volatility"].median())
        prior_week["inv_vol"] = 1.0 / prior_week["volatility"]
        prior_week["weight"] = prior_week["inv_vol"] / prior_week["inv_vol"].sum()
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    
    return prior_week[["symbol", "weight"]]


def compute_portfolio_return(
    df: pd.DataFrame,
    portfolio: pd.DataFrame,
    entry_week_idx: int,
    holding_period: int,
) -> float:
    """
    Compute portfolio return over holding period.
    
    Returns total log return.
    """
    if len(portfolio) == 0:
        return 0.0
    
    # Get returns for holding period
    hold_weeks = range(entry_week_idx, entry_week_idx + holding_period)
    
    total_return = 0.0
    
    for week_idx in hold_weeks:
        week_data = df[df["week_idx"] == week_idx][["symbol", "log_return"]]
        merged = portfolio.merge(week_data, on="symbol", how="left")
        merged["log_return"] = merged["log_return"].fillna(0)
        
        # Weighted return
        week_return = (merged["weight"] * merged["log_return"]).sum()
        total_return += week_return
    
    return total_return


def run_backtest(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    k: int,
    weighting: str,
    holding_period: int,
    regime_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run portfolio backtest with given parameters.
    
    Args:
        regime_filter: None (unconditional), "vix_medium", "flat_trend", "both"
    
    Returns DataFrame with weekly portfolio returns.
    """
    week_indices = sorted(df["week_idx"].unique())
    prediction_weeks = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    # For holding periods > 1, we need to skip weeks
    trade_weeks = prediction_weeks[::holding_period]
    
    results = []
    
    for week_idx in trade_weeks:
        # Check regime filter
        regime_row = regime_df[regime_df["week_idx"] == week_idx]
        if len(regime_row) == 0:
            continue
        
        regime_row = regime_row.iloc[0]
        
        skip_trade = False
        if regime_filter == "vix_medium" and regime_row["vix_regime"] != "medium":
            skip_trade = True
        elif regime_filter == "flat_trend" and regime_row["trend_regime"] != "flat":
            skip_trade = True
        elif regime_filter == "both":
            if regime_row["vix_regime"] != "medium" or regime_row["trend_regime"] != "flat":
                skip_trade = True
        
        # Get week_start
        week_start = regime_row["week_start"]
        
        if skip_trade:
            results.append({
                "week_idx": week_idx,
                "week_start": week_start,
                "traded": False,
                "gross_return": 0.0,
                "transaction_cost": 0.0,
                "net_return": 0.0,
                "n_positions": 0,
            })
            continue
        
        # Select portfolio
        portfolio = select_portfolio(df, week_idx, k, weighting)
        
        if len(portfolio) == 0:
            results.append({
                "week_idx": week_idx,
                "week_start": week_start,
                "traded": False,
                "gross_return": 0.0,
                "transaction_cost": 0.0,
                "net_return": 0.0,
                "n_positions": 0,
            })
            continue
        
        # Compute return
        gross_return = compute_portfolio_return(df, portfolio, week_idx, holding_period)
        
        # Transaction costs: entry + exit (both sides)
        # For K positions, we trade 2K times (buy and sell)
        transaction_cost = 2 * k * (TRANSACTION_COST_BPS / 10000)
        
        net_return = gross_return - transaction_cost
        
        results.append({
            "week_idx": week_idx,
            "week_start": week_start,
            "traded": True,
            "gross_return": gross_return,
            "transaction_cost": transaction_cost,
            "net_return": net_return,
            "n_positions": len(portfolio),
        })
    
    return pd.DataFrame(results)


def analyze_ic_by_decile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute IC by decile to check concentration of signal.
    """
    week_indices = sorted(df["week_idx"].unique())
    prediction_weeks = [w for w in week_indices if w >= MIN_HISTORY_WEEKS]
    
    decile_results = []
    
    for week_idx in prediction_weeks:
        # Prior week return (signal)
        prior = df[df["week_idx"] == week_idx - 1][["symbol", "log_return"]].copy()
        prior = prior.rename(columns={"log_return": "signal"})
        
        # Current week return (outcome)
        curr = df[df["week_idx"] == week_idx][["symbol", "log_return"]].copy()
        curr = curr.rename(columns={"log_return": "outcome"})
        
        merged = prior.merge(curr, on="symbol", how="inner")
        
        if len(merged) < 30:
            continue
        
        # Compute alpha (market-neutral)
        merged["alpha"] = merged["outcome"] - merged["outcome"].mean()
        
        # Assign deciles based on signal (prior return)
        merged["decile"] = pd.qcut(merged["signal"], 10, labels=False, duplicates="drop")
        
        # Mean alpha by decile
        for decile in range(10):
            decile_data = merged[merged["decile"] == decile]
            if len(decile_data) > 0:
                decile_results.append({
                    "week_idx": week_idx,
                    "decile": decile,
                    "mean_alpha": decile_data["alpha"].mean(),
                    "n": len(decile_data),
                })
    
    return pd.DataFrame(decile_results)


def generate_report(
    all_results: dict,
    decile_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate comprehensive backtest report."""
    report_lines = [
        f"# Portfolio Backtest Report: {EXPERIMENT_NAME}",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Strategy",
        "",
        "**Mean-Reversion:** Each week, buy the K worst performers from last week.",
        "",
        f"- Transaction cost: {TRANSACTION_COST_BPS} bps per trade (one-way)",
        f"- Portfolio sizes tested: {PORTFOLIO_SIZES}",
        f"- Weighting schemes: {WEIGHTING_SCHEMES}",
        f"- Holding periods: {HOLDING_PERIODS} weeks",
        "",
        "---",
        "",
        "## 1. Unconditional Performance",
        "",
        "### Gross Returns (before costs)",
        "",
        "| K | Weighting | Hold | Ann. Return | Ann. Vol | Sharpe | Max DD |",
        "|---|-----------|------|-------------|----------|--------|--------|",
    ]
    
    # Summarize unconditional results
    for key, result_df in all_results.items():
        if not key.startswith("unconditional_"):
            continue
        
        # Parse key: unconditional_k10_equal_h1 or unconditional_k10_inverse_volatility_h1
        parts = key.split("_")
        k = parts[1].replace("k", "")
        hold = parts[-1].replace("h", "")
        weighting = "_".join(parts[2:-1])
        
        traded = result_df[result_df["traded"]]
        if len(traded) == 0:
            continue
        
        # Annualized metrics (52 weeks per year)
        periods_per_year = 52 / int(hold)
        mean_return = traded["gross_return"].mean()
        std_return = traded["gross_return"].std()
        
        ann_return = mean_return * periods_per_year
        ann_vol = std_return * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cumulative = (1 + traded["gross_return"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        report_lines.append(
            f"| {k} | {weighting} | {hold}w | {ann_return:.1%} | {ann_vol:.1%} | "
            f"{sharpe:.2f} | {max_dd:.1%} |"
        )
    
    # Net returns
    report_lines.extend([
        "",
        "### Net Returns (after costs)",
        "",
        "| K | Weighting | Hold | Ann. Return | Ann. Vol | Sharpe | Trades/Yr |",
        "|---|-----------|------|-------------|----------|--------|-----------|",
    ])
    
    for key, result_df in all_results.items():
        if not key.startswith("unconditional_"):
            continue
        
        # Parse key: unconditional_k10_equal_h1 or unconditional_k10_inverse_volatility_h1
        parts = key.split("_")
        k = parts[1].replace("k", "")
        hold = parts[-1].replace("h", "")
        weighting = "_".join(parts[2:-1])
        
        traded = result_df[result_df["traded"]]
        if len(traded) == 0:
            continue
        
        periods_per_year = 52 / int(hold)
        mean_return = traded["net_return"].mean()
        std_return = traded["net_return"].std()
        
        ann_return = mean_return * periods_per_year
        ann_vol = std_return * np.sqrt(periods_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        trades_per_year = periods_per_year * int(k) * 2  # entry + exit
        
        report_lines.append(
            f"| {k} | {weighting} | {hold}w | {ann_return:.1%} | {ann_vol:.1%} | "
            f"{sharpe:.2f} | {trades_per_year:.0f} |"
        )
    
    # Regime-filtered results
    report_lines.extend([
        "",
        "---",
        "",
        "## 2. Regime-Filtered Performance",
        "",
        "Using K=10, equal-weight, 1-week hold as baseline.",
        "",
        "| Filter | Weeks Traded | % of Total | Ann. Return (net) | Sharpe |",
        "|--------|--------------|------------|-------------------|--------|",
    ])
    
    baseline_key = "unconditional_k10_equal_h1"
    if baseline_key in all_results:
        baseline = all_results[baseline_key]
        total_weeks = len(baseline)
        
        for filter_name in [None, "vix_medium", "flat_trend", "both"]:
            if filter_name is None:
                key = baseline_key
                label = "Unconditional"
            else:
                key = f"{filter_name}_k10_equal_h1"
                label = filter_name.replace("_", " ").title()
            
            if key not in all_results:
                continue
            
            result_df = all_results[key]
            traded = result_df[result_df["traded"]]
            
            weeks_traded = len(traded)
            pct_traded = weeks_traded / total_weeks if total_weeks > 0 else 0
            
            if len(traded) > 0:
                ann_return = traded["net_return"].mean() * 52
                ann_vol = traded["net_return"].std() * np.sqrt(52)
                sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            else:
                ann_return = 0
                sharpe = 0
            
            report_lines.append(
                f"| {label} | {weeks_traded} | {pct_traded:.0%} | {ann_return:.1%} | {sharpe:.2f} |"
            )
    
    # Decile analysis
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. IC Concentration by Decile",
        "",
        "Does the mean-reversion signal concentrate in the tails?",
        "",
        "| Decile | Mean Alpha | Interpretation |",
        "|--------|------------|----------------|",
    ])
    
    decile_summary = decile_df.groupby("decile")["mean_alpha"].mean()
    
    for decile in range(10):
        if decile in decile_summary.index:
            alpha = decile_summary[decile]
            if decile == 0:
                interp = "← Biggest losers (BUY)"
            elif decile == 9:
                interp = "← Biggest winners (avoid)"
            else:
                interp = ""
            
            report_lines.append(f"| {decile} | {alpha:.4f} | {interp} |")
    
    # Compute spread
    if 0 in decile_summary.index and 9 in decile_summary.index:
        spread = decile_summary[0] - decile_summary[9]
        report_lines.extend([
            "",
            f"**Long-Short Spread (D0 - D9):** {spread:.4f}",
            "",
            f"This represents the expected weekly alpha from buying bottom decile vs top decile.",
        ])
    
    # Conclusions
    report_lines.extend([
        "",
        "---",
        "",
        "## Conclusions",
        "",
    ])
    
    # Find best configuration
    best_sharpe = -np.inf
    best_config = None
    
    for key, result_df in all_results.items():
        traded = result_df[result_df["traded"]]
        if len(traded) > 10:
            ann_return = traded["net_return"].mean() * 52
            ann_vol = traded["net_return"].std() * np.sqrt(52)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = key
    
    if best_config:
        report_lines.append(f"**Best Configuration:** `{best_config}` (Sharpe: {best_sharpe:.2f})")
    
    report_lines.append("")
    report_lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines))
    logging.info(f"Report saved: {report_path}")


@workflow_script("15-run-portfolio-backtest")
def main() -> None:
    """Run portfolio backtest."""
    logging.info(f"Starting {EXPERIMENT_NAME}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logging.info("Loading data...")
    df, regime_df = load_data()
    logging.info(f"Feature matrix: {len(df)} rows, {df['week_idx'].nunique()} weeks")
    logging.info(f"Regime data: {len(regime_df)} weeks")
    
    # Run all backtest configurations
    all_results = {}
    
    # Unconditional backtests
    logging.info("Running unconditional backtests...")
    for k in PORTFOLIO_SIZES:
        for weighting in WEIGHTING_SCHEMES:
            for hold in HOLDING_PERIODS:
                key = f"unconditional_k{k}_{weighting}_h{hold}"
                logging.info(f"  {key}")
                results = run_backtest(df, regime_df, k, weighting, hold, regime_filter=None)
                all_results[key] = results
    
    # Regime-filtered backtests (K=10, equal, 1-week only)
    logging.info("Running regime-filtered backtests...")
    for regime_filter in ["vix_medium", "flat_trend", "both"]:
        key = f"{regime_filter}_k10_equal_h1"
        logging.info(f"  {key}")
        results = run_backtest(df, regime_df, 10, "equal", 1, regime_filter=regime_filter)
        all_results[key] = results
    
    # IC by decile analysis
    logging.info("Analyzing IC by decile...")
    decile_df = analyze_ic_by_decile(df)
    
    # Save all results
    for key, result_df in all_results.items():
        result_df.to_csv(OUTPUT_DIR / f"{key}.csv", index=False)
    
    decile_df.to_csv(OUTPUT_DIR / "decile_analysis.csv", index=False)
    
    # Save config
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "portfolio_sizes": PORTFOLIO_SIZES,
        "weighting_schemes": WEIGHTING_SCHEMES,
        "holding_periods": HOLDING_PERIODS,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "generated": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate report
    generate_report(all_results, decile_df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"PORTFOLIO BACKTEST: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    print("\nTop Configurations (by Sharpe):")
    print("-" * 50)
    
    sharpes = []
    for key, result_df in all_results.items():
        traded = result_df[result_df["traded"]]
        if len(traded) > 10:
            ann_return = traded["net_return"].mean() * 52
            ann_vol = traded["net_return"].std() * np.sqrt(52)
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            sharpes.append((key, sharpe, ann_return, len(traded)))
    
    sharpes.sort(key=lambda x: x[1], reverse=True)
    
    for key, sharpe, ann_ret, n_trades in sharpes[:5]:
        print(f"  {key}: Sharpe={sharpe:.2f}, Ann.Ret={ann_ret:.1%}, N={n_trades}")
    
    print("\n" + "-" * 50)
    print("Decile Analysis (mean alpha):")
    decile_means = decile_df.groupby("decile")["mean_alpha"].mean()
    for d in [0, 1, 8, 9]:
        if d in decile_means.index:
            print(f"  Decile {d}: {decile_means[d]:.4f}")
    
    if 0 in decile_means.index and 9 in decile_means.index:
        print(f"  Long-Short Spread: {decile_means[0] - decile_means[9]:.4f}")
    
    print("=" * 70)
    logging.info("Experiment complete")


if __name__ == "__main__":
    main()

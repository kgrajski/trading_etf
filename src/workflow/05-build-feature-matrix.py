#!/usr/bin/env python3
"""
05-build-feature-matrix.py

Build a long-format feature matrix for rolling-window cross-sectional modeling.

Each row represents one symbol at one week, containing:
- Metadata: symbol, name, category, week_start
- Positional encoding: week_idx, week_of_year, month, week_of_month, quarter, year
- Features: all L1/L2 derived features
- Target: log_return for the FOLLOWING week (shifted)

This format supports:
- Cross-sectional models (predict all symbols for a given week)
- Time-series models (sequences of weeks for each symbol)
- Symbol-independent training (pool all symbols together)

Output:
- data/processed/{tier}/feature_matrix.parquet (primary)
- data/processed/{tier}/feature_matrix_sample.csv (first 1000 rows for inspection)
- data/processed/{tier}/feature_matrix_config.json (metadata)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.workflow.config import (
    DATA_TIER,
    MACRO_SYMBOLS,
    SYMBOL_PREFIX_FILTER,
)
from src.workflow.workflow_utils import (
    get_historical_dir,
    get_metadata_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()

# =============================================================================
# Configuration
# =============================================================================

# Minimum weeks of history required to include a symbol-week in the matrix
MIN_HISTORY_WEEKS = 12  # Need 12 weeks for momentum_12w to be valid

# Feature columns to include (must match 03-generate-features.py output)
FEATURE_COLS = [
    # L1: Single-week derived
    "log_return",
    "log_return_intraweek",
    "log_range",
    "log_volume",
    "log_avg_daily_volume",
    "intra_week_volatility",
    # L2: Temporal/rolling
    "log_return_ma4",
    "log_return_ma12",
    "log_volume_ma4",
    "log_volume_ma12",
    "momentum_4w",
    "momentum_12w",
    "volatility_ma4",
    "volatility_ma12",
    "log_volume_delta",
]

# Metadata columns
METADATA_COLS = ["symbol", "name", "category", "week_start"]

# Positional encoding columns - integer (human-readable, tree models)
POSITIONAL_INT_COLS = [
    "week_idx",       # Sequential integer (global ordering)
    "week_of_year",   # 1-52 (ISO week number)
    "month",          # 1-12
    "week_of_month",  # 1-4
    "quarter",        # 1-4
    "year",           # e.g., 2024
]

# Positional encoding columns - sin/cos (neural networks, time-series models)
# Each periodicity gets a sin/cos pair to preserve cyclical relationships
POSITIONAL_SINCOS_COLS = [
    # Week of year (period=52)
    "pos_week_of_year_sin",
    "pos_week_of_year_cos",
    # Month (period=12)
    "pos_month_sin",
    "pos_month_cos",
    # Week of month (period=4)
    "pos_week_of_month_sin",
    "pos_week_of_month_cos",
    # Quarter (period=4)
    "pos_quarter_sin",
    "pos_quarter_cos",
    # Global position - multiple frequencies for long-range patterns
    # Using periods: 52 (annual), 26 (semi-annual), 13 (quarterly), 4 (monthly)
    "pos_global_52_sin",
    "pos_global_52_cos",
    "pos_global_26_sin",
    "pos_global_26_cos",
    "pos_global_13_sin",
    "pos_global_13_cos",
    "pos_global_4_sin",
    "pos_global_4_cos",
]

POSITIONAL_COLS = POSITIONAL_INT_COLS + POSITIONAL_SINCOS_COLS

TARGET_COL = "target_return"


# =============================================================================
# Helper Functions
# =============================================================================

def load_symbol_metadata(metadata_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata for all symbols (targets + macros).
    
    Returns:
        Dict mapping symbol -> {name, category}
    """
    metadata = {}
    
    # Load target ETFs
    filtered_path = metadata_dir / "filtered_etfs.json"
    if filtered_path.exists():
        with open(filtered_path) as f:
            data = json.load(f)
            # Handle both {"etfs": [...]} and [...] formats
            etf_list = data.get("etfs", data) if isinstance(data, dict) else data
            for etf in etf_list:
                metadata[etf["symbol"]] = {
                    "name": etf.get("name", ""),
                    "category": "target",
                }
    
    # Add macro symbols from config
    for category, symbols in MACRO_SYMBOLS.items():
        for symbol, name in symbols.items():
            metadata[symbol] = {
                "name": name,
                "category": category,
            }
    
    return metadata


def compute_positional_encoding(week_start: pd.Timestamp, week_idx: int) -> Dict[str, float]:
    """Compute positional encoding features for a week.
    
    Includes both integer encodings (for human readability, tree models)
    and sin/cos encodings (for neural networks, time-series models).
    
    Args:
        week_start: Monday date of the week
        week_idx: Global week index (0, 1, 2, ...)
        
    Returns:
        Dict with positional encoding values
    """
    iso_cal = week_start.isocalendar()
    
    # Integer encodings
    week_of_year = iso_cal.week  # 1-52
    month = week_start.month  # 1-12
    day_of_month = week_start.day
    week_of_month = min((day_of_month - 1) // 7 + 1, 4)  # 1-4
    quarter = (week_start.month - 1) // 3 + 1  # 1-4
    year = week_start.year
    
    # Sin/cos helper - converts position to sin/cos for given period
    def sincos(position: float, period: float) -> Tuple[float, float]:
        angle = 2 * np.pi * position / period
        return np.sin(angle), np.cos(angle)
    
    # Compute sin/cos encodings for each periodicity
    woy_sin, woy_cos = sincos(week_of_year, 52)
    month_sin, month_cos = sincos(month, 12)
    wom_sin, wom_cos = sincos(week_of_month, 4)
    quarter_sin, quarter_cos = sincos(quarter, 4)
    
    # Global position with multiple frequencies
    # These capture long-range patterns at different scales
    g52_sin, g52_cos = sincos(week_idx, 52)   # Annual cycle
    g26_sin, g26_cos = sincos(week_idx, 26)   # Semi-annual
    g13_sin, g13_cos = sincos(week_idx, 13)   # Quarterly
    g4_sin, g4_cos = sincos(week_idx, 4)      # Monthly
    
    return {
        # Integer encodings
        "week_of_year": week_of_year,
        "month": month,
        "week_of_month": week_of_month,
        "quarter": quarter,
        "year": year,
        # Sin/cos encodings - cyclical
        "pos_week_of_year_sin": woy_sin,
        "pos_week_of_year_cos": woy_cos,
        "pos_month_sin": month_sin,
        "pos_month_cos": month_cos,
        "pos_week_of_month_sin": wom_sin,
        "pos_week_of_month_cos": wom_cos,
        "pos_quarter_sin": quarter_sin,
        "pos_quarter_cos": quarter_cos,
        # Sin/cos encodings - global position (multiple frequencies)
        "pos_global_52_sin": g52_sin,
        "pos_global_52_cos": g52_cos,
        "pos_global_26_sin": g26_sin,
        "pos_global_26_cos": g26_cos,
        "pos_global_13_sin": g13_sin,
        "pos_global_13_cos": g13_cos,
        "pos_global_4_sin": g4_sin,
        "pos_global_4_cos": g4_cos,
    }


def load_weekly_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Load weekly feature data for a symbol.
    
    Args:
        filepath: Path to weekly CSV
        
    Returns:
        DataFrame or None if loading fails
    """
    try:
        df = pd.read_csv(filepath)
        df["week_start"] = pd.to_datetime(df["week_start"])
        return df.sort_values("week_start").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Failed to load {filepath.name}: {e}")
        return None


def build_symbol_rows(
    df: pd.DataFrame,
    symbol: str,
    name: str,
    category: str,
    global_week_map: Dict[pd.Timestamp, int],
) -> List[Dict[str, Any]]:
    """Build feature matrix rows for a single symbol.
    
    Args:
        df: Weekly feature DataFrame for the symbol
        symbol: Ticker symbol
        name: Symbol description
        category: Symbol category
        global_week_map: Mapping from week_start to global week_idx
        
    Returns:
        List of row dicts
    """
    rows = []
    
    # Create shifted target (next week's return)
    df = df.copy()
    df["target_return"] = df["log_return"].shift(-1)
    
    for idx, row in df.iterrows():
        week_start = row["week_start"]
        
        # Skip if target is NaN (last week)
        if pd.isna(row["target_return"]):
            continue
        
        # Skip if not enough history (features will have NaN)
        if idx < MIN_HISTORY_WEEKS:
            continue
        
        # Check if all features are valid
        feature_values = [row.get(col) for col in FEATURE_COLS]
        if any(pd.isna(v) for v in feature_values):
            continue
        
        # Get global week index
        week_idx = global_week_map.get(week_start, -1)
        
        # Build row
        row_dict = {
            # Metadata
            "symbol": symbol,
            "name": name,
            "category": category,
            "week_start": week_start,
            # Positional encoding (pass week_idx for global position sin/cos)
            "week_idx": week_idx,
            **compute_positional_encoding(week_start, week_idx),
            # Features
            **{col: row[col] for col in FEATURE_COLS if col in row},
            # Target
            "target_return": row["target_return"],
        }
        
        rows.append(row_dict)
    
    return rows


def build_global_week_index(weekly_dir: Path) -> Tuple[Dict[pd.Timestamp, int], List[pd.Timestamp]]:
    """Build a global week index from all available data.
    
    Returns:
        Tuple of (week_start -> week_idx mapping, sorted list of all weeks)
    """
    all_weeks = set()
    
    for csv_file in weekly_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, usecols=["week_start"])
            weeks = pd.to_datetime(df["week_start"])
            all_weeks.update(weeks)
        except Exception:
            continue
    
    sorted_weeks = sorted(all_weeks)
    week_map = {w: i for i, w in enumerate(sorted_weeks)}
    
    return week_map, sorted_weeks


# =============================================================================
# Main
# =============================================================================

@workflow_script("05-build-feature-matrix")
def main() -> None:
    """Main workflow function."""
    # Paths
    metadata_dir = get_metadata_dir()
    weekly_dir = get_historical_dir(DATA_TIER) / "weekly"
    output_dir = Path("data/processed") / DATA_TIER
    os.makedirs(output_dir, exist_ok=True)
    
    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter: {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Min history weeks: {MIN_HISTORY_WEEKS}")
    print(f"  Input: {weekly_dir}")
    print(f"  Output: {output_dir}")
    print()
    
    if not weekly_dir.exists():
        logger.error(f"Weekly data directory not found: {weekly_dir}")
        logger.error("Please run 03-generate-features.py first.")
        return
    
    # Load metadata
    print("Loading symbol metadata...")
    symbol_metadata = load_symbol_metadata(metadata_dir)
    print(f"  Loaded metadata for {len(symbol_metadata)} symbols")
    
    # Build global week index
    print("Building global week index...")
    global_week_map, all_weeks = build_global_week_index(weekly_dir)
    print(f"  Found {len(all_weeks)} unique weeks")
    if all_weeks:
        print(f"  Range: {all_weeks[0].strftime('%Y-%m-%d')} to {all_weeks[-1].strftime('%Y-%m-%d')}")
    print()
    
    # Get files to process
    csv_files = sorted(weekly_dir.glob("*.csv"))
    if SYMBOL_PREFIX_FILTER:
        csv_files = [f for f in csv_files if f.stem.startswith(SYMBOL_PREFIX_FILTER)]
    
    print(f"Processing {len(csv_files)} symbol files...")
    print("-" * 80)
    
    # Build feature matrix
    all_rows = []
    symbols_processed = 0
    symbols_with_data = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        symbol = csv_file.stem
        meta = symbol_metadata.get(symbol, {"name": "", "category": "unknown"})
        
        if i % 100 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}] Processing {symbol}...")
        
        df = load_weekly_data(csv_file)
        if df is None or df.empty:
            continue
        
        symbols_processed += 1
        
        rows = build_symbol_rows(
            df=df,
            symbol=symbol,
            name=meta["name"],
            category=meta["category"],
            global_week_map=global_week_map,
        )
        
        if rows:
            all_rows.extend(rows)
            symbols_with_data += 1
    
    print()
    print(f"Built {len(all_rows)} rows from {symbols_with_data} symbols")
    
    if not all_rows:
        logger.error("No valid rows generated. Check data and configuration.")
        return
    
    # Create DataFrame
    print()
    print("Creating feature matrix DataFrame...")
    feature_matrix = pd.DataFrame(all_rows)
    
    # Sort by week_idx, then symbol (useful for time-series access)
    feature_matrix = feature_matrix.sort_values(
        ["week_idx", "symbol"]
    ).reset_index(drop=True)
    
    # Compute statistics
    n_weeks = feature_matrix["week_idx"].nunique()
    n_symbols = feature_matrix["symbol"].nunique()
    n_features = len(FEATURE_COLS)
    
    # Rows per week statistics
    rows_per_week = feature_matrix.groupby("week_idx").size()
    
    print(f"  Shape: {feature_matrix.shape}")
    print(f"  Unique weeks: {n_weeks}")
    print(f"  Unique symbols: {n_symbols}")
    print(f"  Features per row: {n_features}")
    print(f"  Rows per week: min={rows_per_week.min()}, max={rows_per_week.max()}, mean={rows_per_week.mean():.1f}")
    
    # Save outputs
    print()
    print("Saving outputs...")
    
    # Parquet (primary)
    parquet_path = output_dir / "feature_matrix.parquet"
    feature_matrix.to_parquet(parquet_path, index=False)
    parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
    print(f"  Parquet: {parquet_path} ({parquet_size:.1f} MB)")
    
    # CSV sample (for inspection)
    csv_sample_path = output_dir / "feature_matrix_sample.csv"
    feature_matrix.head(1000).to_csv(csv_sample_path, index=False)
    print(f"  CSV sample: {csv_sample_path} (first 1000 rows)")
    
    # Config JSON (metadata)
    config = {
        "generated_at": datetime.now().isoformat(),
        "data_tier": DATA_TIER,
        "n_rows": len(feature_matrix),
        "n_weeks": n_weeks,
        "n_symbols": n_symbols,
        "n_features": n_features,
        "min_history_weeks": MIN_HISTORY_WEEKS,
        "week_range": {
            "start": all_weeks[0].strftime("%Y-%m-%d") if all_weeks else None,
            "end": all_weeks[-1].strftime("%Y-%m-%d") if all_weeks else None,
        },
        "columns": {
            "metadata": METADATA_COLS,
            "positional_int": POSITIONAL_INT_COLS,
            "positional_sincos": POSITIONAL_SINCOS_COLS,
            "features": FEATURE_COLS,
            "target": TARGET_COL,
        },
        "rows_per_week_stats": {
            "min": int(rows_per_week.min()),
            "max": int(rows_per_week.max()),
            "mean": float(rows_per_week.mean()),
        },
    }
    
    config_path = output_dir / "feature_matrix_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    
    print()
    print_summary(
        total_rows=len(feature_matrix),
        unique_weeks=n_weeks,
        unique_symbols=n_symbols,
        features_per_row=n_features,
        min_rows_per_week=int(rows_per_week.min()),
        max_rows_per_week=int(rows_per_week.max()),
        parquet_size_mb=f"{parquet_size:.1f}",
        output_directory=str(output_dir),
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build cross-sectional feature matrix for prediction experiments.

Creates aligned feature matrices from weekly ETF data where:
- Rows = weeks (aligned across all symbols)
- Columns = features from all symbols (cross-sectional)

Features include:
- Target symbol features (what we're predicting)
- Macro symbol features (predictors only, not prediction targets)
- Specialized cross-symbol features (VIX term structure, yield curve slope, etc.)

This enables predicting symbol X using features from all symbols + macro indicators.

Input: data/historical/{tier}/weekly/*.csv
Output: data/features/{tier}/feature_matrix.parquet, target_matrix.parquet
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from src.training.feature_builder import FeatureBuilder
from src.workflow.config import (
    DATA_TIER,
    MACRO_SYMBOL_LIST,
    MATRIX_FEATURES,
    PREDICTION_HORIZON,
    SPECIALIZED_MACRO_FEATURES,
    SYMBOL_PREFIX_FILTER,
    TARGET_FEATURE,
)
from src.workflow.workflow_utils import (
    get_features_dir,
    get_historical_dir,
    get_metadata_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


def get_symbol_lists(weekly_dir: Path) -> tuple[List[str], List[str]]:
    """Get lists of target and macro symbols from weekly data.

    Applies SYMBOL_PREFIX_FILTER to targets only.
    Macro symbols are identified from MACRO_SYMBOL_LIST config.

    Args:
        weekly_dir: Directory containing weekly CSV files

    Returns:
        Tuple of (target_symbols, macro_symbols)
    """
    # Get all available symbols
    csv_files = sorted(weekly_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]
    all_symbols = [f.stem for f in csv_files]

    # Identify macro symbols
    macro_set: Set[str] = set(MACRO_SYMBOL_LIST)
    macro_symbols = [s for s in all_symbols if s in macro_set]

    # Identify target symbols (everything else)
    target_symbols = [s for s in all_symbols if s not in macro_set]

    # Apply prefix filter to targets only
    if SYMBOL_PREFIX_FILTER:
        target_symbols = [
            s for s in target_symbols if s.startswith(SYMBOL_PREFIX_FILTER)
        ]

    return sorted(target_symbols), sorted(macro_symbols)


@workflow_script("05-build-feature-matrix")
def main() -> None:
    """Main workflow function."""
    # Configuration
    metadata_dir = get_metadata_dir()
    weekly_dir = get_historical_dir(DATA_TIER) / "weekly"
    output_dir = get_features_dir(DATA_TIER)
    os.makedirs(output_dir, exist_ok=True)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter (targets only): {SYMBOL_PREFIX_FILTER or 'None'}")
    print(f"  Features per symbol: {len(MATRIX_FEATURES)}")
    print(f"  Specialized features: {len(SPECIALIZED_MACRO_FEATURES)}")
    print(f"  Prediction horizon: +{PREDICTION_HORIZON} week(s)")
    print(f"  Target feature: {TARGET_FEATURE}")
    print(f"  Input: {weekly_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Get symbol lists
    target_symbols, macro_symbols = get_symbol_lists(weekly_dir)

    print(f"Symbol breakdown:")
    print(f"  Target ETFs: {len(target_symbols)}")
    if target_symbols:
        print(
            f"    {', '.join(target_symbols[:10])}"
            f"{'...' if len(target_symbols) > 10 else ''}"
        )
    print(f"  Macro symbols: {len(macro_symbols)}")
    if macro_symbols:
        print(f"    {', '.join(macro_symbols)}")
    print()

    if not target_symbols:
        logger.error("No target symbols found. Please run previous workflow scripts.")
        return

    # Build feature matrix
    print("Building feature matrix...")
    print("-" * 80)

    builder = FeatureBuilder(
        weekly_data_dir=weekly_dir,
        target_symbols=target_symbols,
        macro_symbols=macro_symbols,
        features=MATRIX_FEATURES,
        specialized_features=SPECIALIZED_MACRO_FEATURES,
    )

    builder.load_weekly_data()
    feature_matrix = builder.build_feature_matrix(align_to_common_range=True)

    print()
    print("Feature matrix summary:")
    print(f"  Shape: {feature_matrix.shape}")
    print(
        f"  Date range: {feature_matrix.index.min().date()} to "
        f"{feature_matrix.index.max().date()}"
    )

    # Count feature types
    target_cols = [c for c in feature_matrix.columns if c.split("_")[0] in target_symbols]
    macro_cols = [c for c in feature_matrix.columns if c.split("_")[0] in macro_symbols]
    specialized_cols = list(SPECIALIZED_MACRO_FEATURES.keys())
    specialized_cols = [c for c in specialized_cols if c in feature_matrix.columns]

    print(f"  Target symbol features: {len(target_cols)}")
    print(f"  Macro symbol features: {len(macro_cols)}")
    print(f"  Specialized features: {len(specialized_cols)}")
    print(f"  Total features: {feature_matrix.shape[1]}")
    print(f"  Memory: {feature_matrix.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()

    # Build target matrix
    print("Building target matrix...")
    print("-" * 80)

    target_matrix = builder.build_target_matrix(
        target_feature=TARGET_FEATURE,
        prediction_horizon=PREDICTION_HORIZON,
    )

    print()
    print("Target matrix summary:")
    print(f"  Shape: {target_matrix.shape}")
    print(f"  Target symbols: {target_matrix.shape[1]}")
    print(
        f"  Non-null predictions possible: "
        f"{target_matrix.notna().all(axis=1).sum()} weeks"
    )
    print()

    # Save
    print("Saving matrices...")
    builder.save(output_dir)

    config: Dict[str, Any] = {
        "data_tier": DATA_TIER,
        "symbol_filter": SYMBOL_PREFIX_FILTER,
        "target_symbols": target_symbols,
        "macro_symbols": macro_symbols,
        "features": MATRIX_FEATURES,
        "specialized_features": list(SPECIALIZED_MACRO_FEATURES.keys()),
        "prediction_horizon": PREDICTION_HORIZON,
        "target_feature": TARGET_FEATURE,
        "feature_matrix_shape": list(feature_matrix.shape),
        "target_matrix_shape": list(target_matrix.shape),
        "date_range": {
            "start": str(feature_matrix.index.min().date()),
            "end": str(feature_matrix.index.max().date()),
        },
        "created_at": datetime.now().isoformat(),
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    print()

    # Preview
    print("Feature matrix preview (first 5 rows, first 10 columns):")
    print(feature_matrix.iloc[:5, :10].to_string())
    print()

    # Show specialized features
    if specialized_cols:
        print("Specialized features preview:")
        print(feature_matrix[specialized_cols].head().to_string())
        print()

    # Summary
    print_summary(
        target_symbols=len(target_symbols),
        macro_symbols=len(macro_symbols),
        weeks=len(feature_matrix),
        features_per_symbol=len(MATRIX_FEATURES),
        specialized_features=len(specialized_cols),
        total_features=feature_matrix.shape[1],
        output=str(output_dir),
    )


if __name__ == "__main__":
    main()

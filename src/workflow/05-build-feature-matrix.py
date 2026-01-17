#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build cross-sectional feature matrix for prediction experiments.

Creates aligned feature matrices from weekly ETF data where:
- Rows = weeks (aligned across all symbols)
- Columns = features from all symbols (cross-sectional)

This enables predicting symbol X using features from all symbols.

Input: data/historical/{tier}/weekly/*.csv
Output: data/features/{tier}/feature_matrix.parquet, target_matrix.parquet
"""

import json
import os
from datetime import datetime

from src.training.feature_builder import FeatureBuilder
from src.workflow.config import (
    DATA_TIER,
    MATRIX_FEATURES,
    PREDICTION_HORIZON,
    SYMBOL_PREFIX_FILTER,
    TARGET_FEATURE,
)
from src.workflow.workflow_utils import (
    get_features_dir,
    get_historical_dir,
    print_summary,
    setup_logging,
    workflow_script,
)

logger = setup_logging()


@workflow_script("05-build-feature-matrix")
def main() -> None:
    """Main workflow function."""
    # Configuration
    weekly_dir = get_historical_dir(DATA_TIER) / "weekly"
    output_dir = get_features_dir(DATA_TIER)
    os.makedirs(output_dir, exist_ok=True)

    print("Configuration:")
    print(f"  Data tier: {DATA_TIER}")
    print(f"  Symbol filter: {SYMBOL_PREFIX_FILTER or 'None (all symbols)'}")
    print(f"  Features per symbol: {len(MATRIX_FEATURES)}")
    print(f"  Prediction horizon: +{PREDICTION_HORIZON} week(s)")
    print(f"  Target feature: {TARGET_FEATURE}")
    print(f"  Input: {weekly_dir}")
    print(f"  Output: {output_dir}")
    print()

    # Get symbols
    csv_files = sorted(weekly_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("_")]

    if SYMBOL_PREFIX_FILTER:
        csv_files = [f for f in csv_files if f.stem.startswith(SYMBOL_PREFIX_FILTER)]

    symbols = [f.stem for f in csv_files]
    print(f"Found {len(symbols)} symbols to process")
    print(f"  Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
    print()

    if not symbols:
        logger.error("No symbols found. Please run previous workflow scripts first.")
        return

    # Build feature matrix
    print("Building feature matrix...")
    print("-" * 80)

    builder = FeatureBuilder(
        weekly_data_dir=weekly_dir,
        symbols=symbols,
        features=MATRIX_FEATURES,
    )

    builder.load_weekly_data()
    feature_matrix = builder.build_feature_matrix(align_to_common_range=True)

    print()
    print("Feature matrix summary:")
    print(f"  Shape: {feature_matrix.shape}")
    print(
        f"  Date range: {feature_matrix.index.min().date()} to {feature_matrix.index.max().date()}"
    )
    print(f"  Columns per symbol: {len(MATRIX_FEATURES)}")
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
    print(f"  Symbols: {target_matrix.shape[1]}")
    print(
        f"  Non-null predictions possible: {target_matrix.notna().all(axis=1).sum()} weeks"
    )
    print()

    # Save
    print("Saving matrices...")
    builder.save(output_dir)

    config = {
        "data_tier": DATA_TIER,
        "symbol_filter": SYMBOL_PREFIX_FILTER,
        "symbols": symbols,
        "features": MATRIX_FEATURES,
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

    # Summary
    print_summary(
        symbols=len(symbols),
        weeks=len(feature_matrix),
        features_per_symbol=len(MATRIX_FEATURES),
        total_features=feature_matrix.shape[1],
        output=str(output_dir),
    )


if __name__ == "__main__":
    main()

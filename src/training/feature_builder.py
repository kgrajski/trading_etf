# -*- coding: utf-8 -*-
"""
Feature matrix builder for cross-sectional prediction.

Constructs aligned feature matrices from weekly ETF data where:
- Rows = weeks (aligned across all symbols)
- Columns = features from all symbols (cross-sectional)

Supports:
- Target ETF features (prediction targets)
- Macro symbol features (predictors only)
- Specialized cross-symbol features (e.g., VIX term structure)

This enables predicting symbol X using features from all symbols.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features to extract from each symbol's weekly data
DEFAULT_FEATURES = [
    "log_return",
    "log_range",
    "log_volume",
    "momentum_4w",
    "momentum_12w",
    "intra_week_volatility",
    "log_volume_delta",
]


class FeatureBuilder:
    """Builds cross-sectional feature matrices from weekly ETF data.

    Supports both target ETFs (prediction targets) and macro symbols
    (feature-only, used as predictors but not predicted).
    """

    def __init__(
        self,
        weekly_data_dir: Path,
        target_symbols: Optional[List[str]] = None,
        macro_symbols: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        specialized_features: Optional[Dict[str, Tuple[str, str, str]]] = None,
    ):
        """Initialize feature builder.

        Args:
            weekly_data_dir: Directory containing weekly CSV files
            target_symbols: List of target symbols (prediction targets)
            macro_symbols: List of macro symbols (features only)
            features: List of features per symbol (None = defaults)
            specialized_features: Dict of cross-symbol features
                Format: {"name": (symbol1, symbol2, operation)}
                Operations: "ratio", "spread"
        """
        self.weekly_data_dir = Path(weekly_data_dir)
        self.features = features or DEFAULT_FEATURES
        self._target_symbols = target_symbols or []
        self._macro_symbols = macro_symbols or []
        self._specialized_features = specialized_features or {}
        self._weekly_data: Dict[str, pd.DataFrame] = {}
        self._feature_matrix: Optional[pd.DataFrame] = None
        self._target_matrix: Optional[pd.DataFrame] = None

    @property
    def symbols(self) -> List[str]:
        """Get list of symbols to process (deprecated - use all_symbols)."""
        return self.all_symbols

    @property
    def all_symbols(self) -> List[str]:
        """Get list of all symbols (targets + macro)."""
        return sorted(set(self._target_symbols + self._macro_symbols))

    @property
    def target_symbols(self) -> List[str]:
        """Get list of target symbols."""
        return self._target_symbols

    @property
    def macro_symbols(self) -> List[str]:
        """Get list of macro symbols."""
        return self._macro_symbols

    def load_weekly_data(self) -> Dict[str, pd.DataFrame]:
        """Load weekly data for all symbols.

        Returns:
            Dict mapping symbol to DataFrame
        """
        all_symbols = self.all_symbols
        logger.info(f"Loading weekly data for {len(all_symbols)} symbols...")
        logger.info(f"  Targets: {len(self._target_symbols)}")
        logger.info(f"  Macro: {len(self._macro_symbols)}")

        for symbol in all_symbols:
            csv_path = self.weekly_data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                logger.warning(f"Missing weekly data for {symbol}")
                continue

            df = pd.read_csv(csv_path)
            df["week_start"] = pd.to_datetime(df["week_start"])
            df = df.sort_values("week_start").reset_index(drop=True)
            self._weekly_data[symbol] = df

        logger.info(f"Loaded data for {len(self._weekly_data)} symbols")
        return self._weekly_data

    def get_common_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get date range common to all symbols.

        Returns:
            (min_date, max_date) tuple
        """
        if not self._weekly_data:
            self.load_weekly_data()

        min_dates = []
        max_dates = []

        for symbol, df in self._weekly_data.items():
            min_dates.append(df["week_start"].min())
            max_dates.append(df["week_start"].max())

        # Common range = latest start to earliest end
        common_start = max(min_dates)
        common_end = min(max_dates)

        logger.info(f"Common date range: {common_start.date()} to {common_end.date()}")
        return common_start, common_end

    def _compute_specialized_features(
        self, all_weeks: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """Compute specialized cross-symbol features.

        Args:
            all_weeks: List of week timestamps to compute for

        Returns:
            DataFrame with specialized features
        """
        if not self._specialized_features:
            return pd.DataFrame(index=all_weeks)

        specialized_dfs = {}

        for feature_name, (sym1, sym2, operation) in self._specialized_features.items():
            if sym1 not in self._weekly_data or sym2 not in self._weekly_data:
                logger.warning(
                    f"Cannot compute {feature_name}: missing {sym1} or {sym2}"
                )
                continue

            df1 = self._weekly_data[sym1].set_index("week_start")
            df2 = self._weekly_data[sym2].set_index("week_start")

            if operation == "ratio":
                # Log ratio of close prices
                values = np.log(df1["close"] / df2["close"])
            elif operation == "spread":
                # Difference in log returns
                values = df1["log_return"] - df2["log_return"]
            else:
                logger.warning(f"Unknown operation: {operation}")
                continue

            specialized_dfs[feature_name] = values

        if specialized_dfs:
            result = pd.DataFrame(specialized_dfs)
            result = result.loc[result.index.isin(all_weeks)].sort_index()
            return result

        return pd.DataFrame(index=all_weeks)

    def build_feature_matrix(
        self,
        align_to_common_range: bool = True,
    ) -> pd.DataFrame:
        """Build cross-sectional feature matrix.

        Creates a matrix where:
        - Index = week_start dates
        - Columns include:
          - {target_symbol}_{feature} for all target symbols
          - {macro_symbol}_{feature} for all macro symbols
          - Specialized cross-symbol features

        Args:
            align_to_common_range: If True, only include weeks present in all symbols

        Returns:
            DataFrame with shape (n_weeks, n_total_features)
        """
        if not self._weekly_data:
            self.load_weekly_data()

        logger.info("Building feature matrix...")

        # Get all unique weeks
        all_weeks = set()
        for df in self._weekly_data.values():
            all_weeks.update(df["week_start"].tolist())
        all_weeks = sorted(all_weeks)

        if align_to_common_range:
            common_start, common_end = self.get_common_date_range()
            all_weeks = [w for w in all_weeks if common_start <= w <= common_end]

        logger.info(f"Feature matrix will have {len(all_weeks)} weeks")

        # Build feature columns for each symbol
        feature_dfs = []

        # Process target symbols first
        for symbol in sorted(self._target_symbols):
            if symbol not in self._weekly_data:
                continue
            df = self._weekly_data[symbol].copy()
            df = df.set_index("week_start")

            symbol_features = {}
            for feature in self.features:
                if feature in df.columns:
                    col_name = f"{symbol}_{feature}"
                    symbol_features[col_name] = df[feature]

            if symbol_features:
                symbol_df = pd.DataFrame(symbol_features)
                feature_dfs.append(symbol_df)

        # Process macro symbols
        for symbol in sorted(self._macro_symbols):
            if symbol not in self._weekly_data:
                continue
            df = self._weekly_data[symbol].copy()
            df = df.set_index("week_start")

            symbol_features = {}
            for feature in self.features:
                if feature in df.columns:
                    col_name = f"{symbol}_{feature}"
                    symbol_features[col_name] = df[feature]

            if symbol_features:
                symbol_df = pd.DataFrame(symbol_features)
                feature_dfs.append(symbol_df)

        # Combine all symbol features
        if not feature_dfs:
            raise ValueError("No features extracted from any symbol")

        feature_matrix = pd.concat(feature_dfs, axis=1)

        # Add specialized features
        specialized_df = self._compute_specialized_features(all_weeks)
        if not specialized_df.empty:
            feature_matrix = pd.concat([feature_matrix, specialized_df], axis=1)
            logger.info(f"Added {len(specialized_df.columns)} specialized features")

        # Align to common weeks
        feature_matrix = feature_matrix.loc[
            feature_matrix.index.isin(all_weeks)
        ].sort_index()

        self._feature_matrix = feature_matrix

        n_target_features = len(self._target_symbols) * len(self.features)
        n_macro_features = len(self._macro_symbols) * len(self.features)
        n_specialized = len(specialized_df.columns) if not specialized_df.empty else 0

        logger.info(
            f"Feature matrix shape: {feature_matrix.shape} "
            f"({len(all_weeks)} weeks, {n_target_features} target features, "
            f"{n_macro_features} macro features, {n_specialized} specialized)"
        )

        return feature_matrix

    def build_target_matrix(
        self,
        target_feature: str = "log_return",
        prediction_horizon: int = 1,
    ) -> pd.DataFrame:
        """Build target matrix for prediction.

        Creates a matrix of future returns for TARGET symbols only, shifted by
        prediction_horizon. Macro symbols are NOT included as targets.

        Args:
            target_feature: Feature to predict (default: log_return)
            prediction_horizon: Number of weeks ahead to predict

        Returns:
            DataFrame with shape (n_weeks, n_target_symbols)
        """
        if self._feature_matrix is None:
            self.build_feature_matrix()

        logger.info(
            f"Building target matrix: {target_feature}, horizon={prediction_horizon}"
        )
        logger.info(f"Target symbols only: {len(self._target_symbols)}")

        target_dfs = {}

        # Only include target symbols in target matrix (not macro symbols)
        for symbol in sorted(self._target_symbols):
            if symbol not in self._weekly_data:
                continue

            df = self._weekly_data[symbol].copy()
            df = df.set_index("week_start")

            if target_feature in df.columns:
                # Shift to get future value
                # Negative shift means we're looking ahead
                future_values = df[target_feature].shift(-prediction_horizon)
                target_dfs[symbol] = future_values

        target_matrix = pd.DataFrame(target_dfs)

        # Align to feature matrix index
        target_matrix = target_matrix.loc[
            target_matrix.index.isin(self._feature_matrix.index)
        ].sort_index()

        self._target_matrix = target_matrix

        logger.info(f"Target matrix shape: {target_matrix.shape}")

        return target_matrix

    def get_train_test_split(
        self,
        train_end_week: pd.Timestamp,
        target_symbol: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test split for a specific training cutoff.

        Args:
            train_end_week: Last week to include in training
            target_symbol: Symbol to predict

        Returns:
            (X_train, y_train, X_test, y_test) arrays
        """
        if self._feature_matrix is None or self._target_matrix is None:
            raise ValueError("Must build feature and target matrices first")

        # Training data: all weeks up to and including train_end_week
        train_mask = self._feature_matrix.index <= train_end_week
        test_mask = self._feature_matrix.index > train_end_week

        X_train = self._feature_matrix.loc[train_mask].values
        X_test = self._feature_matrix.loc[test_mask].values

        y_train = self._target_matrix.loc[train_mask, target_symbol].values
        y_test = self._target_matrix.loc[test_mask, target_symbol].values

        return X_train, y_train, X_test, y_test

    def save(self, output_dir: Path) -> None:
        """Save feature and target matrices to parquet.

        Args:
            output_dir: Directory to save matrices
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self._feature_matrix is not None:
            feature_path = output_dir / "feature_matrix.parquet"
            self._feature_matrix.to_parquet(feature_path)
            logger.info(f"Saved feature matrix to {feature_path}")

        if self._target_matrix is not None:
            target_path = output_dir / "target_matrix.parquet"
            self._target_matrix.to_parquet(target_path)
            logger.info(f"Saved target matrix to {target_path}")

        # Save metadata
        metadata: Dict[str, Any] = {
            "target_symbols": self._target_symbols,
            "macro_symbols": self._macro_symbols,
            "features": self.features,
            "specialized_features": list(self._specialized_features.keys()),
            "n_weeks": (
                len(self._feature_matrix) if self._feature_matrix is not None else 0
            ),
            "date_range": {
                "start": (
                    str(self._feature_matrix.index.min())
                    if self._feature_matrix is not None
                    else None
                ),
                "end": (
                    str(self._feature_matrix.index.max())
                    if self._feature_matrix is not None
                    else None
                ),
            },
            "feature_matrix_shape": (
                list(self._feature_matrix.shape)
                if self._feature_matrix is not None
                else None
            ),
            "target_matrix_shape": (
                list(self._target_matrix.shape)
                if self._target_matrix is not None
                else None
            ),
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, features_dir: Path) -> "FeatureBuilder":
        """Load saved feature matrices.

        Args:
            features_dir: Directory containing saved matrices

        Returns:
            FeatureBuilder instance with loaded data
        """
        features_dir = Path(features_dir)

        # Load metadata
        metadata_path = features_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            target_symbols = metadata.get("target_symbols", [])
            macro_symbols = metadata.get("macro_symbols", [])
            features = metadata.get("features", DEFAULT_FEATURES)
        else:
            target_symbols = []
            macro_symbols = []
            features = DEFAULT_FEATURES

        builder = cls(
            weekly_data_dir=features_dir,  # Dummy path
            target_symbols=target_symbols,
            macro_symbols=macro_symbols,
            features=features,
        )

        feature_path = features_dir / "feature_matrix.parquet"
        if feature_path.exists():
            builder._feature_matrix = pd.read_parquet(feature_path)

        target_path = features_dir / "target_matrix.parquet"
        if target_path.exists():
            builder._target_matrix = pd.read_parquet(target_path)

        return builder

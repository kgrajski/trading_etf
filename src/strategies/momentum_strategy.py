# -*- coding: utf-8 -*-
"""Momentum-based trading strategy for ETFs.

This strategy enters positions based on price and volume momentum,
and exits based on momentum reversal or profit targets.

Author: kag
Created: 2025-12-01
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy.

    Entry criteria:
    - Positive price momentum (N-week return > threshold)
    - Volume above average
    - Price above moving average

    Exit criteria:
    - Momentum reversal (negative momentum)
    - Stop-loss hit
    - Profit target reached
    """

    def __init__(self):
        """Initialize momentum strategy."""
        super().__init__("momentum")

    def generate_signals(
        self,
        weekly_data: Dict[str, pd.DataFrame],
        week_start: str,
        current_positions: List[str],
        params: Dict,
    ) -> List[Signal]:
        """Generate trading signals based on momentum.

        Args:
            weekly_data: Dictionary mapping symbol to DataFrame with features
            week_start: Week start date
            current_positions: Currently held symbols
            params: Strategy parameters:
                - lookback_weeks: Number of weeks for momentum calculation
                - momentum_threshold: Minimum momentum % to enter
                - volume_threshold: Minimum volume ratio vs average
                - max_positions: Maximum number of positions
                - stop_loss_pct: Stop-loss percentage (negative)
                - profit_target_pct: Profit target percentage (positive)

        Returns:
            List of Signal objects
        """
        signals = []

        lookback = params.get("lookback_weeks", 4)
        momentum_threshold = params.get("momentum_threshold", 0.02)  # 2%
        volume_threshold = params.get("volume_threshold", 1.2)
        max_positions = params.get("max_positions", 5)
        stop_loss = params.get("stop_loss_pct", -0.05)  # -5%
        profit_target = params.get("profit_target_pct", 0.10)  # 10%

        # Get current week's data for all symbols
        current_week_data = {}
        for symbol, df in weekly_data.items():
            week_row = df[df["week_start"] == week_start]
            if not week_row.empty:
                current_week_data[symbol] = week_row.iloc[0]

        # Check exit signals for current positions
        for symbol in current_positions:
            if symbol not in current_week_data:
                continue

            row = current_week_data[symbol]
            df = weekly_data[symbol]

            # Find entry week for this position (simplified - would track in backtester)
            # For now, check if momentum has reversed
            momentum_col = f"return_{lookback}w"
            if momentum_col in row and pd.notna(row[momentum_col]):
                if row[momentum_col] < 0:  # Momentum reversed
                    signals.append(
                        Signal(
                            symbol=symbol,
                            action=Signal.EXIT,
                            week_start=week_start,
                            confidence=0.8,
                            metadata={"reason": "momentum_reversal"},
                        )
                    )
                    continue

            # Check stop-loss and profit target (would need entry price from backtester)
            # This is simplified - actual implementation would track entry prices

        # Generate entry signals
        candidates = []
        for symbol, row in current_week_data.items():
            if symbol in current_positions:
                continue  # Already holding

            df = weekly_data[symbol]

            # Check entry criteria
            momentum_col = f"return_{lookback}w"
            if momentum_col not in row or pd.isna(row[momentum_col]):
                continue

            momentum = row[momentum_col] / 100  # Convert from percentage

            # Momentum check
            if momentum < momentum_threshold:
                continue

            # Volume check
            volume_ratio = row.get("volume_ratio", 0)
            if volume_ratio < volume_threshold:
                continue

            # Price vs MA check (optional)
            price_vs_ma = row.get("price_vs_ma4", 0)
            if price_vs_ma < 0:  # Price below MA
                continue

            # Calculate score (momentum * volume strength)
            score = momentum * volume_ratio

            candidates.append(
                {
                    "symbol": symbol,
                    "score": score,
                    "momentum": momentum,
                    "volume_ratio": volume_ratio,
                }
            )

        # Sort by score and select top candidates
        candidates.sort(key=lambda x: x["score"], reverse=True)
        num_to_enter = max_positions - len(current_positions)

        for candidate in candidates[:num_to_enter]:
            signals.append(
                Signal(
                    symbol=candidate["symbol"],
                    action=Signal.ENTER,
                    week_start=week_start,
                    confidence=min(candidate["score"] * 10, 1.0),  # Normalize to 0-1
                    metadata={
                        "momentum": candidate["momentum"],
                        "volume_ratio": candidate["volume_ratio"],
                        "score": candidate["score"],
                    },
                )
            )

        return signals

    def get_required_features(self) -> List[str]:
        """Return list of required feature columns.

        Returns:
            List of feature column names
        """
        return [
            "week_start",
            "close",
            "volume",
            "return_2w",
            "return_4w",
            "return_8w",
            "volume_ratio",
            "price_vs_ma4",
            "ma4",
            "ma8",
        ]

# -*- coding: utf-8 -*-
"""Base strategy class for ETF trading strategies.

This module defines the abstract base class that all trading strategies
must implement.

Author: kag
Created: 2025-12-01
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class Signal:
    """Represents a trading signal for a symbol."""

    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"

    def __init__(
        self,
        symbol: str,
        action: str,
        week_start: str,
        confidence: float = 1.0,
        metadata: Optional[Dict] = None,
    ):
        """Initialize signal.

        Args:
            symbol: ETF symbol
            action: 'enter', 'exit', or 'hold'
            week_start: Week start date (Monday)
            confidence: Confidence score (0-1)
            metadata: Optional metadata about the signal
        """
        self.symbol = symbol
        self.action = action
        self.week_start = week_start
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert signal to dictionary."""
        return {
            "symbol": self.symbol,
            "action": self.action,
            "week_start": self.week_start,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must inherit from this class and implement the
    generate_signals method.
    """

    def __init__(self, name: str):
        """Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name

    @abstractmethod
    def generate_signals(
        self,
        weekly_data: Dict[str, pd.DataFrame],
        week_start: str,
        current_positions: List[str],
        params: Dict,
    ) -> List[Signal]:
        """Generate trading signals for a given week.

        Args:
            weekly_data: Dictionary mapping symbol to DataFrame with weekly features
            week_start: Week start date (Monday) in YYYY-MM-DD format
            current_positions: List of symbols currently held
            params: Strategy parameters (hyperparameters)

        Returns:
            List of Signal objects
        """
        pass

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """Return list of required feature columns.

        Returns:
            List of feature column names
        """
        pass

    def get_name(self) -> str:
        """Get strategy name.

        Returns:
            Strategy name string
        """
        return self.name

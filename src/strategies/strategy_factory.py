# -*- coding: utf-8 -*-
"""Factory for creating strategy instances.

This module provides a factory pattern for instantiating strategy objects.

Author: kag
Created: 2025-12-01
"""

from src.strategies.base_strategy import BaseStrategy
from src.strategies.momentum_strategy import MomentumStrategy


class StrategyFactory:
    """Factory for creating strategy instances."""

    @staticmethod
    def create(strategy_name: str) -> BaseStrategy:
        """Create a strategy instance.

        Args:
            strategy_name: Name of the strategy ('momentum', etc.)

        Returns:
            BaseStrategy instance

        Raises:
            ValueError: If strategy_name is not recognized
        """
        strategy_name = strategy_name.lower()

        if strategy_name == "momentum":
            return MomentumStrategy()
        else:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. " f"Available strategies: momentum"
            )

    @staticmethod
    def list_available_strategies() -> list:
        """Return list of available strategies.

        Returns:
            List of strategy name strings
        """
        return ["momentum"]

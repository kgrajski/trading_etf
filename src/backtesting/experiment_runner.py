# -*- coding: utf-8 -*-
"""Experiment runner for ETF trading strategies.

This module orchestrates running experiments with hyperparameter search
and walk-forward optimization.

Author: kag
Created: 2025-12-01
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ray

from src.backtesting.portfolio_backtester import PortfolioBacktester
from src.strategies.strategy_factory import StrategyFactory


@ray.remote
def run_single_backtest(config: dict) -> dict:
    """Run a single backtest configuration (Ray remote function).

    Args:
        config: Configuration dictionary with:
            - weekly_data_ref: Ray reference to weekly data
            - strategy_name: Strategy name
            - params: Strategy parameters
            - weeks: List of week start dates
            - initial_capital: Starting capital

    Returns:
        Dictionary with results
    """
    # Get data from object store
    weekly_data = ray.get(config["weekly_data_ref"])

    # Create strategy
    strategy = StrategyFactory.create(config["strategy_name"])

    # Run backtest
    backtester = PortfolioBacktester(initial_capital=config["initial_capital"])
    results_df = backtester.run_backtest(
        weekly_data=weekly_data,
        strategy=strategy,
        params=config["params"],
        weeks=config["weeks"],
    )

    # Calculate metrics
    metrics = backtester.calculate_metrics()

    # Add parameter info
    metrics.update({"strategy": config["strategy_name"], "params": config["params"]})

    return metrics


class ExperimentRunner:
    """Runs experiments with hyperparameter search."""

    def __init__(self, experiment_dir: str):
        """Initialize experiment runner.

        Args:
            experiment_dir: Directory for experiment results
        """
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)

    def run_hyperparameter_search(
        self,
        weekly_data: Dict[str, pd.DataFrame],
        strategy_name: str,
        param_grid: Dict,
        weeks: List[str],
        initial_capital: float = 100000.0,
    ) -> pd.DataFrame:
        """Run hyperparameter search using Ray.

        Args:
            weekly_data: Dictionary mapping symbol to DataFrame with features
            strategy_name: Strategy name
            param_grid: Dictionary of parameter lists to search
            weeks: List of week start dates
            initial_capital: Starting capital

        Returns:
            DataFrame with results for each parameter combination
        """
        # Put data in Ray object store
        weekly_data_ref = ray.put(weekly_data)

        # Generate all parameter combinations
        configs = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        from itertools import product

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            configs.append(
                {
                    "weekly_data_ref": weekly_data_ref,
                    "strategy_name": strategy_name,
                    "params": params,
                    "weeks": weeks,
                    "initial_capital": initial_capital,
                }
            )

        # Run in parallel
        futures = [run_single_backtest.remote(c) for c in configs]
        results = ray.get(futures)

        # Convert to DataFrame
        return pd.DataFrame(results)

    def run_walk_forward(
        self,
        weekly_data: Dict[str, pd.DataFrame],
        strategy_name: str,
        param_grid: Dict,
        all_weeks: List[str],
        train_weeks: int = 52,
        operate_weeks: int = 1,
        initial_capital: float = 100000.0,
    ) -> Dict:
        """Run walk-forward optimization.

        Args:
            weekly_data: Dictionary mapping symbol to DataFrame with features
            strategy_name: Strategy name
            param_grid: Parameter grid for search
            all_weeks: All available week start dates
            train_weeks: Number of weeks for training window
            operate_weeks: Number of weeks to operate before retraining
            initial_capital: Starting capital

        Returns:
            Dictionary with walk-forward results
        """
        # Partition weeks into train/operate windows
        num_windows = (len(all_weeks) - train_weeks) // operate_weeks

        window_results = []
        cumulative_trades = []

        for window_idx in range(num_windows):
            train_start_idx = window_idx * operate_weeks
            train_end_idx = train_start_idx + train_weeks
            operate_start_idx = train_end_idx
            operate_end_idx = operate_start_idx + operate_weeks

            if operate_end_idx > len(all_weeks):
                break

            train_weeks_list = all_weeks[train_start_idx:train_end_idx]
            operate_weeks_list = all_weeks[operate_start_idx:operate_end_idx]

            # Run hyperparameter search on training data
            train_results = self.run_hyperparameter_search(
                weekly_data=weekly_data,
                strategy_name=strategy_name,
                param_grid=param_grid,
                weeks=train_weeks_list,
                initial_capital=initial_capital,
            )

            if train_results.empty:
                continue

            # Select best parameters (by Sharpe ratio)
            best_params = train_results.nlargest(1, "sharpe_ratio").iloc[0]["params"]

            # Run on operate period
            strategy = StrategyFactory.create(strategy_name)
            backtester = PortfolioBacktester(initial_capital=initial_capital)
            operate_results = backtester.run_backtest(
                weekly_data=weekly_data,
                strategy=strategy,
                params=best_params,
                weeks=operate_weeks_list,
            )

            operate_metrics = backtester.calculate_metrics()
            closed_positions = backtester.get_closed_positions()

            window_results.append(
                {
                    "window_idx": window_idx,
                    "train_start": train_weeks_list[0],
                    "train_end": train_weeks_list[-1],
                    "operate_start": operate_weeks_list[0],
                    "operate_end": operate_weeks_list[-1],
                    "best_params": best_params,
                    "operate_metrics": operate_metrics,
                }
            )

            if not closed_positions.empty:
                cumulative_trades.append(closed_positions)

        # Aggregate results
        if cumulative_trades:
            all_trades = pd.concat(cumulative_trades, ignore_index=True)
        else:
            all_trades = pd.DataFrame()

        return {
            "window_results": window_results,
            "all_trades": all_trades,
            "config": {
                "strategy": strategy_name,
                "train_weeks": train_weeks,
                "operate_weeks": operate_weeks,
                "num_windows": len(window_results),
            },
        }

    def save_results(self, results: Dict, filename: str = "results.json"):
        """Save experiment results to file.

        Args:
            results: Results dictionary
            filename: Output filename
        """
        output_path = os.path.join(self.experiment_dir, filename)

        # Convert DataFrames to dict for JSON serialization
        results_copy = results.copy()
        if "all_trades" in results_copy and isinstance(
            results_copy["all_trades"], pd.DataFrame
        ):
            results_copy["all_trades"] = results_copy["all_trades"].to_dict("records")

        with open(output_path, "w") as f:
            json.dump(results_copy, f, indent=2, default=str)

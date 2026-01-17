# -*- coding: utf-8 -*-
"""Portfolio visualization for ETF trading results.

This module provides visualization functions for portfolio-level backtesting
results, including equity curves and performance metrics.

Author: kag
Created: 2025-12-01
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class PortfolioVisualizer:
    """Visualizer for portfolio backtesting results."""

    def __init__(self):
        """Initialize visualizer."""
        pass

    def create_equity_curve_figure(
        self, weekly_records: pd.DataFrame, initial_capital: float = 100000.0
    ) -> go.Figure:
        """Create equity curve showing portfolio value over time.

        Args:
            weekly_records: DataFrame with weekly portfolio metrics
            initial_capital: Starting capital

        Returns:
            Plotly figure object
        """
        fig = go.Figure()

        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(weekly_records["week_start"]),
                y=weekly_records["total_value"],
                mode="lines+markers",
                name="Portfolio Value",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
                customdata=weekly_records[
                    ["total_return_pct", "drawdown_pct", "num_positions"]
                ].values,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Value: $%{y:,.2f}<br>"
                    "Return: %{customdata[0]:.2f}%<br>"
                    "Drawdown: %{customdata[1]:.2f}%<br>"
                    "Positions: %{customdata[2]:.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Add starting capital line
        fig.add_hline(
            y=initial_capital,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            annotation_text=f"Starting Capital: ${initial_capital:,.0f}",
            annotation_position="left",
        )

        # Calculate final metrics
        final_value = weekly_records["total_value"].iloc[-1]
        total_return = weekly_records["total_return_pct"].iloc[-1]
        max_dd = weekly_records["drawdown_pct"].max()

        # Update layout
        fig.update_layout(
            title=f"Portfolio Equity Curve - Return: {total_return:.2f}% | Max DD: {max_dd:.2f}%",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        return fig

    def create_performance_summary_figure(
        self, metrics: Dict, trades_df: pd.DataFrame
    ) -> go.Figure:
        """Create summary figure with key performance metrics.

        Args:
            metrics: Dictionary with performance metrics
            trades_df: DataFrame with closed positions

        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Return Distribution",
                "Win/Loss Breakdown",
                "Returns Over Time",
                "Position Sizes",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Return distribution
        if not trades_df.empty and "return_pct" in trades_df.columns:
            fig.add_trace(
                go.Histogram(
                    x=trades_df["return_pct"], nbinsx=30, name="Return Distribution"
                ),
                row=1,
                col=1,
            )

        # Win/Loss breakdown
        if not trades_df.empty and "return_pct" in trades_df.columns:
            winners = len(trades_df[trades_df["return_pct"] > 0])
            losers = len(trades_df[trades_df["return_pct"] <= 0])

            fig.add_trace(
                go.Bar(
                    x=["Winners", "Losers"],
                    y=[winners, losers],
                    name="Win/Loss",
                    marker_color=["green", "red"],
                ),
                row=1,
                col=2,
            )

        # Returns over time
        if not trades_df.empty and "entry_week" in trades_df.columns:
            trades_sorted = trades_df.sort_values("entry_week")
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(trades_sorted["entry_week"]),
                    y=trades_sorted["return_pct"],
                    mode="markers",
                    name="Trade Returns",
                    marker=dict(
                        color=trades_sorted["return_pct"],
                        colorscale="RdYlGn",
                        size=8,
                        showscale=True,
                    ),
                ),
                row=2,
                col=1,
            )

        # Position sizes (by symbol)
        if not trades_df.empty and "symbol" in trades_df.columns:
            symbol_counts = trades_df["symbol"].value_counts()
            fig.add_trace(
                go.Bar(
                    x=symbol_counts.index,
                    y=symbol_counts.values,
                    name="Trades by Symbol",
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Performance Summary",
            template="plotly_white",
            height=800,
            showlegend=False,
        )

        return fig

    def save_figure(self, fig: go.Figure, filepath: str):
        """Save figure to HTML file.

        Args:
            fig: Plotly figure
            filepath: Output file path
        """
        fig.write_html(filepath)

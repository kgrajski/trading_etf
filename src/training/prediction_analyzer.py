"""Post-prediction analysis for experiment results."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def log_to_pct(log_return: float) -> float:
    """Convert log return to percentage."""
    return (np.exp(log_return) - 1) * 100


class PredictionAnalyzer:
    """Analyze predictions from experiment results.
    
    Generates:
    1. Per-symbol performance rankings
    2. Directional accuracy analysis with return distributions
    """
    
    def __init__(self, predictions_df: pd.DataFrame, model_col: str = "pred_ridge_log"):
        """Initialize analyzer.
        
        Args:
            predictions_df: DataFrame with columns: symbol, week_start, actual_log, pred_*_log
            model_col: Which model's predictions to analyze
        """
        self.df = predictions_df.copy()
        self.model_col = model_col
        self.model_name = model_col.replace("pred_", "").replace("_log", "").title()
        
        # Add derived columns
        self.df["actual_pct"] = self.df["actual_log"].apply(log_to_pct)
        self.df["pred_pct"] = self.df[model_col].apply(log_to_pct)
        self.df["error_log"] = self.df[model_col] - self.df["actual_log"]
        self.df["error_pct"] = self.df["pred_pct"] - self.df["actual_pct"]
        self.df["abs_error_pct"] = self.df["error_pct"].abs()
        
        # Directional analysis
        self.df["actual_dir"] = np.sign(self.df["actual_log"])  # +1, 0, -1
        self.df["pred_dir"] = np.sign(self.df[model_col])
        self.df["dir_correct"] = self.df["actual_dir"] == self.df["pred_dir"]
        
        # Categorize into 4 quadrants
        self.df["quadrant"] = self.df.apply(self._classify_quadrant, axis=1)
    
    def _classify_quadrant(self, row) -> str:
        """Classify prediction into one of 4 quadrants."""
        actual_up = row["actual_log"] > 0
        pred_up = row[self.model_col] > 0
        
        if actual_up and pred_up:
            return "correct_up"      # Predicted up, was up ✓
        elif not actual_up and not pred_up:
            return "correct_down"    # Predicted down, was down ✓
        elif pred_up and not actual_up:
            return "wrong_up"        # Predicted up, was down ✗
        else:
            return "wrong_down"      # Predicted down, was up ✗
    
    def per_symbol_performance(self, top_n: int = 20) -> pd.DataFrame:
        """Compute per-symbol prediction performance.
        
        Returns DataFrame with columns:
        - symbol, n_weeks, mae_pct, rmse_pct, dir_accuracy, mean_actual_pct
        """
        grouped = self.df.groupby("symbol").agg({
            "actual_pct": ["count", "mean", "std"],
            "error_pct": lambda x: np.mean(np.abs(x)),  # MAE
            "dir_correct": "mean",
            self.model_col: "mean",
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            "symbol", "n_weeks", "mean_actual_pct", "std_actual_pct",
            "mae_pct", "dir_accuracy", "mean_pred_log"
        ]
        
        # Compute RMSE
        rmse_by_symbol = self.df.groupby("symbol")["error_pct"].apply(
            lambda x: np.sqrt(np.mean(x**2))
        ).reset_index()
        rmse_by_symbol.columns = ["symbol", "rmse_pct"]
        
        grouped = grouped.merge(rmse_by_symbol, on="symbol")
        
        # Sort by directional accuracy (descending)
        grouped = grouped.sort_values("dir_accuracy", ascending=False)
        
        return grouped
    
    def directional_analysis(self) -> Dict[str, pd.DataFrame]:
        """Analyze return distributions by directional correctness.
        
        Returns dict with:
        - summary: Overall stats per quadrant
        - distributions: Raw data for each quadrant
        """
        quadrants = ["correct_up", "correct_down", "wrong_up", "wrong_down"]
        
        summary_rows = []
        distributions = {}
        
        for quad in quadrants:
            subset = self.df[self.df["quadrant"] == quad]
            distributions[quad] = subset["actual_pct"].values
            
            if len(subset) > 0:
                summary_rows.append({
                    "quadrant": quad,
                    "count": len(subset),
                    "pct_of_total": len(subset) / len(self.df) * 100,
                    "mean_actual_pct": subset["actual_pct"].mean(),
                    "median_actual_pct": subset["actual_pct"].median(),
                    "std_actual_pct": subset["actual_pct"].std(),
                    "min_actual_pct": subset["actual_pct"].min(),
                    "max_actual_pct": subset["actual_pct"].max(),
                })
        
        summary = pd.DataFrame(summary_rows)
        
        return {
            "summary": summary,
            "distributions": distributions,
        }
    
    def plot_directional_distributions(self) -> go.Figure:
        """Create histogram of returns by directional quadrant."""
        analysis = self.directional_analysis()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Correct ↑ (Pred Up, Was Up)",
                "Correct ↓ (Pred Down, Was Down)",
                "Wrong ↑ (Pred Up, Was Down)",
                "Wrong ↓ (Pred Down, Was Up)",
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )
        
        quadrant_info = [
            ("correct_up", 1, 1, "#2ECC71"),      # Green
            ("correct_down", 1, 2, "#27AE60"),    # Darker green
            ("wrong_up", 2, 1, "#E74C3C"),        # Red
            ("wrong_down", 2, 2, "#C0392B"),      # Darker red
        ]
        
        for quad, row, col, color in quadrant_info:
            data = analysis["distributions"][quad]
            if len(data) > 0:
                summary = analysis["summary"]
                quad_summary = summary[summary["quadrant"] == quad].iloc[0]
                
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        nbinsx=30,
                        marker_color=color,
                        opacity=0.75,
                        name=quad,
                        showlegend=False,
                        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
                    ),
                    row=row, col=col,
                )
                
                # Add mean line
                mean_val = quad_summary["mean_actual_pct"]
                fig.add_vline(
                    x=mean_val, row=row, col=col,
                    line_dash="dash", line_color="white",
                    annotation_text=f"μ={mean_val:.2f}%",
                    annotation_position="top right",
                )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{self.model_name}</b> - Return Distributions by Directional Correctness",
                x=0.5,
                xanchor="center",
            ),
            height=600,
            template="plotly_dark",
        )
        
        fig.update_xaxes(title_text="Actual Return (%)")
        fig.update_yaxes(title_text="Count")
        
        return fig
    
    def plot_per_symbol_rankings(self, top_n: int = 20) -> go.Figure:
        """Create bar chart of top/bottom symbols by directional accuracy."""
        rankings = self.per_symbol_performance()
        
        # Top and bottom N
        top = rankings.head(top_n)
        bottom = rankings.tail(top_n).iloc[::-1]  # Reverse for display
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"Top {top_n} by Directional Accuracy",
                f"Bottom {top_n} by Directional Accuracy",
            ],
            horizontal_spacing=0.15,
        )
        
        # Top symbols
        fig.add_trace(
            go.Bar(
                x=top["dir_accuracy"] * 100,
                y=top["symbol"],
                orientation="h",
                marker_color="#2ECC71",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Dir. Acc: %{x:.1f}%<br>"
                    "MAE: %{customdata[0]:.2f}%<br>"
                    "n=%{customdata[1]}<extra></extra>"
                ),
                customdata=top[["mae_pct", "n_weeks"]].values,
            ),
            row=1, col=1,
        )
        
        # Bottom symbols
        fig.add_trace(
            go.Bar(
                x=bottom["dir_accuracy"] * 100,
                y=bottom["symbol"],
                orientation="h",
                marker_color="#E74C3C",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Dir. Acc: %{x:.1f}%<br>"
                    "MAE: %{customdata[0]:.2f}%<br>"
                    "n=%{customdata[1]}<extra></extra>"
                ),
                customdata=bottom[["mae_pct", "n_weeks"]].values,
            ),
            row=1, col=2,
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{self.model_name}</b> - Per-Symbol Directional Accuracy",
                x=0.5,
                xanchor="center",
            ),
            height=max(400, 25 * top_n),
            template="plotly_dark",
            showlegend=False,
        )
        
        fig.update_xaxes(title_text="Directional Accuracy (%)", range=[0, 100])
        
        return fig
    
    def generate_report(self) -> str:
        """Generate markdown report of prediction analysis."""
        lines = []
        
        lines.append(f"# Prediction Analysis: {self.model_name}")
        lines.append("")
        lines.append(f"**Total predictions:** {len(self.df)}")
        lines.append(f"**Unique symbols:** {self.df['symbol'].nunique()}")
        lines.append("")
        
        # Overall directional accuracy
        overall_dir_acc = self.df["dir_correct"].mean()
        lines.append(f"**Overall Directional Accuracy:** {overall_dir_acc:.1%}")
        lines.append("")
        
        # Directional breakdown
        lines.append("## Directional Analysis")
        lines.append("")
        
        analysis = self.directional_analysis()
        summary = analysis["summary"]
        
        lines.append("| Quadrant | Count | % of Total | Mean Return | Median | Std |")
        lines.append("|----------|-------|------------|-------------|--------|-----|")
        
        for _, row in summary.iterrows():
            quad_label = row["quadrant"].replace("_", " ").title()
            lines.append(
                f"| {quad_label} | {row['count']:,} | {row['pct_of_total']:.1f}% | "
                f"{row['mean_actual_pct']:+.2f}% | {row['median_actual_pct']:+.2f}% | "
                f"{row['std_actual_pct']:.2f}% |"
            )
        
        lines.append("")
        
        # Betting insight
        correct_up = summary[summary["quadrant"] == "correct_up"]
        correct_down = summary[summary["quadrant"] == "correct_down"]
        wrong_up = summary[summary["quadrant"] == "wrong_up"]
        wrong_down = summary[summary["quadrant"] == "wrong_down"]
        
        if len(correct_up) > 0 and len(wrong_up) > 0:
            # If we bet on predicted UP:
            up_bets = self.df[self.df["pred_dir"] == 1]
            if len(up_bets) > 0:
                up_mean_return = up_bets["actual_pct"].mean()
                up_win_rate = (up_bets["actual_dir"] == 1).mean()
                lines.append("### If We Bet on Predicted UP:")
                lines.append(f"- **Bets placed:** {len(up_bets)}")
                lines.append(f"- **Win rate:** {up_win_rate:.1%}")
                lines.append(f"- **Mean actual return:** {up_mean_return:+.2f}%")
                lines.append("")
        
        if len(correct_down) > 0 and len(wrong_down) > 0:
            # If we bet on predicted DOWN (short):
            down_bets = self.df[self.df["pred_dir"] == -1]
            if len(down_bets) > 0:
                down_mean_return = -down_bets["actual_pct"].mean()  # Profit from short
                down_win_rate = (down_bets["actual_dir"] == -1).mean()
                lines.append("### If We Bet on Predicted DOWN (Short):")
                lines.append(f"- **Bets placed:** {len(down_bets)}")
                lines.append(f"- **Win rate:** {down_win_rate:.1%}")
                lines.append(f"- **Mean actual return (if shorted):** {down_mean_return:+.2f}%")
                lines.append("")
        
        # Per-symbol rankings
        lines.append("## Top 20 Symbols by Directional Accuracy")
        lines.append("")
        
        rankings = self.per_symbol_performance(top_n=20)
        top20 = rankings.head(20)
        
        lines.append("| Rank | Symbol | Dir. Acc | MAE (%) | Mean Actual (%) | n |")
        lines.append("|------|--------|----------|---------|-----------------|---|")
        
        for i, (_, row) in enumerate(top20.iterrows(), 1):
            lines.append(
                f"| {i} | {row['symbol']} | {row['dir_accuracy']:.1%} | "
                f"{row['mae_pct']:.2f} | {row['mean_actual_pct']:+.2f} | {row['n_weeks']} |"
            )
        
        lines.append("")
        
        # Bottom 20
        lines.append("## Bottom 20 Symbols by Directional Accuracy")
        lines.append("")
        
        bottom20 = rankings.tail(20).iloc[::-1]
        
        lines.append("| Rank | Symbol | Dir. Acc | MAE (%) | Mean Actual (%) | n |")
        lines.append("|------|--------|----------|---------|-----------------|---|")
        
        for i, (_, row) in enumerate(bottom20.iterrows(), 1):
            lines.append(
                f"| {i} | {row['symbol']} | {row['dir_accuracy']:.1%} | "
                f"{row['mae_pct']:.2f} | {row['mean_actual_pct']:+.2f} | {row['n_weeks']} |"
            )
        
        return "\n".join(lines)

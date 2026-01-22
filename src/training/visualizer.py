"""Visualization generation for experiment results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def log_to_pct(log_return: float) -> float:
    """Convert log return to percentage return.
    
    Formula: pct = (exp(log_return) - 1) * 100
    """
    return (np.exp(log_return) - 1) * 100


class ExperimentVisualizer:
    """Generate interactive Plotly visualizations for experiment results."""
    
    def __init__(self, output_dir: Path):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._plot_paths: List[Path] = []
    
    def save_plot(self, fig: go.Figure, filename: str) -> Path:
        """Save a Plotly figure as uncompressed HTML for direct browser viewing.
        
        Args:
            fig: Plotly figure
            filename: Output filename (without .html extension)
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.html"
        
        html_content = fig.to_html(include_plotlyjs="cdn", full_html=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self._plot_paths.append(output_path)
        return output_path
    
    def plot_cv_comparison(
        self,
        cv_results: Dict[str, Any],  # model_type -> CVResult
        filename: str = "cv_comparison",
    ) -> Path:
        """Create bar chart comparing models across CV metrics.
        
        Args:
            cv_results: Dict of model_type to CVResult
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        model_names = []
        metrics_data = {}
        
        for model_type, result in cv_results.items():
            model_names.append(result.model_name)
            for metric, value in result.mean_metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = {"means": [], "stds": []}
                metrics_data[metric]["means"].append(value)
                metrics_data[metric]["stds"].append(result.std_metrics[metric])
        
        # Create subplots for each metric
        n_metrics = len(metrics_data)
        fig = make_subplots(
            rows=1, cols=n_metrics,
            subplot_titles=[m.upper().replace("_", " ") for m in metrics_data.keys()],
            horizontal_spacing=0.08,
        )
        
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]
        
        for col, (metric, data) in enumerate(metrics_data.items(), 1):
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=data["means"],
                    error_y=dict(type="data", array=data["stds"], visible=True),
                    marker_color=colors[:len(model_names)],
                    showlegend=False,
                    hovertemplate="%{x}<br>%{y:.4f} ± %{error_y.array:.4f}<extra></extra>",
                ),
                row=1, col=col,
            )
        
        fig.update_layout(
            title=dict(
                text="<b>Cross-Validation Results</b><br><sup>Mean ± Std across 5 folds</sup>",
                x=0.5,
                xanchor="center",
            ),
            height=400,
            template="plotly_dark",
            margin=dict(t=80, b=60),
        )
        
        return self.save_plot(fig, filename)
    
    def plot_scatter_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        metadata: Optional[pd.DataFrame] = None,
        filename: str = "test_scatter",
    ) -> Path:
        """Create scatter plot of predicted vs actual values.
        
        Args:
            y_true: Actual values (log returns)
            y_pred: Predicted values (log returns)
            model_name: Name of the model
            metadata: Optional DataFrame with symbol, week_start for hover
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        # Compute correlation for display
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Build hover text with both log return and percentage
        if metadata is not None and len(metadata) == len(y_true):
            hover_text = [
                f"<b>{row['symbol']}</b><br>"
                f"Week: {row['week_start']}<br>"
                f"Actual: {actual:.4f} ({log_to_pct(actual):+.2f}%)<br>"
                f"Predicted: {pred:.4f} ({log_to_pct(pred):+.2f}%)<br>"
                f"Error: {pred - actual:.4f} ({log_to_pct(pred) - log_to_pct(actual):+.2f}%)"
                for (_, row), actual, pred in zip(metadata.iterrows(), y_true, y_pred)
            ]
        else:
            hover_text = [
                f"Actual: {actual:.4f} ({log_to_pct(actual):+.2f}%)<br>"
                f"Predicted: {pred:.4f} ({log_to_pct(pred):+.2f}%)"
                for actual, pred in zip(y_true, y_pred)
            ]
        
        fig = go.Figure()
        
        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode="markers",
                marker=dict(
                    size=5,
                    color="#2E86AB",
                    opacity=0.5,
                    line=dict(width=0.5, color="white"),
                ),
                text=hover_text,
                hoverinfo="text",
                name="Predictions",
            )
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#FF6B6B", dash="dash", width=2),
                name="Perfect Prediction",
            )
        )
        
        # Zero lines
        fig.add_hline(y=0, line_dash="dot", line_color="#666", opacity=0.5)
        fig.add_vline(x=0, line_dash="dot", line_color="#666", opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text=f"<b>{model_name}</b> - Predicted vs Actual Returns<br>"
                     f"<sup>R²={r2:.4f}, Corr={corr:.4f}, n={len(y_true)}</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Actual Return",
            yaxis_title="Predicted Return",
            height=600,
            width=700,
            template="plotly_dark",
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )
        
        # Make axes equal scale
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        
        return self.save_plot(fig, filename)
    
    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        model_name: str,
        top_n: int = 15,
        filename: str = "feature_importance",
    ) -> Path:
        """Create horizontal bar chart of feature importance.
        
        Args:
            importance_dict: Dict of feature name to importance score
            model_name: Name of the model
            top_n: Number of top features to show
            filename: Output filename
            
        Returns:
            Path to saved plot
        """
        # Sort by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]
        
        features = [f[0] for f in sorted_features][::-1]  # Reverse for horizontal bar
        importances = [f[1] for f in sorted_features][::-1]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=importances,
                y=features,
                orientation="h",
                marker_color="#2E86AB",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{model_name}</b> - Feature Importance<br>"
                     f"<sup>Top {top_n} features</sup>",
                x=0.5,
                xanchor="center",
            ),
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, 30 * len(features)),
            template="plotly_dark",
            margin=dict(l=150),
        )
        
        return self.save_plot(fig, filename)
    
    def create_inspector(self, filename: str = "_inspector") -> Path:
        """Create HTML inspector/gallery for all generated plots.
        
        Args:
            filename: Output filename (without extension)
            
        Returns:
            Path to inspector HTML
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Experiment Results Inspector</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            color: #4ECDC4;
            border-bottom: 2px solid #4ECDC4;
            padding-bottom: 10px;
        }}
        .plot-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .plot-card {{
            background: #252547;
            border-radius: 8px;
            padding: 15px;
            transition: transform 0.2s;
        }}
        .plot-card:hover {{
            transform: translateY(-3px);
            background: #2d2d5a;
        }}
        .plot-card a {{
            color: #4ECDC4;
            text-decoration: none;
            font-weight: 500;
        }}
        .plot-card a:hover {{
            text-decoration: underline;
        }}
        .plot-card p {{
            color: #888;
            font-size: 0.9em;
            margin: 5px 0 0 0;
        }}
    </style>
</head>
<body>
    <h1>📊 Experiment Results Inspector</h1>
    <div class="plot-list">
"""
        
        for plot_path in self._plot_paths:
            name = plot_path.stem.replace(".html", "").replace("_", " ").title()
            html_content += f"""
        <div class="plot-card">
            <a href="{plot_path.name}" target="_blank">{name}</a>
            <p>{plot_path.name}</p>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>"""
        
        output_path = self.output_dir / f"{filename}.html"
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return output_path

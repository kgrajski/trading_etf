#!/usr/bin/env python3
"""
07-visualize-predictions.py

Generate per-symbol prediction dashboards.

For each symbol, creates a 4-panel dashboard:
1. Actual vs Predicted time series (log return)
2. Actual vs Predicted time series (% return)
3. Prediction error histogram (dev vs test) - log
4. Prediction error histogram (dev vs test) - %

Uses out-of-fold (OOF) predictions from CV for development set
and final model predictions for test set.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.workflow.workflow_utils import print_summary, setup_logging, workflow_script

logger = setup_logging()


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "exp004_alpha_norm"
EXPERIMENT_DIR = Path("experiments") / EXPERIMENT_NAME
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OUTPUT_DIR = EXPERIMENT_DIR / "symbol_dashboards"

# Models to visualize (will detect from data)
MODELS = ["linear", "ridge", "lasso", "random_forest", "xgboost"]

# Colors for each model
MODEL_COLORS = {
    "linear": "#F18F01",       # Orange
    "ridge": "#2ECC71",        # Green
    "lasso": "#9B59B6",        # Purple
    "random_forest": "#E74C3C", # Red
    "xgboost": "#3498DB",      # Blue
}
COLOR_ACTUAL = "#FFFFFF"  # White for actual
COLOR_DEV_BG = "#3498DB"  # Blue for dev histogram
COLOR_TEST_BG = "#E74C3C"  # Red for test histogram


# =============================================================================
# Helper Functions
# =============================================================================

def load_predictions() -> pd.DataFrame:
    """Load and combine OOF and test predictions."""
    oof_path = PREDICTIONS_DIR / "oof_predictions.csv"
    test_path = PREDICTIONS_DIR / "test_predictions.csv"
    
    dfs = []
    
    if oof_path.exists():
        oof_df = pd.read_csv(oof_path, parse_dates=["week_start"])
        dfs.append(oof_df)
        print(f"  Loaded OOF predictions: {len(oof_df):,} rows")
    else:
        print(f"  Warning: OOF predictions not found at {oof_path}")
    
    if test_path.exists():
        test_df = pd.read_csv(test_path, parse_dates=["week_start"])
        dfs.append(test_df)
        print(f"  Loaded test predictions: {len(test_df):,} rows")
    
    if not dfs:
        raise FileNotFoundError("No prediction files found")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["symbol", "week_start"])
    
    return combined


def detect_models(df: pd.DataFrame) -> List[str]:
    """Detect which models have predictions in the DataFrame."""
    models = []
    for model in MODELS:
        if f"pred_{model}_log" in df.columns:
            models.append(model)
    return models


def create_symbol_dashboard(
    df: pd.DataFrame,
    symbol: str,
    models: List[str],
) -> go.Figure:
    """Create 4-panel dashboard for a single symbol with ALL models overlaid.
    
    Panels:
    1. Time series: Actual vs All Model Predictions (log return)
    2. Time series: Actual vs All Model Predictions (% return)
    3. Error histogram by model (dev set) - log
    4. Error histogram by model (dev set) - %
    """
    symbol_df = df[df["symbol"] == symbol].copy()
    symbol_df = symbol_df.sort_values("week_start")
    
    if len(symbol_df) == 0:
        return None
    
    # Get symbol name
    symbol_name = symbol_df["name"].iloc[0] if "name" in symbol_df.columns else symbol
    
    # Split dev and test
    dev_df = symbol_df[symbol_df["dataset"] == "dev"]
    test_df = symbol_df[symbol_df["dataset"] == "test"]
    
    # Create figure with 2x2 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Time Series (Log Return) - All Models",
            "Time Series (% Return) - All Models",
            "Error Distribution by Model (Log)",
            "Error Distribution by Model (%)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
    )
    
    # =========================================================================
    # Panel 1: Time series - Log return (all models)
    # =========================================================================
    # Actual (always shown)
    fig.add_trace(
        go.Scatter(
            x=symbol_df["week_start"],
            y=symbol_df["actual_log"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=COLOR_ACTUAL, width=2),
            marker=dict(size=5),
            legendgroup="actual",
            hovertemplate="<b>Actual</b><br>Week: %{x}<br>Value: %{y:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )
    
    # Each model's predictions
    for model in models:
        pred_col = f"pred_{model}_log"
        if pred_col in symbol_df.columns:
            color = MODEL_COLORS.get(model, "#888888")
            fig.add_trace(
                go.Scatter(
                    x=symbol_df["week_start"],
                    y=symbol_df[pred_col],
                    mode="lines",
                    name=model.replace("_", " ").title(),
                    line=dict(color=color, width=1.5, dash="dash"),
                    legendgroup=model,
                    hovertemplate=f"<b>{model}</b><br>Week: %{{x}}<br>Pred: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=1,
            )
    
    # Shade test region
    if len(test_df) > 0:
        test_start = test_df["week_start"].min()
        fig.add_vrect(
            x0=test_start, x1=symbol_df["week_start"].max(),
            fillcolor="#E74C3C", opacity=0.1,
            layer="below", line_width=0,
            row=1, col=1,
            annotation_text="TEST", annotation_position="top left",
        )
    
    fig.add_hline(y=0, line_dash="dot", line_color="#666", opacity=0.5, row=1, col=1)
    
    # =========================================================================
    # Panel 2: Time series - % return (all models)
    # =========================================================================
    fig.add_trace(
        go.Scatter(
            x=symbol_df["week_start"],
            y=symbol_df["actual_pct"],
            mode="lines+markers",
            name="Actual %",
            line=dict(color=COLOR_ACTUAL, width=2),
            marker=dict(size=5),
            legendgroup="actual",
            showlegend=False,
            hovertemplate="<b>Actual</b><br>Week: %{x}<br>Value: %{y:.2f}%<extra></extra>",
        ),
        row=1, col=2,
    )
    
    for model in models:
        pred_col = f"pred_{model}_pct"
        if pred_col in symbol_df.columns:
            color = MODEL_COLORS.get(model, "#888888")
            fig.add_trace(
                go.Scatter(
                    x=symbol_df["week_start"],
                    y=symbol_df[pred_col],
                    mode="lines",
                    name=f"{model} %",
                    line=dict(color=color, width=1.5, dash="dash"),
                    legendgroup=model,
                    showlegend=False,
                    hovertemplate=f"<b>{model}</b><br>Week: %{{x}}<br>Pred: %{{y:.2f}}%<extra></extra>",
                ),
                row=1, col=2,
            )
    
    if len(test_df) > 0:
        fig.add_vrect(
            x0=test_start, x1=symbol_df["week_start"].max(),
            fillcolor="#E74C3C", opacity=0.1,
            layer="below", line_width=0,
            row=1, col=2,
        )
    
    fig.add_hline(y=0, line_dash="dot", line_color="#666", opacity=0.5, row=1, col=2)
    
    # =========================================================================
    # Panel 3: Error histogram by model - Log (dev + test combined)
    # =========================================================================
    for model in models:
        pred_col = f"pred_{model}_log"
        if pred_col in symbol_df.columns:
            errors = symbol_df["actual_log"] - symbol_df[pred_col]
            color = MODEL_COLORS.get(model, "#888888")
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=f"{model} err",
                    marker_color=color,
                    opacity=0.5,
                    nbinsx=25,
                    legendgroup=model,
                    showlegend=False,
                    hovertemplate=f"<b>{model}</b><br>Error: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=2, col=1,
            )
    
    fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=1)
    
    # =========================================================================
    # Panel 4: Error histogram by model - %
    # =========================================================================
    for model in models:
        pred_col = f"pred_{model}_pct"
        if pred_col in symbol_df.columns:
            errors = symbol_df["actual_pct"] - symbol_df[pred_col]
            color = MODEL_COLORS.get(model, "#888888")
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=f"{model} err %",
                    marker_color=color,
                    opacity=0.5,
                    nbinsx=25,
                    legendgroup=model,
                    showlegend=False,
                    hovertemplate=f"<b>{model}</b><br>Error: %{{x:.2f}}%<br>Count: %{{y}}<extra></extra>",
                ),
                row=2, col=2,
            )
    
    fig.add_vline(x=0, line_dash="dash", line_color="white", row=2, col=2)
    
    # =========================================================================
    # Layout
    # =========================================================================
    # Compute summary stats for best model (ridge as reference)
    ref_model = "ridge" if "ridge" in models else models[0]
    pred_col_log = f"pred_{ref_model}_log"
    
    dev_mae = np.nan
    dev_dir_acc = np.nan
    test_mae = np.nan
    test_dir_acc = np.nan
    
    if len(dev_df) > 0 and pred_col_log in dev_df.columns:
        dev_errors = dev_df["actual_log"] - dev_df[pred_col_log]
        dev_mae = np.mean(np.abs(dev_errors))
        dev_dir_acc = np.mean(np.sign(dev_df["actual_log"]) == np.sign(dev_df[pred_col_log]))
    
    if len(test_df) > 0 and pred_col_log in test_df.columns:
        test_errors = test_df["actual_log"] - test_df[pred_col_log]
        test_mae = np.mean(np.abs(test_errors))
        test_dir_acc = np.mean(np.sign(test_df["actual_log"]) == np.sign(test_df[pred_col_log]))
    
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{symbol}</b> - {symbol_name}<br>"
                f"<sup>{ref_model.title()} reference: Dev MAE={dev_mae:.4f}, Dir.Acc={dev_dir_acc:.1%} (n={len(dev_df)}) | "
                f"Test MAE={test_mae:.4f}, Dir.Acc={test_dir_acc:.1%} (n={len(test_df)})</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=750,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        barmode="overlay",
    )
    
    # Axis labels
    fig.update_xaxes(title_text="Week", row=1, col=1)
    fig.update_xaxes(title_text="Week", row=1, col=2)
    fig.update_xaxes(title_text="Prediction Error (log)", row=2, col=1)
    fig.update_xaxes(title_text="Prediction Error (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="Log Return", row=1, col=1)
    fig.update_yaxes(title_text="% Return", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    # Link time axes of top row panels
    fig.update_xaxes(matches="x", row=1, col=2)
    
    return fig


def create_inspector(symbols: List[str], output_dir: Path) -> Path:
    """Create HTML inspector for all symbol dashboards."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Dashboards - {EXPERIMENT_NAME}</title>
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
        .info {{
            color: #888;
            margin-bottom: 20px;
        }}
        .symbol-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 20px;
        }}
        .symbol-card {{
            background: #252547;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            transition: transform 0.2s, background 0.2s;
        }}
        .symbol-card:hover {{
            transform: translateY(-2px);
            background: #2d2d5a;
        }}
        .symbol-card a {{
            color: #4ECDC4;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
        }}
        .symbol-card a:hover {{
            text-decoration: underline;
        }}
        .search-box {{
            padding: 10px;
            font-size: 16px;
            width: 300px;
            border: none;
            border-radius: 6px;
            background: #252547;
            color: #eee;
            margin-bottom: 10px;
        }}
        .search-box:focus {{
            outline: 2px solid #4ECDC4;
        }}
    </style>
</head>
<body>
    <h1>📈 Prediction Dashboards</h1>
    <p class="info">Experiment: {EXPERIMENT_NAME} | All Models | Symbols: {len(symbols)}</p>
    
    <input type="text" class="search-box" id="searchBox" placeholder="Search symbols..." onkeyup="filterSymbols()">
    
    <div class="symbol-grid" id="symbolGrid">
"""
    
    for symbol in sorted(symbols):
        html_content += f"""
        <div class="symbol-card" data-symbol="{symbol.lower()}">
            <a href="{symbol}_dashboard.html" target="_blank">{symbol}</a>
        </div>
"""
    
    html_content += """
    </div>
    
    <script>
        function filterSymbols() {
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            const cards = document.querySelectorAll('.symbol-card');
            cards.forEach(card => {
                const symbol = card.getAttribute('data-symbol');
                card.style.display = symbol.includes(searchText) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>"""
    
    inspector_path = output_dir / "_inspector.html"
    with open(inspector_path, "w") as f:
        f.write(html_content)
    
    return inspector_path


# =============================================================================
# Main
# =============================================================================

@workflow_script("07-visualize-predictions")
def main() -> None:
    """Generate per-symbol prediction dashboards."""
    
    print("=" * 80)
    print(f"PREDICTION VISUALIZATION: {EXPERIMENT_NAME}")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load predictions
    print("Step 1: Loading predictions...")
    df = load_predictions()
    print(f"  Total rows: {len(df):,}")
    print()
    
    # Detect available models
    models = detect_models(df)
    print(f"  Models detected: {models}")
    print()
    
    # Get unique symbols
    symbols = df["symbol"].unique().tolist()
    print(f"Step 2: Generating dashboards for {len(symbols)} symbols...")
    
    generated = 0
    for i, symbol in enumerate(symbols):
        fig = create_symbol_dashboard(df, symbol, models)
        if fig is not None:
            output_path = OUTPUT_DIR / f"{symbol}_dashboard.html"
            fig.write_html(str(output_path))
            generated += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(symbols)}")
    
    print(f"  Generated {generated} dashboards")
    print()
    
    # Create inspector
    print("Step 3: Creating inspector...")
    inspector_path = create_inspector(symbols, OUTPUT_DIR)
    print(f"  Inspector: {inspector_path}")
    print()
    
    print_summary(
        experiment=EXPERIMENT_NAME,
        symbols=len(symbols),
        dashboards_generated=generated,
        output_directory=str(OUTPUT_DIR),
        inspector=str(inspector_path),
    )


if __name__ == "__main__":
    main()

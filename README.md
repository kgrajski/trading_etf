# Trading ETF

ETF Weekly Trading Strategy with Back-Testing

## Overview

This project implements a research and development workbench for developing and testing simple trading strategies for a population of ETFs by analyzing historical weekly pricing and volume data. The goal is to generate ranked recommendations for week-ahead trading periods together with anticipated gain.

## Project Structure

```
trading_etf/
├── src/
│   ├── data/              # Data sources, ETL, feature engineering
│   ├── strategies/        # Trading strategy implementations
│   ├── backtesting/       # Portfolio backtesting engine
│   ├── visualization/     # Results visualization
│   └── workflow/          # Workflow scripts
├── data/
│   ├── external/          # Raw weekly data by symbol
│   ├── interim/           # Standardized weekly data
│   ├── processed/         # Features added
│   └── metadata/          # Symbol metadata, filters
├── experiments/           # Experiment configs and results
└── reports/               # Generated reports and visualizations
```

## Quick Start

### 1. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Alpaca API

Create a `.env` file in the project root:

```
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
```

### 3. Run Workflow

```bash
# Step 1: Get ETF universe
python src/workflow/00-get-etf-universe.py

# Step 2: Fetch weekly data
python src/workflow/02-fetch-weekly-data.py

# Step 3: Build features
python src/workflow/03-build-weekly-features.py

# Step 4: Run experiment
python src/workflow/04-run-experiment.py experiments/exp001/config.json
```

## Experiment Configuration

Experiments are configured via JSON files in the `experiments/` directory. Example:

```json
{
  "experiment_id": "exp001",
  "symbols": ["SPY", "QQQ", "IWM"],
  "time_unit": "week",
  "historical_range": {
    "start": "2021-01-01",
    "end": "2024-12-31"
  },
  "strategy": {
    "name": "momentum",
    "hyperparameters": {
      "lookback_weeks": [2, 4, 8],
      "momentum_threshold": [0.02, 0.05, 0.10],
      "volume_threshold": [1.2, 1.5, 2.0],
      "max_positions": [3, 5, 10]
    }
  },
  "initial_capital": 100000.0,
  "walk_forward": {
    "enabled": false,
    "train_weeks": 52,
    "operate_weeks": 1
  }
}
```

## Strategies

### Momentum Strategy

The momentum strategy enters positions based on:
- Positive price momentum (N-week return > threshold)
- Volume above average
- Price above moving average

Exits based on:
- Momentum reversal
- Stop-loss
- Profit target

## Outputs

Each experiment generates:
- Portfolio equity curve
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Trade-by-trade results
- Weekly recommendations

Results are saved in `experiments/{experiment_id}/results.json`

## Development

This project follows a modular design:
- **Data Sources**: Factory pattern for multiple data providers (Alpaca, etc.)
- **Strategies**: Abstract base class for easy strategy implementation
- **Backtesting**: Portfolio-level backtesting with weekly rebalancing
- **Experiments**: Config-driven experiment framework with hyperparameter search

## License

MIT

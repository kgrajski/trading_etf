# Project Context & Continuity Guide

## Project Status

This is an **ETF weekly trading strategy project** (`trading_etf`). The MVP foundation is complete and ready for testing.

## What Was Just Created

### Project Structure
- **Data Layer**: Alpaca source integration, symbol filtering, weekly feature engineering
- **Strategy Layer**: Base strategy class, momentum strategy implementation, strategy factory
- **Backtesting**: Portfolio backtester for weekly trading with multiple positions
- **Experiments**: Experiment runner with Ray parallelization, walk-forward optimization support
- **Visualization**: Portfolio visualizer for equity curves and performance metrics
- **Workflow**: 5 numbered scripts (00-05) for the complete pipeline

### Key Files Created
- `src/data/`: Data sources, ETL, feature engineering, symbol management
- `src/strategies/`: Trading strategy implementations (momentum strategy ready)
- `src/backtesting/`: Portfolio backtester and experiment runner
- `src/workflow/`: Complete workflow pipeline scripts
- `experiments/exp001/config.json`: Initial experiment configuration
- `README.md`: Project overview and quick start
- `SETUP.md`: Detailed setup instructions

### Default Configuration
- **15 default ETFs**: SPY, QQQ, IWM, DIA, VTI, XLE, XLF, XLI, XLK, XLV, XLP, XLY, XLB, XLU
- **Momentum strategy**: Price/volume momentum with configurable hyperparameters
- **Initial capital**: $100,000
- **Time unit**: Weekly (Monday-Friday)

## Current State

✅ **Completed:**
- Project structure and directory layout
- All core modules and classes
- Workflow scripts (00-05)
- Initial experiment configuration
- Documentation (README, SETUP)

⏳ **Not Yet Done:**
- Virtual environment setup
- Dependency installation
- Alpaca API configuration
- Data fetching and testing
- Running first experiment

## Next Steps (In Order)

### 1. Set Up Environment
```bash
cd /Users/kag/Development/Projects/trading_etf
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Alpaca API
Create `.env` file in project root:
```
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
```

Get your API keys from https://app.alpaca.markets/ (use Paper Trading keys for testing)

### 3. Test Workflow Step by Step
```bash
# Step 1: Get ETF universe (no API needed)
python src/workflow/00-get-etf-universe.py

# Step 2: Fetch weekly data (needs Alpaca API)
python src/workflow/02-fetch-weekly-data.py

# Step 3: Build features
python src/workflow/03-build-weekly-features.py

# Step 4: Run experiment
python src/workflow/04-run-experiment.py experiments/exp001/config.json

# Step 5: Visualize results
python src/workflow/05-visualize-results.py experiments/exp001/results.json
```

## Design Decisions Made

1. **Project Independence**: This is a standalone project with no dependencies on other projects. All code is self-contained within `trading_etf`. If similar functionality exists elsewhere, it is duplicated here rather than shared.
2. **Weekly Time Unit**: Fixed to weekly bars (Monday-Friday) for ETF sector rotation strategy
3. **Portfolio-Level Backtesting**: Supports multiple positions simultaneously
4. **Equal Weight Position Sizing**: Each position gets capital / max_positions
5. **Global Hyperparameters**: Same parameters for all symbols (simpler, faster)
6. **Factory Patterns**: Data sources and strategies use factory pattern for extensibility
7. **Ray Parallelization**: Hyperparameter search uses Ray for parallel execution

## Architecture Highlights

- **Modular Design**: Easy to add new strategies, data sources, or features
- **Config-Driven**: Experiments defined via JSON config files
- **Walk-Forward Ready**: Framework supports walk-forward optimization

## Potential Issues to Watch For

1. **Alpaca API Rate Limits**: May need delays between requests
2. **Data Format Differences**: Weekly data format may need adjustment
3. **Missing Features**: Some features may need previous week data (handled in feature engineering)
4. **Date Range**: Default fetches 3 years - may take time for 15 symbols
5. **Ray Initialization**: May need to adjust worker count based on system

## Files to Reference

- `README.md`: Project overview and quick start
- `SETUP.md`: Detailed setup instructions
- `CODING_STANDARDS.md`: Code formatting and style guidelines
- `experiments/exp001/config.json`: Example experiment configuration
- `src/strategies/momentum_strategy.py`: Strategy implementation example
- `src/backtesting/portfolio_backtester.py`: Backtesting engine

## When Starting a New Chat

Simply say:
> "Please read CONTEXT.md to get oriented on this project."

This file provides all the context needed to continue working on the project.

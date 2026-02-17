# Setup Guide

## Initial Setup

### 1. Create Virtual Environment

```bash
cd trading_etf
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Configure Alpaca API

Create a `.env` file in the project root:

```
APCA_API_KEY_ID=your_key_here
APCA_API_SECRET_KEY=your_secret_here
```

Get your API keys from https://app.alpaca.markets/ (use Paper Trading keys for testing)

## Running the Workflow

### Step 1: Get ETF Universe

```bash
python src/workflow/00-get-etf-universe.py
```

This creates a default list of 15 major ETFs and saves it to `data/metadata/symbols.json`.

### Step 2: Fetch Weekly Data

```bash
python src/workflow/02-fetch-weekly-data.py
```

This fetches weekly bars from Alpaca for all symbols in the universe. Data is saved to `data/external/{symbol}/`.

**Note**: This may take several minutes depending on the number of symbols and date range.

### Step 3: Build Features

```bash
python src/workflow/03-build-weekly-features.py
```

This processes the raw weekly data and adds features (momentum, volume, etc.). Processed data is saved to `data/processed/{symbol}/weekly_features.csv`.

### Step 4: Run Experiment

```bash
python src/workflow/04-run-experiment.py experiments/exp001/config.json
```

This runs the backtest with hyperparameter search. Results are saved to `experiments/exp001/results.json`.

### Step 5: Visualize Results (Optional)

```bash
python src/workflow/05-visualize-results.py experiments/exp001/results.json
```

## Experiment Configuration

Edit `experiments/exp001/config.json` to customize:

- **symbols**: List of ETFs to trade
- **historical_range**: Date range for backtesting
- **strategy.hyperparameters**: Parameter grid for search
- **initial_capital**: Starting capital
- **walk_forward**: Enable walk-forward optimization

## Project Structure

- `src/data/`: Data sources, ETL, feature engineering
- `src/strategies/`: Trading strategy implementations
- `src/backtesting/`: Portfolio backtesting engine
- `src/visualization/`: Results visualization
- `src/workflow/`: Workflow scripts
- `experiments/`: Experiment configs and results
- `data/`: Raw and processed data

## Next Steps

1. Run the initial workflow with the default ETF list
2. Review results and adjust hyperparameters
3. Experiment with different strategies (add new ones in `src/strategies/`)
4. Enable walk-forward optimization for more realistic results
5. Add more ETFs to the universe

## Troubleshooting

- **Alpaca API errors**: Check your `.env` file and API credentials
- **No data returned**: Verify symbols are valid and date range is correct
- **Import errors**: Make sure you've run `pip install -e .` to install the package

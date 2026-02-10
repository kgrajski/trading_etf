# ETF Weekly Trading System with Agentic AI Analyst
## MVP R&D Workbench Purposes Only

A research-to-production platform for developing, backtesting, and operating a weekly ETF trading strategy — augmented by an **agentic AI analyst** built with LangGraph that provides qualitative news-driven assessments of trade candidates.

## What This Project Does

Each week, the system:

1. **Fetches** daily market data for ~2,200 ETFs (Alpaca API)
2. **Engineers** features — returns, volatility, momentum, cyclical encodings
3. **Backtests** a mean-reversion strategy to validate parameters
4. **Generates** ranked trade candidates with entry/exit prices and position sizes
5. **Runs an AI analyst** that searches financial news, assesses each candidate, and produces a qualitative overlay (GREEN / YELLOW / RED flags)
6. **Produces** an interactive HTML dashboard combining quantitative and qualitative signals

The entire pipeline runs via a single script (`scripts/weekly_update.sh`) and completes in ~10 minutes.

---

## Architecture

```mermaid
flowchart TD
    subgraph pipeline ["Weekly Pipeline"]
        A["Data Fetch<br>(Alpaca API)"] --> B["Feature Engineering<br>(pandas)"]
        B --> C["Backtest<br>(mean-reversion)"]
        C --> D["Trade Generation<br>(ranked candidates)"]
    end

    subgraph agent ["Agentic AI Analyst"]
        E["load"] --> F["fetch_news<br>(Tavily Search)"]
        F --> G["analyze_themes<br>(LLM)"]
        G --> H["analyze_symbols<br>(LLM × N)"]
        H --> I["review_and_refine<br>(LLM — Reflection)"]
    end

    D --> E
    I --> J["Interactive Dashboard<br>+ MLflow Tracking"]
```

## Agentic AI — The Interesting Part

The analyst (`src/analyst/`) implements a **LangGraph workflow** with the **Reflection Pattern**:

| Node | What It Does | Tool / Model |
|------|-------------|-------------|
| `load` | Initialize state | — |
| `fetch_news` | Search financial news per symbol | Tavily API |
| `analyze_themes` | Identify macro themes across candidates | LLM (Gemini / GPT) |
| `analyze_symbols` | Per-symbol qualitative assessment | LLM × N candidates |
| `review_and_refine` | **Reflection** — senior analyst reviews junior's work, adjusts flags | LLM (optionally different model) |

**Key design choices:**

- **Reflection Pattern**: The reviewer node critiques the initial analysis, catches inconsistencies, and adjusts conviction flags — mimicking a senior/junior analyst dynamic
- **Multi-Model Routing**: Different LLMs for analysis vs. review (e.g., Gemini Flash for bulk, GPT-4o for critical review). Configurable via environment variables
- **Qualitative-Only Scope**: The agent assesses *news*, not quant metrics. It answers: "Is this drop transient or structural?" The quant system handles everything else
- **MLflow Integration**: Every run logs parameters, token usage, costs, flag distributions, and artifacts

### Databricks Deployment

The agent has been ported to a self-contained Databricks notebook (`notebooks/analyst_databricks.py`) with:
- Widget-based API key configuration
- Native MLflow experiment tracking
- `displayHTML()` for interactive reports
- Zero dependencies on the local `src/` package structure

---

## Project Structure

```
trading_etf/
├── scripts/
│   └── weekly_update.sh            # One-command weekly pipeline
├── src/
│   ├── analyst/                     # Agentic AI analyst
│   │   ├── graph.py                #   LangGraph workflow (5 nodes)
│   │   ├── run.py                  #   CLI entry point + MLflow logging
│   │   ├── instrumentation.py      #   Detailed metrics (tokens, cost, latency)
│   │   ├── logging_config.py       #   Trace logging (full prompts/responses)
│   │   └── tools/
│   │       └── search.py           #   Tavily web search tool
│   ├── data/                        # Data sources, ETL, feature engineering
│   │   ├── alpaca_source.py        #   Alpaca Markets API integration
│   │   ├── market_data_source.py
│   │   ├── weekly_feature_engineering.py
│   │   ├── etf_filter.py
│   │   ├── symbol_filter.py
│   │   ├── symbol_list_manager.py
│   │   ├── metadata_manager.py
│   │   └── data_source_factory.py
│   ├── backtesting/                 # Strategy backtesting engine
│   │   ├── mean_reversion_backtester.py
│   │   ├── portfolio_backtester.py
│   │   └── experiment_runner.py
│   ├── strategies/                  # Trading strategies
│   │   ├── base_strategy.py
│   │   ├── momentum_strategy.py
│   │   └── strategy_factory.py
│   ├── models/                      # ML model registry
│   ├── training/                    # ML model training (sklearn, XGBoost, PyTorch)
│   │   ├── cross_validator.py
│   │   ├── data_loader.py
│   │   ├── feature_builder.py
│   │   ├── evaluator.py
│   │   ├── model_factory.py
│   │   ├── normalizer.py
│   │   ├── prediction_analyzer.py
│   │   ├── visualizer.py
│   │   └── models/                 #   Linear, tree, classification models
│   ├── reports/                     # Generated weekly retrospectives
│   ├── utils/                       # Date utilities, device detection
│   ├── visualization/               # Plotly-based visualizations
│   └── workflow/
│       ├── config.py                # Shared configuration
│       ├── workflow_utils.py        # Shared utilities
│       ├── pipeline/                # Production scripts (00–05, 21a–21b)
│       └── research/               # Research scripts (06–22)
├── notebooks/
│   ├── analyst_databricks.py       # Databricks notebook (self-contained)
│   ├── analyst_databricks_confirmed.py  # Validated on Databricks
│   ├── test_local.py               # Local validation runner
│   └── sample_candidates.csv       # Sample data for testing
├── docs/                            # Feature docs, weekly update guide
├── experiments/exp001/              # Sample experiment config
├── pyproject.toml                   # Package configuration
├── setup.py
├── requirements.txt
├── RESEARCH_LOG.md                  # Research journey narrative
├── CODING_STANDARDS.md
├── CONTEXT.md
└── SETUP.md
```

Note: The pipeline generates `data/`, `experiments/` results, `pre_production/` candidates, and `logs/` at runtime — these are gitignored and reproducible by rerunning the workflow.

## Research Journey

This project evolved through several phases:

1. **Momentum strategies** — classic trend-following with grid search over lookback windows, volume thresholds, and position sizing
2. **ML classification** — scikit-learn, XGBoost, and PyTorch models predicting weekly direction. Explored rolling cross-validation, regime detection, and feature importance
3. **Mean-reversion** — the current strategy. Buy ETFs that dropped significantly (>2σ), targeting a 1–3 week bounce. Backtested across ~2,200 symbols with parameter optimization
4. **Agentic AI overlay** — LangGraph-based analyst that adds qualitative news assessment on top of the quantitative signal
5. **Databricks deployment** — ported the agent to Databricks with MLflow tracking

The `src/workflow/research/` directory contains 25+ experiment scripts documenting this evolution.

---

## Quick Start

### Prerequisites

- Python 3.11+
- API keys: [Alpaca](https://alpaca.markets/) (market data), [Tavily](https://tavily.com/) (news search), and one of: [Google AI](https://ai.google.dev/) (Gemini) or [OpenAI](https://platform.openai.com/) (GPT)

### Setup

```bash
git clone https://github.com/kgrajski/trading_etf.git
cd trading_etf

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets

TAVILY_API_KEY=your_tavily_key
GOOGLE_API_KEY=your_google_key
ANALYST_LLM_MODEL=gemini-2.0-flash
```

### Run the Weekly Pipeline

```bash
bash scripts/weekly_update.sh
```

### Run Just the Analyst

```bash
python -m src.analyst.run path/to/candidates.csv
```

Output (JSON report, HTML report, graph visualization) is saved alongside the input CSV.

### View MLflow Results

```bash
mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

---

## Technologies

| Layer | Technologies |
|-------|-------------|
| **Data** | Alpaca Markets API, pandas, NumPy |
| **Strategy** | Custom mean-reversion backtester, scipy |
| **ML** | scikit-learn, XGBoost, PyTorch |
| **Agentic AI** | LangGraph, LangChain, Tavily Search |
| **LLMs** | Gemini 2.0 Flash, GPT-4o, Claude 3.5 Sonnet (multi-model routing) |
| **Tracking** | MLflow (local + Databricks) |
| **Visualization** | Plotly, custom HTML dashboards |
| **Deployment** | Databricks notebooks, DBFS |
| **Infrastructure** | Python 3.11, dotenv, shell pipeline |

## License

MIT

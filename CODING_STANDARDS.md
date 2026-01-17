# Coding Standards - trading_etf Project

> *Based on [PEP 8](https://peps.python.org/pep-0008/), [Black](https://black.readthedocs.io/), and Python community best practices*

## Code Formatting

### **REQUIRED: Use Black Formatter**

All Python code MUST be formatted using [Black](https://black.readthedocs.io/):

```bash
# Format your code before committing
black src/ tests/ scripts/

# Check if code is properly formatted
black --check src/ tests/ scripts/
```

**Configuration**: We use Black's default settings (88 character line length) as defined in `pyproject.toml`.

### **REQUIRED: Use Absolute Imports with `src.` Prefix**

All internal module imports MUST use absolute imports with the `src.` prefix:

```python
# ✅ CORRECT
from src.data.data_source_factory import DataSourceFactory
from src.strategies.momentum_strategy import MomentumStrategy
from src.backtesting.portfolio_backtester import PortfolioBacktester
from src.utils.date_utils import get_week_start

# ❌ INCORRECT - Never use these patterns
from data.data_source_factory import DataSourceFactory  # Missing src prefix
from .momentum_strategy import MomentumStrategy         # Relative import
from ..utils.date_utils import get_week_start          # Relative import
```

### Import Organization (Following PEP 8)

Group imports in this order with blank lines between groups:

```python
# 1. Standard library imports
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party library imports
import numpy as np
import pandas as pd
import ray
from alpaca.data.historical import StockHistoricalDataClient

# 3. Internal module imports (always with src. prefix)
from src.data.data_source_factory import DataSourceFactory
from src.utils.date_utils import get_week_start
```

## Code Quality Standards

### Type Hints (Following PEP 484)

Use type hints for all function signatures and class methods:

```python
# ✅ CORRECT
def fetch_weekly_data(
    symbol: str,
    start_date: str,
    end_date: str,
    data_source: Optional[MarketDataSource] = None,
) -> pd.DataFrame:
    """Fetch weekly market data for a symbol."""
    return df

# ❌ INCORRECT
def fetch_weekly_data(symbol, start_date, end_date, data_source=None):
    return df
```

### Documentation (Following PEP 257)

All public modules, classes, and functions MUST have docstrings:

```python
def calculate_momentum(
    prices: pd.Series,
    lookback_periods: int,
    min_volume: Optional[float] = None
) -> pd.Series:
    """Calculate price momentum indicator.

    Args:
        prices: Series of closing prices
        lookback_periods: Number of periods to look back
        min_volume: Optional minimum volume filter

    Returns:
        Series containing momentum values

    Raises:
        ValueError: If lookback_periods exceeds data length
    """
```

### Naming Conventions (Following PEP 8)

- **Functions and variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private attributes**: `_leading_underscore`
- **Files and modules**: `snake_case.py`

## Workflow Script Pattern

Scripts in `src/workflow/` should follow this pattern:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brief description of what this workflow script does.

Author: kag
Created: YYYY-MM-DD
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.symbol_list_manager import SymbolListManager


def main():
    """Main workflow function."""
    # =========================================================================
    # INTRO LOGGING
    # =========================================================================
    script_name = "00-get-etf-universe"
    start_time = time.perf_counter()
    print("*** " + script_name + " - START ***")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # CONFIGURATION (all config vars here - no CLI args)
    # =========================================================================
    project_dir = project_root
    metadata_dir = os.path.join(project_dir, "data", "metadata")
    
    # Config variables...
    
    # =========================================================================
    # MAIN PROCESSING
    # =========================================================================
    
    # Your code here...
    
    # =========================================================================
    # OUTRO LOGGING
    # =========================================================================
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    print("=" * 80)
    print(f"Total execution time: {total_duration:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("*** " + script_name + " - END ***")


if __name__ == "__main__":
    main()
```

**Key principles:**
- 100% self-contained (no CLI arguments, except for config file paths where needed)
- All configuration at top of `main()`
- Intro/outro logging for reproducibility
- "One button gets it all"

## Automated Tools

### Required Tools

- **Black**: Code formatting (`black src/ tests/ scripts/`)
- **isort**: Import sorting (`isort src/ tests/ scripts/`)
- **flake8**: Style and quality checking

### Quick Commands

```bash
# Format all code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Check style compliance
flake8 src/ tests/ scripts/

# Run all formatting (no prompting)
black src/ tests/ scripts/ && isort src/ tests/ scripts/
```

## For AI Assistants

When working on this codebase:

1. **ALWAYS** format code with Black defaults (88 char line length)
2. **ALWAYS** use absolute imports with `src.` prefix for internal modules
3. **NEVER** use relative imports (`from .` or `from ..`)
4. **ALWAYS** add type hints to function signatures
5. **ALWAYS** include docstrings for public functions and classes
6. Follow PEP 8 naming conventions
7. Follow the workflow script pattern for `src/workflow/` scripts
8. If you see non-compliant code, fix it immediately

## References

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Black - The Uncompromising Code Formatter](https://black.readthedocs.io/)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)

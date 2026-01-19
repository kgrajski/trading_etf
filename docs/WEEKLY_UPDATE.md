# Weekly Data Update Process

This document describes how to perform weekly data updates for the ETF trading project.

## Overview

The weekly update process:
1. Fetches new daily data from Alpaca (incremental mode)
2. Regenerates weekly features from daily data
3. Regenerates visualization dashboards
4. Rebuilds the cross-sectional feature matrix

## Schedule

### Recommended: Saturday 8:00 AM

Friday's market data typically becomes fully available on Alpaca by Saturday morning. Running on Saturday ensures all Friday data is captured.

### Alternative: Sunday Morning

If Saturday updates fail due to delayed data, Sunday morning provides a buffer.

### Avoid: Friday Evening

Friday evening may be too early for complete Friday data, especially the official closing auction prices.

## Manual Execution

### Quick Run

```bash
cd /Users/kag/Development/Projects/trading_etf
source .venv/bin/activate
./scripts/weekly_update.sh
```

### With Logging

```bash
cd /Users/kag/Development/Projects/trading_etf
source .venv/bin/activate
mkdir -p logs
./scripts/weekly_update.sh 2>&1 | tee logs/update_$(date +%Y%m%d).log
```

### Step-by-Step Manual Run

If you prefer to run each step individually:

```bash
cd /Users/kag/Development/Projects/trading_etf
source .venv/bin/activate

# Step 1: Fetch new data (incremental mode)
python src/workflow/02-fetch-daily-data.py

# Step 2: Generate weekly features
python src/workflow/03-generate-features.py

# Step 3: Generate visualizations (optional but recommended)
python src/workflow/04-visualize-features.py

# Step 4: Rebuild feature matrix
python src/workflow/05-build-feature-matrix.py
```

## Automated Execution (Cron)

### macOS/Linux Cron Setup

1. Open crontab editor:
   ```bash
   crontab -e
   ```

2. Add the following line (runs Saturday 8:00 AM):
   ```
   0 8 * * 6 /Users/kag/Development/Projects/trading_etf/scripts/weekly_update.sh >> /Users/kag/Development/Projects/trading_etf/logs/cron.log 2>&1
   ```

3. Create logs directory:
   ```bash
   mkdir -p /Users/kag/Development/Projects/trading_etf/logs
   ```

### Cron Expression Explained

```
0 8 * * 6
│ │ │ │ │
│ │ │ │ └── Day of week (6 = Saturday)
│ │ │ └──── Month (every month)
│ │ └────── Day of month (every day)
│ └──────── Hour (8 AM)
└────────── Minute (0)
```

## Configuration

### Data Tier

The update uses the `iex` (free) tier by default. To change:

Edit `src/workflow/config.py`:
```python
DATA_TIER: str = "sip"  # For paid tier
```

### Incremental Mode

Incremental mode is enabled by default, which:
- Checks existing data for each symbol
- Fetches only new data since the last update
- Re-fetches the last 5 days to catch any corrections

To force a full refresh, edit `src/workflow/config.py`:
```python
INCREMENTAL_MODE: bool = False
```

### Symbol Filter

By default, only symbols starting with "X" are processed (MVP mode):

```python
SYMBOL_PREFIX_FILTER: Optional[str] = "X"
```

To process all symbols, set to `None`:
```python
SYMBOL_PREFIX_FILTER: Optional[str] = None
```

## Troubleshooting

### Alpaca API Errors

**Rate Limiting**: The script includes built-in delays (0.3s between requests). If you still hit rate limits, increase `API_DELAY_SECONDS` in `config.py`.

**Authentication**: Ensure `.env` file contains valid credentials:
```
APCA_API_KEY_ID=your_key_id
APCA_API_SECRET_KEY=your_secret_key
```

### Missing Data

**Symbol Not Found**: Some symbols may not have data for certain days (holidays, delistings). The script logs these but continues.

**Incomplete Weeks**: Weeks with fewer than 5 trading days (holidays) are still processed. The `trading_days` column in weekly data reflects actual trading days.

### Disk Space

The data directory grows slowly:
- Daily CSVs: ~50 KB per symbol per year
- Weekly CSVs: ~10 KB per symbol
- Visualizations: ~80 KB per symbol (uncompressed HTML)
- Feature matrix: ~1 MB (Parquet)

### Log Files

Check logs for errors:
```bash
# View latest cron log
tail -100 logs/cron.log

# View specific update log
cat logs/update_20260119.log
```

## Data Locations

| Data | Location |
|------|----------|
| Daily bars | `data/historical/iex/daily/` |
| Weekly features | `data/historical/iex/weekly/` |
| Visualizations | `data/visualizations/iex/` |
| Feature matrix | `data/features/iex/` |
| Fetch manifest | `data/metadata/fetch_manifest.json` |

## Validation

After each update, verify:

1. **Check new data**:
   ```bash
   tail -5 data/historical/iex/daily/XLF.csv
   ```

2. **Check weekly features**:
   ```bash
   tail -3 data/historical/iex/weekly/XLF.csv
   ```

3. **Inspect visualizations**:
   Open `data/visualizations/iex/_inspector.html` in a browser.

4. **Check feature matrix**:
   ```python
   import pandas as pd
   fm = pd.read_parquet("data/features/iex/feature_matrix.parquet")
   print(fm.shape)
   print(fm.index[-5:])
   ```

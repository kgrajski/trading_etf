# -*- coding: utf-8 -*-
"""
Centralized configuration for workflow scripts.

All configurable parameters in one place for easy modification.
Individual scripts can override these as needed.
"""

from typing import List, Optional, Set

# =============================================================================
# DATA TIER CONFIGURATION
# =============================================================================

# Data tier: "iex" (free) or "sip" (paid)
# Determines data source and output directory structure
DATA_TIER: str = "iex"

# =============================================================================
# SYMBOL FILTERING
# =============================================================================

# MVP filter: Set to a prefix to limit symbols (e.g., "X" for testing)
# Set to None to process all symbols
SYMBOL_PREFIX_FILTER: Optional[str] = "X"

# =============================================================================
# DATE RANGE CONFIGURATION
# =============================================================================

# Historical date range for data fetching
# Set to None for dynamic calculation
DATE_RANGE_START: Optional[str] = "2024-01-01"
DATE_RANGE_END: Optional[str] = None  # None = today

# Years of history to fetch if DATE_RANGE_START is None
YEARS_OF_HISTORY: int = 2

# =============================================================================
# EXCHANGE FILTERING (Script 01)
# =============================================================================

# Allowed exchanges from Alpaca AssetExchange enum
# ARCA is primary for most ETFs
ALLOWED_EXCHANGES: Set[str] = {
    "ARCA",
    "NYSE",
    "NASDAQ",
    "BATS",
    "AMEX",
}

# Patterns indicating leveraged or inverse ETFs (exclude these)
LEVERAGED_INVERSE_PATTERNS: List[str] = [
    "ultra",
    "2x",
    "3x",
    "-2x",
    "-3x",
    "inverse",
    "short",
    "bear",
    "daily",  # Often indicates leveraged daily reset
    "proshares ultra",
    "proshares short",
    "direxion",
]

# =============================================================================
# FEATURE CONFIGURATION (Script 03)
# =============================================================================

# Features to derive from daily data
FEATURES_L1 = {
    "L1_log_return": True,
    "L1_log_range": True,
    "L1_log_volume": True,
    "L1_log_avg_daily_volume": True,
    "L1_intra_week_volatility": True,
}

FEATURES_L2 = {
    "L2_log_volume_delta": True,
    "L2_log_return_ma4": True,
    "L2_log_volume_ma4": True,
    "L2_momentum_4w": True,
    "L2_volatility_ma4": True,
    "L2_log_return_ma12": True,
    "L2_log_volume_ma12": True,
    "L2_momentum_12w": True,
    "L2_volatility_ma12": True,
}

# Combine for convenience
FEATURES_ENABLED = {**FEATURES_L1, **FEATURES_L2}

# Lookback periods (in weeks)
LOOKBACK_PERIODS = {
    "short": 1,  # For week-over-week deltas
    "medium": 4,  # For moving averages
    "long": 12,  # For longer trends
}

# =============================================================================
# FEATURE MATRIX CONFIGURATION (Script 05)
# =============================================================================

# Features to include in the cross-sectional feature matrix
MATRIX_FEATURES: List[str] = [
    "log_return",
    "log_range",
    "log_volume",
    "momentum_4w",
    "momentum_12w",
    "intra_week_volatility",
    "log_volume_delta",
]

# Prediction configuration
PREDICTION_HORIZON: int = 1  # Predict +1 week ahead
TARGET_FEATURE: str = "log_return"  # What to predict

# =============================================================================
# VISUALIZATION CONFIGURATION (Script 04)
# =============================================================================

# Whether to gzip compress HTML outputs
# False = can open directly in browser
# True = requires local server (smaller files)
COMPRESS_HTML: bool = False

# Color scheme for visualizations
VIZ_COLORS = {
    "price": "#2E86AB",
    "price_fill": "rgba(46, 134, 171, 0.2)",
    "volume": "#A23B72",
    "positive": "#28A745",
    "negative": "#DC3545",
    "ma4": "#FF8C00",
    "ma12": "#6F42C1",
    "zero": "#888888",
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Rate limiting for API requests (seconds between calls)
API_DELAY_SECONDS: float = 0.3

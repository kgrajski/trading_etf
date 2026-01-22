# -*- coding: utf-8 -*-
"""
Centralized configuration for workflow scripts.

All configurable parameters in one place for easy modification.
Individual scripts can override these as needed.
"""

from typing import Dict, List, Optional, Set

# =============================================================================
# DATA TIER CONFIGURATION
# =============================================================================

# Data tier: "iex" (free) or "sip" (paid)
# Determines data source and output directory structure
DATA_TIER: str = "iex"

# =============================================================================
# SYMBOL FILTERING
# =============================================================================

# Symbol prefix filter: Set to a prefix to limit symbols (e.g., "X" for testing)
# Set to None to process all symbols from the filtered universe
# NOTE: Macro symbols are NEVER filtered by prefix (always included)
SYMBOL_PREFIX_FILTER: Optional[str] = None  # None = all symbols

# =============================================================================
# DATE RANGE CONFIGURATION
# =============================================================================

# Historical date range for data fetching
# 5 years of history for coverage analysis
DATE_RANGE_START: Optional[str] = "2021-01-01"
DATE_RANGE_END: Optional[str] = None  # None = today

# Years of history to fetch if DATE_RANGE_START is None
YEARS_OF_HISTORY: int = 5

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
    "L1_log_return": True,  # Week-over-week close-to-close return
    "L1_log_return_intraweek": True,  # Intra-week open-to-close return
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

# =============================================================================
# MACRO SYMBOL CONFIGURATION
# =============================================================================
# These ETFs serve as proxies for futures markets not directly available via Alpaca.
# They are used as FEATURES only (not prediction targets).
#
# Categories:
#   - volatility: VIX futures proxies measuring market fear/uncertainty
#   - treasury: Interest rate sensitivity proxies for yield curve dynamics
#   - dollar: Currency proxy for dollar strength
#   - commodities: Cross-asset signals (safe-haven, energy)

MACRO_SYMBOLS = {
    "volatility": {
        # VIX futures proxies - measure market fear/uncertainty
        # VIXY tracks short-term VIX futures (1-2 month), sensitive to near-term vol
        "VIXY": "ProShares VIX Short-Term Futures ETF",
        # VIXM tracks mid-term VIX futures (4-7 month), less sensitive to daily moves
        "VIXM": "ProShares VIX Mid-Term Futures ETF",
    },
    "treasury": {
        # Interest rate sensitivity proxies - yield curve dynamics
        # Duration indicates sensitivity to rate changes (higher = more sensitive)
        "TLT": "iShares 20+ Year Treasury Bond ETF (long duration ~17yr)",
        "IEF": "iShares 7-10 Year Treasury Bond ETF (intermediate ~7.5yr)",
        "SHY": "iShares 1-3 Year Treasury Bond ETF (short duration ~2yr)",
        # BIL tracks ultra-short T-bills, nearly no rate sensitivity
        "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF (Fed Funds proxy)",
        # TIP tracks inflation-protected securities, reflects real rate expectations
        "TIP": "iShares TIPS Bond ETF (inflation-linked, real rate proxy)",
    },
    "dollar": {
        # Currency proxy - dollar strength indicator
        # UUP rises when dollar strengthens vs basket of major currencies
        "UUP": "Invesco DB US Dollar Index Bullish Fund",
    },
    "commodities": {
        # Cross-asset signals for risk sentiment and sector dynamics
        # GLD is classic safe-haven, rises during uncertainty/inflation fears
        "GLD": "SPDR Gold Shares (safe-haven demand proxy)",
        # USO tracks WTI crude oil, leading indicator for energy sector
        "USO": "United States Oil Fund (energy sector indicator)",
    },
}

# Flatten for iteration - list of all macro symbol tickers
MACRO_SYMBOL_LIST: List[str] = [
    sym for cat in MACRO_SYMBOLS.values() for sym in cat.keys()
]

# Map symbol to category for easy lookup
MACRO_SYMBOL_CATEGORIES: dict = {
    sym: cat for cat, symbols in MACRO_SYMBOLS.items() for sym in symbols.keys()
}

# =============================================================================
# SPECIALIZED MACRO FEATURES
# =============================================================================
# Cross-symbol derived features computed from macro symbols.
# Format: "feature_name": (symbol1, symbol2, operation)
# Operations: "ratio" = log(sym1/sym2), "spread" = sym1 - sym2

SPECIALIZED_MACRO_FEATURES = {
    # VIX term structure: VIXY/VIXM ratio indicates contango (>1) or backwardation (<1)
    # Contango = normal, futures > spot; Backwardation = fear, futures < spot
    "vix_term_structure": ("VIXY", "VIXM", "ratio"),
    # Yield curve slope: TLT/SHY ratio, falling = flattening/inversion (recession signal)
    "yield_curve_slope": ("TLT", "SHY", "ratio"),
    # Real rate proxy: TLT-TIP spread, higher = lower inflation expectations
    "real_rate_proxy": ("TLT", "TIP", "spread"),
}

# =============================================================================
# INCREMENTAL UPDATE CONFIGURATION
# =============================================================================

# Incremental mode: True = only fetch new data since last update
# False = full refresh from DATE_RANGE_START
INCREMENTAL_MODE: bool = True

# When in incremental mode, re-fetch last N days to catch any corrections
# or late-arriving data adjustments (e.g., adjusted close recalculations)
LOOKBACK_BUFFER_DAYS: int = 5

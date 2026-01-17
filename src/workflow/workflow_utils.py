# -*- coding: utf-8 -*-
"""
Common utilities for workflow scripts.

Provides:
- Project root path resolution
- Logging setup
- Workflow script decorator for consistent intro/outro logging
- Common path helpers
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

# =============================================================================
# PROJECT ROOT
# =============================================================================

# Calculate once at module load
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Ensure project root is in Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for workflow scripts.

    Args:
        level: Logging level (default: INFO)

    Returns:
        Configured logger for the calling module
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    return logging.getLogger(__name__)


# =============================================================================
# PATH HELPERS
# =============================================================================


def get_data_dir() -> Path:
    """Get the data directory path."""
    return PROJECT_ROOT / "data"


def get_metadata_dir() -> Path:
    """Get the metadata directory path."""
    return get_data_dir() / "metadata"


def get_historical_dir(tier: str) -> Path:
    """Get the historical data directory for a given tier.

    Args:
        tier: Data tier (e.g., 'iex', 'sip')

    Returns:
        Path to historical/{tier}/ directory
    """
    return get_data_dir() / "historical" / tier


def get_features_dir(tier: str) -> Path:
    """Get the features directory for a given tier.

    Args:
        tier: Data tier (e.g., 'iex', 'sip')

    Returns:
        Path to features/{tier}/ directory
    """
    return get_data_dir() / "features" / tier


def get_visualizations_dir(tier: str) -> Path:
    """Get the visualizations directory for a given tier.

    Args:
        tier: Data tier (e.g., 'iex', 'sip')

    Returns:
        Path to visualizations/{tier}/ directory
    """
    return get_data_dir() / "visualizations" / tier


# =============================================================================
# WORKFLOW SCRIPT DECORATOR
# =============================================================================


@contextmanager
def workflow_context(script_name: str):
    """Context manager for workflow script execution with timing and logging.

    Usage:
        with workflow_context("02-fetch-daily-data") as ctx:
            # Your workflow code here
            pass

    Args:
        script_name: Name of the workflow script (e.g., "02-fetch-daily-data")

    Yields:
        dict with 'start_time' key for duration calculations
    """
    # INTRO
    start_time = time.perf_counter()
    print(f"*** {script_name} - START ***")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    ctx = {"start_time": start_time}

    try:
        yield ctx
    finally:
        # OUTRO
        end_time = time.perf_counter()
        total_duration = end_time - start_time

        print()
        print("=" * 80)
        print(f"Total execution time: {total_duration:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"*** {script_name} - END ***")


def workflow_script(script_name: str) -> Callable:
    """Decorator for workflow script main functions.

    Wraps the main() function with consistent intro/outro logging.

    Usage:
        @workflow_script("02-fetch-daily-data")
        def main():
            # Your workflow code here
            pass

    Args:
        script_name: Name of the workflow script

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with workflow_context(script_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# SUMMARY PRINTING
# =============================================================================


def print_config(**kwargs) -> None:
    """Print configuration in a consistent format.

    Args:
        **kwargs: Key-value pairs to print
    """
    print("Configuration:")
    for key, value in kwargs.items():
        # Convert underscores to spaces and title case
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")
    print()


def print_summary(title: str = "SUMMARY", **kwargs) -> None:
    """Print summary in a consistent format.

    Args:
        title: Summary section title
        **kwargs: Key-value pairs to print
    """
    print("=" * 80)
    print(title)
    print("=" * 80)
    for key, value in kwargs.items():
        label = key.replace("_", " ").title()
        print(f"  {label}: {value}")
    print()

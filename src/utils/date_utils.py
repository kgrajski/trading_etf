# -*- coding: utf-8 -*-
"""Date utility functions for market data workflows.

This module provides helper functions for working with dates in the context
of market data collection, including gap detection and date range generation.

Author: kag
Created: 2025-12-01
"""

from datetime import datetime, timedelta
from typing import List, Set


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """Generate list of potential trading dates (excluding weekends).

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        List of date strings in YYYY-MM-DD format
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start

    while current <= end:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return dates


def get_week_start(date: str) -> str:
    """Get the Monday (week start) for a given date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Monday date string in YYYY-MM-DD format
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    # Get Monday of the week (weekday 0)
    days_since_monday = dt.weekday()
    monday = dt - timedelta(days=days_since_monday)
    return monday.strftime("%Y-%m-%d")


def get_week_end(date: str) -> str:
    """Get the Friday (week end) for a given date.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        Friday date string in YYYY-MM-DD format
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    # Get Friday of the week (weekday 4)
    days_since_monday = dt.weekday()
    friday = dt + timedelta(days=(4 - days_since_monday))
    return friday.strftime("%Y-%m-%d")


def partition_into_weeks(dates: List[str]) -> List[List[str]]:
    """Partition trading dates into weeks (Mon-Fri groups).

    Args:
        dates: Sorted list of date strings (YYYY-MM-DD)

    Returns:
        List of weeks, each week is a list of dates
    """
    weeks = []
    current_week = []
    current_week_key = None

    for date_str in dates:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        week_num = dt.isocalendar()[1]
        year = dt.isocalendar()[0]
        week_key = (year, week_num)

        if current_week_key is None:
            current_week_key = week_key

        if week_key != current_week_key:
            if current_week:
                weeks.append(current_week)
            current_week = [date_str]
            current_week_key = week_key
        else:
            current_week.append(date_str)

    if current_week:
        weeks.append(current_week)

    return weeks


def get_yesterday() -> str:
    """Get yesterday's date as a string.

    Returns:
        Date string in YYYY-MM-DD format
    """
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def is_weekend(date: str) -> bool:
    """Check if a date is a weekend.

    Args:
        date: Date string in YYYY-MM-DD format

    Returns:
        True if Saturday or Sunday
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    return dt.weekday() >= 5

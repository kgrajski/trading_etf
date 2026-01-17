# -*- coding: utf-8 -*-
"""ETF filtering and sponsor extraction utilities.

This module provides functions to identify ETFs and extract sponsor information
from asset names, focusing on the three major sponsors: iShares (BlackRock),
Vanguard, and State Street (SPDR).
"""

from typing import Optional

# Target sponsors for filtering
TARGET_SPONSORS = {
    "ishares": "iShares (BlackRock)",
    "vanguard": "Vanguard",
    "spdr": "State Street (SPDR)",
    "state street": "State Street (SPDR)",
}


def is_etf(asset_name: str) -> bool:
    """Check if an asset is likely an ETF based on its name.

    Args:
        asset_name: The asset name from Alpaca

    Returns:
        True if asset appears to be an ETF, False otherwise
    """
    if not asset_name:
        return False

    name_lower = asset_name.lower()

    # Positive indicators (ETF-related terms)
    etf_indicators = ["etf", "trust", "fund", "index fund"]

    # Negative indicators (likely not ETFs)
    non_etf_indicators = [
        "common stock",
        "class a",
        "class b",
        "class c",
        "preferred",
        "warrant",
        "right",
        "unit",
    ]

    # Check for negative indicators first
    for indicator in non_etf_indicators:
        if indicator in name_lower:
            return False

    # Check for positive indicators
    for indicator in etf_indicators:
        if indicator in name_lower:
            return True

    return False


def extract_sponsor(asset_name: str) -> Optional[str]:
    """Extract sponsor name from ETF asset name.

    Args:
        asset_name: The asset name from Alpaca

    Returns:
        Sponsor name if found, None otherwise
    """
    if not asset_name:
        return None

    name_lower = asset_name.lower()

    # Check for each target sponsor (order matters - check SPDR before State Street)
    if "spdr" in name_lower:
        return TARGET_SPONSORS["spdr"]
    elif "ishares" in name_lower:
        return TARGET_SPONSORS["ishares"]
    elif "vanguard" in name_lower:
        return TARGET_SPONSORS["vanguard"]
    elif "state street" in name_lower:
        return TARGET_SPONSORS["state street"]

    return None


def is_target_sponsor(sponsor: Optional[str]) -> bool:
    """Check if sponsor is one of our target sponsors.

    Args:
        sponsor: Sponsor name (from extract_sponsor)

    Returns:
        True if sponsor is iShares, Vanguard, or State Street
    """
    if not sponsor:
        return False

    sponsor_lower = sponsor.lower()
    target_values = [v.lower() for v in TARGET_SPONSORS.values()]

    return sponsor_lower in target_values

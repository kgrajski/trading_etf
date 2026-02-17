"""
Web search tool using Tavily API.

Searches financial news sources for ETF-specific analysis including
price drivers, sector dynamics, yield curve effects, regulatory actions,
and technical analysis.
"""
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

# Financial analysis sources — prioritize sites that explain WHY prices moved,
# not just that they moved. These sources provide the specific market mechanics
# (yield curve, NIM compression, technical levels, regulatory catalysts) that
# distinguish useful analysis from generic commentary.
FINANCIAL_SOURCES = [
    "seekingalpha.com",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "etf.com",
    "fool.com",
    "barrons.com",
    "finviz.com",
    "nasdaq.com",
    "investopedia.com",
    "zacks.com",
    "morningstar.com",
]


def search_news(query: str, max_results: int = 5, search_depth: str = "basic") -> List[Dict[str, Any]]:
    """
    Search for recent financial news and analysis.
    
    Args:
        query: Search query targeting specific market mechanics
        max_results: Maximum number of results
        search_depth: "basic" (fast) or "advanced" (deeper, more expensive)
        
    Returns:
        List of search results with title, url, content, score
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "TAVILY_API_KEY not set"}]
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_domains=FINANCIAL_SOURCES,
        )
        
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:800],
                "score": r.get("score", 0),
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e)}]


def search_market_overview(sector_hints: List[str] = None) -> List[Dict[str, Any]]:
    """
    Search for broad market context — macro events, sector rotation, and
    market-wide stories that affect multiple ETFs simultaneously.
    
    This catches events like FOMC decisions, yield curve shifts,
    technology disruptions, and geopolitical developments that
    per-symbol searches miss.
    
    Args:
        sector_hints: Optional sector terms extracted from candidate names
        
    Returns:
        List of market overview articles (deduplicated)
    """
    results = []
    seen_urls = set()
    
    # Query 1: Why did the market decline this week?
    q1 = "stock market decline this week major factors reasons"
    for r in search_news(q1, max_results=5, search_depth="advanced"):
        if "error" not in r and r.get("url") not in seen_urls:
            results.append(r)
            seen_urls.add(r.get("url"))
    
    # Query 2: Sector rotation and macro themes
    q2 = "sector rotation ETF flows this week rates economy market drivers"
    for r in search_news(q2, max_results=4):
        if "error" not in r and r.get("url") not in seen_urls:
            results.append(r)
            seen_urls.add(r.get("url"))
    
    # Query 3: Sector-specific if hints provided
    if sector_hints:
        top_sectors = " ".join(sector_hints[:3])
        q3 = f"{top_sectors} ETF sector decline analysis this week"
        for r in search_news(q3, max_results=3):
            if "error" not in r and r.get("url") not in seen_urls:
                results.append(r)
                seen_urls.add(r.get("url"))
    
    return results


def extract_sector_hints(candidates: List[Dict[str, Any]]) -> List[str]:
    """Extract dominant sector themes from candidate ETF names."""
    from collections import Counter
    
    sector_keywords = Counter()
    sector_map = {
        "financial": "financial", "bank": "banking", "insurance": "insurance",
        "broker": "broker", "transport": "transportation", "tech": "technology",
        "software": "software", "metal": "metals mining", "energy": "energy",
        "health": "healthcare", "china": "China", "europe": "European",
        "growth": "growth", "fintech": "fintech",
    }
    
    for c in candidates:
        name = c.get("etf_name", c.get("name", "")).lower()
        for pattern, sector in sector_map.items():
            if pattern in name:
                sector_keywords[sector] += 1
    
    return [sector for sector, _ in sector_keywords.most_common(4)]


def search_symbol_news(symbol: str, name: str = "") -> List[Dict[str, Any]]:
    """
    Search for news and analysis about a specific ETF symbol.
    
    Runs two targeted queries:
    1. Why the ETF declined — seeks specific market mechanics
    2. Sector/theme analysis — broader context for the move
    
    Args:
        symbol: ETF ticker symbol
        name: ETF name (optional, improves search quality)
        
    Returns:
        List of relevant news articles (deduplicated)
    """
    results = []
    seen_urls = set()
    
    # Query 1: Why did this ETF decline? (targets specific drivers)
    if name:
        q1 = f"why did {symbol} {name} ETF decline drop this week"
    else:
        q1 = f"why did {symbol} ETF decline drop this week"
    
    for r in search_news(q1, max_results=3):
        if "error" not in r and r.get("url") not in seen_urls:
            results.append(r)
            seen_urls.add(r.get("url"))
    
    # Query 2: Sector analysis and outlook
    sector_terms = _extract_sector_terms(name)
    q2 = f"{symbol} ETF {sector_terms} outlook analysis yield curve rates"
    
    for r in search_news(q2, max_results=3):
        if "error" not in r and r.get("url") not in seen_urls:
            results.append(r)
            seen_urls.add(r.get("url"))
    
    return results


def _extract_sector_terms(name: str) -> str:
    """Extract sector-relevant terms from ETF name for targeted search."""
    if not name:
        return "sector"
    
    name_lower = name.lower()
    
    # Map ETF name patterns to sector-specific search terms
    sector_map = {
        "financial": "financial sector banks interest rates yield curve",
        "bank": "banking sector net interest margin credit risk",
        "regional bank": "regional banks CRE commercial real estate",
        "insurance": "insurance sector underwriting premiums",
        "broker": "broker dealer capital markets trading volume",
        "transport": "transportation logistics freight demand economic growth",
        "tech": "technology sector earnings growth spending",
        "software": "software SaaS enterprise spending",
        "metals": "metals mining commodities prices",
        "energy": "energy oil prices OPEC production",
        "health": "healthcare medical devices FDA approval",
        "china": "China economy trade policy tariffs",
        "europe": "European markets ECB rates geopolitical",
        "growth": "growth stocks valuation rates duration",
        "value": "value stocks rotation",
    }
    
    for pattern, terms in sector_map.items():
        if pattern in name_lower:
            return terms
    
    return name.split()[0] + " sector"

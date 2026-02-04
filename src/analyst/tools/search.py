"""
Web search tool using Tavily API.
"""
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def search_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for recent news about a topic.
    
    Args:
        query: Search query (e.g., "SLV silver ETF news")
        max_results: Maximum number of results
        
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
            search_depth="basic",
            max_results=max_results,
            include_domains=["reuters.com", "bloomberg.com", "wsj.com", 
                           "cnbc.com", "marketwatch.com", "finance.yahoo.com",
                           "seekingalpha.com", "fool.com"],
        )
        
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],  # Truncate
                "score": r.get("score", 0),
            })
        
        return results
    
    except Exception as e:
        return [{"error": str(e)}]


def search_symbol_news(symbol: str, name: str = "") -> List[Dict[str, Any]]:
    """
    Search for news about a specific ETF symbol.
    
    Args:
        symbol: ETF ticker symbol
        name: ETF name (optional, improves search quality)
        
    Returns:
        List of relevant news articles
    """
    # Build a targeted query
    if name:
        # Extract key terms from name
        query = f"{symbol} ETF {name.split()[0]} news price"
    else:
        query = f"{symbol} ETF news price movement"
    
    return search_news(query, max_results=3)

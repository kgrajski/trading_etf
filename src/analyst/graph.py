"""
LangGraph-based ETF Trading Analyst
====================================

This module implements an **agentic workflow** for analyzing ETF trade candidates
using LangGraph. It demonstrates key concepts from agentic AI systems, including
the **Reflection Pattern** for self-review and quality improvement.

ARCHITECTURE OVERVIEW
---------------------

    ┌─────────────────────────────────────────────────────────────────┐
    │          SINGLE-AGENT SYSTEM with REFLECTION                    │
    │  (One agent with multiple nodes + self-review capability)       │
    └─────────────────────────────────────────────────────────────────┘

                            ┌──────────┐
                            │  START   │
                            └────┬─────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │     load       │  ← Initialize state
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  fetch_news    │  ← Tool use (Tavily API)
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ analyze_themes │  ← LLM call (thematic analysis)
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │analyze_symbols │  ← LLM calls (per-symbol)
                        └────────┬───────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   review_and_refine    │  ← REFLECTION: Self-review
                    │   • Check consistency  │
                    │   • Spot red flags     │
                    │   • Adjust convictions │
                    └────────────┬───────────┘
                                 │
                                 ▼
                            ┌──────────┐
                            │   END    │
                            └──────────┘


DESIGN PATTERNS USED
--------------------

This implementation combines two key patterns from DeepLearning.ai courses:

    Pattern         │ Description                      │ Used Here?
    ────────────────┼──────────────────────────────────┼───────────
    Sequential      │ Linear A → B → C flow            │ ✅ YES
    **Reflection**  │ Generate → Review → Refine       │ ✅ YES (NEW!)
    Parallel        │ Fan-out/fan-in execution         │ No
    Router          │ Conditional branching            │ No
    Supervisor      │ Orchestrator + worker agents     │ No
    Hierarchical    │ Nested agent teams               │ No

THE REFLECTION PATTERN
----------------------

The Reflection Pattern is a powerful technique where an agent reviews its own
output before finalizing. This mimics how human experts work:

    1. Junior Analyst produces initial analysis
    2. Senior Analyst reviews for errors, inconsistencies
    3. Adjustments are made based on review
    4. Final output includes both analysis AND review notes

In our implementation:
- `analyze_themes` + `analyze_symbols` = Initial analysis (Junior Analyst)
- `review_and_refine` = Self-review (Senior Analyst perspective)

Benefits:
- Catches logical inconsistencies
- Identifies red flags the initial analysis missed
- Adjusts overconfident or underconfident scores
- Provides audit trail of reasoning


MULTI-MODEL ROUTING (Model Heterogeneity)
-----------------------------------------

This module supports using DIFFERENT LLMs for different roles:

    Environment Variable        │ Role           │ Default
    ────────────────────────────┼────────────────┼─────────────────
    ANALYST_LLM_MODEL           │ Analysis nodes │ gemini-1.5-flash
    ANALYST_REVIEWER_MODEL      │ Review node    │ (same as analyst)

Benefits of multi-model routing:

1. **Cost Optimization**: Use cheap/fast model for bulk analysis,
   premium model only for critical review judgment.
   
   Example: gemini-2.0-flash ($0.10/1M) for analysis
            gpt-4o ($2.50/1M) for review = 66x cost savings on analysis!

2. **Diverse Perspectives**: Different models have different training
   and may catch different issues. Using a different reviewer model
   provides a "fresh eye" on the analysis.

3. **Specialization**: Some models excel at synthesis (Gemini Flash),
   others at critical reasoning (GPT-4o, Claude Sonnet).

Example configurations:

    # Single model (default - simple, consistent)
    ANALYST_LLM_MODEL=gemini-2.0-flash
    
    # Multi-model (cost optimized)
    ANALYST_LLM_MODEL=gemini-2.0-flash    # Fast, cheap for bulk
    ANALYST_REVIEWER_MODEL=gpt-4o          # Premium for review
    
    # Multi-model (diverse perspectives)
    ANALYST_LLM_MODEL=gemini-2.0-flash    # Google model
    ANALYST_REVIEWER_MODEL=claude-3-5-sonnet-20241022  # Anthropic model


KEY CONCEPTS (DeepLearning.ai Agentic AI)
-----------------------------------------

1. **State**: The `AnalystState` TypedDict defines what information flows
   through the graph. State is the "memory" that persists across nodes.

2. **Nodes**: Functions that transform state. Each node receives the full
   state, performs work, and returns updated state.

3. **Edges**: Define the flow between nodes. Can be:
   - Static edges (A always goes to B)
   - Conditional edges (A goes to B or C based on state)

4. **Tools**: External capabilities the agent can use. Here we use:
   - Tavily Search API (web search for news)
   - LLM (OpenAI/Anthropic/Gemini for analysis)

5. **Compiled Graph**: The workflow is "compiled" into an executable that
   manages state transitions automatically.


SINGLE AGENT vs MULTI-AGENT
---------------------------

This is a **SINGLE AGENT** system with multiple processing nodes.

    ┌─────────────────────────────────────────────────────────────────┐
    │ SINGLE AGENT (this module)    │ MULTI-AGENT (not implemented)   │
    ├───────────────────────────────┼─────────────────────────────────┤
    │ One graph, multiple nodes     │ Multiple graphs communicating   │
    │ Shared state across nodes     │ Each agent has own state        │
    │ Sequential/parallel nodes     │ Agents can delegate to others   │
    │ Simpler, easier to debug      │ More complex, more flexible     │
    │ Good for: focused workflows   │ Good for: complex reasoning     │
    └───────────────────────────────┴─────────────────────────────────┘

A multi-agent version might have:
- **Research Agent**: Gathers news and data
- **Analysis Agent**: Interprets data
- **Risk Agent**: Evaluates downside risks
- **Supervisor Agent**: Coordinates and synthesizes


EXTENDING THIS MODULE
---------------------

To add conditional branching:

    def should_continue(state: AnalystState) -> str:
        if state["news_cache"]:
            return "analyze_themes"
        return "skip_to_output"

    workflow.add_conditional_edges(
        "fetch_news",
        should_continue,
        {"analyze_themes": "analyze_themes", "skip_to_output": END}
    )

To add parallel execution (requires async):

    # Use langgraph's map-reduce for parallel symbol analysis


USAGE
-----

    from src.analyst.graph import analyze_candidates, export_graph_visualization

    # Run analysis
    results = analyze_candidates(candidates_list)

    # Export graph diagram
    export_graph_visualization("graph.png")


REFERENCES
----------

- LangGraph docs: https://python.langchain.com/docs/langgraph
- DeepLearning.ai "AI Agents in LangGraph": https://www.deeplearning.ai/short-courses/
- LangChain Expression Language: https://python.langchain.com/docs/expression_language/

"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.analyst.tools.search import search_symbol_news, search_market_overview, extract_sector_hints

load_dotenv()


# =============================================================================
# Usage Tracking
# =============================================================================

@dataclass
class UsageTracker:
    """Track LLM API usage and costs."""
    
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    
    # Per-call details
    call_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Pricing per 1M tokens (as of Jan 2025)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }
    
    def record(self, model: str, input_tokens: int, output_tokens: int, node: str = ""):
        """Record a single LLM call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.call_details.append({
            "node": node,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })
    
    def estimate_cost(self, model: str) -> float:
        """Estimate total cost based on model pricing."""
        pricing = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        input_cost = (self.input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def summary(self, model: str) -> str:
        """Generate usage summary."""
        cost = self.estimate_cost(model)
        return (
            f"\n{'='*60}\n"
            f"LLM USAGE SUMMARY\n"
            f"{'='*60}\n"
            f"  Model: {model}\n"
            f"  Total Calls: {self.calls}\n"
            f"  Input Tokens: {self.input_tokens:,}\n"
            f"  Output Tokens: {self.output_tokens:,}\n"
            f"  Total Tokens: {self.input_tokens + self.output_tokens:,}\n"
            f"  Estimated Cost: ${cost:.4f}\n"
            f"{'='*60}\n"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON output."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "call_details": self.call_details,
        }


# Global tracker instance
_usage_tracker = UsageTracker()


def get_usage_tracker() -> UsageTracker:
    """Get the global usage tracker."""
    return _usage_tracker


def reset_usage_tracker():
    """Reset the global usage tracker."""
    global _usage_tracker
    _usage_tracker = UsageTracker()


# =============================================================================
# State Definition
# =============================================================================

class AnalystState(TypedDict):
    """
    State that flows through the analyst graph.
    
    This TypedDict defines the "memory" of our agent - all information
    that persists across nodes. Each node can read from and write to
    this state.
    """
    # Input
    candidates: List[Dict[str, Any]]
    
    # Intermediate
    market_context: List[Dict[str, Any]]  # Broad market overview articles
    news_cache: Dict[str, List[Dict[str, Any]]]  # symbol -> news articles
    
    # Output - Initial Analysis
    thematic_analysis: Optional[Dict[str, Any]]
    symbol_analyses: Dict[str, Dict[str, Any]]  # symbol -> analysis
    
    # Output - Reflection/Review
    review_results: Optional[Dict[str, Any]]  # Reviewer's critique and adjustments
    
    # Metadata
    errors: List[str]


# =============================================================================
# LLM Setup
# =============================================================================

# Model configuration - supports multi-model routing
# 
# ANALYST_LLM_MODEL: Used for analysis nodes (themes, symbols)
#   - Can be cheaper/faster model for bulk work
#   - Default: gemini-2.0-flash (fast, cheap)
#
# ANALYST_REVIEWER_MODEL: Used for review/reflection node
#   - Can be premium model for critical judgment
#   - Default: same as ANALYST_LLM_MODEL (single model)
#   - Set to different model for heterogeneous routing
#
# Examples:
#   Single model:  ANALYST_LLM_MODEL=gemini-1.5-flash
#   Multi-model:   ANALYST_LLM_MODEL=gemini-1.5-flash
#                  ANALYST_REVIEWER_MODEL=gpt-4o
#
LLM_MODEL = os.getenv("ANALYST_LLM_MODEL", "gemini-2.0-flash")
REVIEWER_MODEL = os.getenv("ANALYST_REVIEWER_MODEL", "")  # Empty = use LLM_MODEL


def get_model_for_role(role: str = "analyst") -> str:
    """Get the model name for a specific role."""
    if role == "reviewer" and REVIEWER_MODEL:
        return REVIEWER_MODEL
    return LLM_MODEL


def _create_llm(model: str):
    """Create an LLM instance for the given model name."""
    # OpenAI models
    if model.startswith("gpt-"):
        return ChatOpenAI(
            model=model,
            temperature=0.3,
        )
    
    # Anthropic models (requires langchain-anthropic)
    elif model.startswith("claude-"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                temperature=0.3,
            )
        except ImportError:
            print("  WARNING: langchain-anthropic not installed, falling back to OpenAI")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Google models (requires langchain-google-genai)
    elif model.startswith("gemini-"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=0.3,
            )
        except ImportError:
            print("  WARNING: langchain-google-genai not installed, falling back to OpenAI")
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # Default to OpenAI
    else:
        return ChatOpenAI(model=model, temperature=0.3)


def get_llm(role: str = "analyst"):
    """
    Get the configured LLM for a specific role.
    
    Args:
        role: Either "analyst" (themes/symbols) or "reviewer" (reflection)
              Different roles can use different models for cost optimization
              or diverse perspectives.
    
    Returns:
        A LangChain chat model instance
    """
    model = get_model_for_role(role)
    return _create_llm(model)


def invoke_llm_with_tracking(llm, prompt: str, node: str = "", role: str = "analyst") -> str:
    """
    Invoke LLM and track usage with full logging and instrumentation.
    
    This function:
    1. Calls the LLM with the prompt (timed)
    2. Extracts token usage metadata
    3. Records usage in the tracker
    4. Records detailed metrics in instrumenter
    5. Logs the full prompt/response for traceability
    
    Args:
        llm: The LangChain LLM instance
        prompt: The prompt to send
        node: Name of the node making the call (for logging)
        role: The role using this LLM ("analyst" or "reviewer")
        
    Returns:
        str: The LLM response content
    """
    import time
    
    # Get the model name for this role (for tracking)
    model_name = get_model_for_role(role)
    
    # Time the LLM call
    start_time = time.time()
    response = llm.invoke(prompt)
    duration_ms = (time.time() - start_time) * 1000
    
    # Extract usage metadata if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
    else:
        # Estimate if not available (roughly 4 chars per token)
        input_tokens = len(prompt) // 4
        output_tokens = len(response.content) // 4
    
    # Record in usage tracker (for backward compatibility)
    tracker = get_usage_tracker()
    tracker.record(model_name, input_tokens, output_tokens, node)
    
    # Record in instrumenter (detailed metrics)
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        inst.record_llm_call(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            node=node,
            prompt_chars=len(prompt),
            response_chars=len(response.content),
        )
    except ImportError:
        pass
    
    # Log for traceability
    try:
        from src.analyst.logging_config import get_logger
        logger = get_logger()
        if logger:
            logger.log_llm_call(
                node=node,
                prompt=prompt,
                response=response.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model_name,
            )
    except ImportError:
        pass  # Logging not available
    
    return response.content


# =============================================================================
# Graph Nodes
# =============================================================================

def load_candidates(state: AnalystState) -> AnalystState:
    """Initialize state - candidates should already be loaded."""
    # Track node execution
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        with inst.track_node("load"):
            if not state.get("market_context"):
                state["market_context"] = []
            if not state.get("news_cache"):
                state["news_cache"] = {}
            if not state.get("symbol_analyses"):
                state["symbol_analyses"] = {}
            if not state.get("errors"):
                state["errors"] = []
    except ImportError:
        if not state.get("market_context"):
            state["market_context"] = []
        if not state.get("news_cache"):
            state["news_cache"] = {}
        if not state.get("symbol_analyses"):
            state["symbol_analyses"] = {}
        if not state.get("errors"):
            state["errors"] = []
    return state


def fetch_news(state: AnalystState) -> AnalystState:
    """Fetch news for all candidate symbols."""
    import time
    
    # Track node execution
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        node_context = inst.track_node("fetch_news")
        node_context.__enter__()
    except ImportError:
        inst = None
        node_context = None
    
    # Step 1: Broad market overview (catches macro events per-symbol searches miss)
    print("  Fetching market overview...")
    import time as _time
    
    sector_hints = extract_sector_hints(state["candidates"])
    overview_start = _time.time()
    state["market_context"] = search_market_overview(sector_hints)
    overview_ms = (_time.time() - overview_start) * 1000
    
    print(f"    Market overview: {len(state['market_context'])} articles ({overview_ms:.0f}ms)")
    if inst:
        inst.record_tool_call(
            tool="tavily_market_overview",
            duration_ms=overview_ms,
            success=len(state["market_context"]) > 0,
            result_size=len(state["market_context"]),
            node="fetch_news",
        )
    
    # Step 2: Per-symbol news
    print("  Fetching news for symbols...")
    
    for candidate in state["candidates"]:
        symbol = candidate.get("symbol", "")
        name = candidate.get("etf_name", "")
        
        if symbol and symbol not in state["news_cache"]:
            print(f"    Searching news for {symbol}...")
            
            # Time the tool call
            start_time = time.time()
            news = search_symbol_news(symbol, name)
            duration_ms = (time.time() - start_time) * 1000
            
            state["news_cache"][symbol] = news
            
            # Record tool call in instrumenter
            if inst:
                success = not (news and len(news) == 1 and "error" in news[0])
                inst.record_tool_call(
                    tool="tavily_search",
                    duration_ms=duration_ms,
                    success=success,
                    result_size=len(news) if success else 0,
                    node="fetch_news",
                )
    
    print(f"  Fetched news for {len(state['news_cache'])} symbols")
    
    if node_context:
        node_context.__exit__(None, None, None)
    
    return state


def analyze_themes(state: AnalystState) -> AnalystState:
    """Identify thematic patterns across candidates using the premium model."""
    # Track node execution
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        node_context = inst.track_node("analyze_themes")
        node_context.__enter__()
    except ImportError:
        inst = None
        node_context = None
    
    print("  Analyzing themes...")
    
    # Use reviewer (premium) model for synthesis — this is where reasoning quality matters most
    llm = get_llm("reviewer")
    
    # Build context
    candidates_summary = []
    for c in state["candidates"]:
        symbol = c.get("symbol", "")
        name = c.get("etf_name", "")
        ret = c.get("pct_return", 0)
        candidates_summary.append(f"- {symbol}: {name} (return: {ret:.1f}%)")
    
    # Market overview context (broad macro stories)
    market_context_text = []
    for article in state.get("market_context", []):
        if "error" not in article:
            title = article.get('title', '')
            content = article.get('content', '')[:400]
            market_context_text.append(f"• {title}\n  {content}")
    
    # Per-symbol news (feed ALL to theme analysis)
    news_summary = []
    for symbol, articles in state["news_cache"].items():
        for article in articles:
            if "error" not in article:
                title = article.get('title', '')
                content = article.get('content', '')[:300]
                news_summary.append(f"[{symbol}] {title}\n  {content}")
    
    prompt = f"""You are a SENIOR FINANCIAL ANALYST writing a market intelligence brief for an ETF trading desk. Your analysis must be specific, evidence-based, and cite concrete market mechanisms — not generic commentary.

CONTEXT: These ETFs dropped significantly this week and are mean-reversion candidates (our quant system targets a 1-3 week bounce). Your job: explain WHY they dropped using specific market mechanics, and assess whether those drivers are transient or structural.

CANDIDATES:
{chr(10).join(candidates_summary)}

=== BROAD MARKET CONTEXT (this week's major market-moving events) ===
{chr(10).join(market_context_text) if market_context_text else "No broad market context available"}

=== PER-SYMBOL NEWS ===
{chr(10).join(news_summary) if news_summary else "No per-symbol news available"}

ANALYTICAL FRAMEWORK — For each theme, evaluate relevance of these dimensions:

1. TECHNICAL FACTORS: Did the sector break below key moving averages (50-day, 200-day)? Support levels breached?
2. RATE / YIELD CURVE: Is a flattening yield curve compressing bank NIMs? Are rate expectations shifting?
3. CREDIT / FUNDAMENTALS: Emerging credit concerns, CRE exposure, delinquencies, earnings misses?
4. REGULATORY / LEGAL: Specific lawsuits, enforcement actions, new regulations?
5. MACRO / GEOPOLITICAL: Fed policy signals, inflation data, trade tensions, geopolitical events?
6. SECTOR ROTATION: Money flowing out of this sector — into what, and why?
7. MARKET-WIDE EVENTS: Major technology shifts, pandemic effects, or other cross-sector shocks?

For each theme, cite which dimensions are relevant and provide evidence from the news above.
If a dimension is NOT relevant, do not mention it.
If the news does NOT support a claim, do not fabricate one — say evidence is limited.

Respond in JSON format:
{{
    "themes": [
        {{
            "name": "Descriptive Theme Name",
            "symbols": ["SYM1", "SYM2"],
            "news_driver": "Specific market mechanism with evidence from news (e.g., 'Visa antitrust lawsuit triggered $1.2B settlement concerns across financial services')",
            "drop_type": "transient/structural/unclear",
            "relevant_dimensions": ["technical", "rates", "regulatory"],
            "narrative": "3-4 sentences: what specific mechanism drove these drops, why it matters, and whether the evidence suggests recovery in 1-3 weeks"
        }}
    ],
    "upcoming_catalysts": ["Specific dated catalyst (e.g., 'FOMC meeting March 18-19')", "catalyst2"],
    "overall_news_assessment": "favorable/mixed/unfavorable",
    "summary": "3-4 sentence executive summary. A reader should learn specific market conditions, not 'mixed signals across sectors.'"
}}
"""

    try:
        content = invoke_llm_with_tracking(llm, prompt, node="analyze_themes", role="reviewer")
        
        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        state["thematic_analysis"] = json.loads(content)
        print("  Thematic analysis complete")
        
    except Exception as e:
        state["errors"].append(f"Theme analysis error: {str(e)}")
        state["thematic_analysis"] = {
            "themes": [],
            "overall_sentiment": "unknown",
            "summary": f"Analysis failed: {str(e)}"
        }
    
    if node_context:
        node_context.__exit__(None, None, None)
    
    return state


def analyze_symbols(state: AnalystState) -> AnalystState:
    """Generate symbol-specific analysis with conviction scores."""
    # Track node execution
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        node_context = inst.track_node("analyze_symbols")
        node_context.__enter__()
    except ImportError:
        inst = None
        node_context = None
    
    print("  Analyzing individual symbols...")
    
    llm = get_llm()
    
    for candidate in state["candidates"]:
        symbol = candidate.get("symbol", "")
        name = candidate.get("etf_name", "")
        ret = candidate.get("pct_return", 0)
        sigma = candidate.get("sigma", None)
        beta = candidate.get("beta", None)
        
        # Build market context summary for injection
        market_ctx = state.get("market_context", [])
        market_ctx_text = ""
        if market_ctx:
            market_lines = []
            for article in market_ctx[:5]:
                if "error" not in article:
                    market_lines.append(f"• {article.get('title', '')}: {article.get('content', '')[:200]}")
            market_ctx_text = chr(10).join(market_lines)
        
        # Get news for this symbol (include content, not just titles)
        news = state["news_cache"].get(symbol, [])
        news_text = ""
        citations = []
        for article in news:
            if "error" not in article:
                title = article.get('title', '')
                content = article.get('content', '')[:400]
                news_text += f"- {title}\n  {content}\n\n"
                citations.append({
                    "title": title,
                    "url": article.get("url", "")
                })
        
        prompt = f"""You are a FINANCIAL ANALYST assessing whether a specific ETF's decline will reverse within 1-3 weeks.

SYMBOL: {symbol}
NAME: {name}
WEEKLY RETURN: {ret:.2f}%

=== BROAD MARKET CONTEXT THIS WEEK ===
{market_ctx_text if market_ctx_text else "No broad market context available"}

=== NEWS SPECIFIC TO {symbol} ===
{news_text if news_text else "No symbol-specific news found."}

IMPORTANT RULES:
- ONLY cite information that is actually in the news above. Do NOT fabricate claims.
- If the news above does not explain this ETF's decline, say so honestly and set evidence_quality to "weak" or "none".
- Check that the news is actually ABOUT this ETF or its sector — ignore unrelated articles.
- Use the broad market context to identify macro factors (technology shifts, rate changes, sector rotation) that may have affected this ETF even if not mentioned in symbol-specific news.

STRUCTURED ANALYSIS — evaluate each dimension's relevance to this ETF's decline:
1. TECHNICAL: Break below key MAs, support levels, technical patterns?
2. RATES / YIELD CURVE: Rate environment impact on this sector (NIM compression, duration risk)?
3. CREDIT / FUNDAMENTALS: Earnings, credit quality, balance sheet concerns?
4. REGULATORY / LEGAL: Lawsuits, enforcement, policy changes affecting this sector?
5. MACRO / GEOPOLITICAL: Fed signals, trade tensions, geopolitical events?
6. SECTOR ROTATION: Flows out of this sector, into what?
7. MARKET-WIDE EVENTS: Major technology shifts, disruptive innovations, or cross-sector shocks?

Respond in JSON format:
{{
    "flag": "GREEN/YELLOW/RED",
    "flag_reason": "One specific sentence citing evidence from the news above",
    "news_summary": "2-3 sentences on the specific mechanism(s) driving this drop, citing only what the news actually says",
    "drop_assessment": "transient/structural/unclear",
    "relevant_dimensions": ["List which of the 7 dimensions above are relevant, e.g. 'rates', 'regulatory'"],
    "bullish_signals": ["Specific evidence from news suggesting recovery"],
    "bearish_signals": ["Specific evidence from news suggesting continued decline"],
    "key_concern": "Single most important risk factor from the evidence — or 'none' if GREEN",
    "evidence_quality": "strong/moderate/weak/none"
}}

FLAG GUIDELINES:
- GREEN: Evidence points to transient drop. No structural headwinds. evidence_quality should be moderate or strong.
- YELLOW: Mixed or insufficient evidence. Some concerns but recovery plausible.
- RED: Evidence of structural headwind. Bounce unlikely in 1-3 weeks.
- If evidence_quality is "none", flag MUST be YELLOW (we don't have enough information to judge).
"""
        
        try:
            content = invoke_llm_with_tracking(llm, prompt, node=f"analyze_{symbol}")
            
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            analysis = json.loads(content)
            analysis["citations"] = citations
            state["symbol_analyses"][symbol] = analysis
            
            flag = analysis.get('flag', '?')
            print(f"    {symbol}: {flag} - {analysis.get('flag_reason', '')[:50]}")
            
        except Exception as e:
            state["errors"].append(f"Symbol {symbol} analysis error: {str(e)}")
            state["symbol_analyses"][symbol] = {
                "flag": "YELLOW",
                "flag_reason": "Analysis failed",
                "news_summary": f"Error: {str(e)}",
                "drop_assessment": "unclear",
                "bullish_signals": [],
                "bearish_signals": ["Analysis failed"],
                "key_concern": "Unable to assess news",
                "citations": citations
            }
    
    print(f"  Analyzed {len(state['symbol_analyses'])} symbols")
    
    if node_context:
        node_context.__exit__(None, None, None)
    
    return state


def review_and_refine(state: AnalystState) -> AnalystState:
    """
    REFLECTION NODE: Review and refine the analysis.
    
    This implements the Reflection Pattern from agentic AI:
    1. Review thematic analysis for logical consistency
    2. Review symbol recommendations for red flags
    3. Identify potential errors or overconfidence
    4. Suggest conviction adjustments
    
    The reviewer acts as a "senior analyst" checking the junior analyst's work.
    
    MODEL ROUTING: This node uses ANALYST_REVIEWER_MODEL if configured,
    allowing a different (potentially premium) model to perform critical review.
    This enables cost optimization (cheap model for bulk, premium for review)
    or diverse perspectives (different models catch different issues).
    """
    # Track node execution
    try:
        from src.analyst.instrumentation import get_instrumenter
        inst = get_instrumenter()
        node_context = inst.track_node("review_and_refine")
        node_context.__enter__()
    except ImportError:
        inst = None
        node_context = None
    
    # Use reviewer model (may be different from analyst model)
    reviewer_model = get_model_for_role("reviewer")
    print(f"  Reviewing and refining analysis (Reflection Pattern)...")
    print(f"    Reviewer model: {reviewer_model}")
    
    llm = get_llm("reviewer")
    
    # Build summary of all analyses for review
    thematic = state.get("thematic_analysis", {})
    symbol_analyses = state.get("symbol_analyses", {})
    
    # Format thematic summary
    themes_text = ""
    for theme in thematic.get("themes", []):
        themes_text += f"- {theme.get('name')}: {', '.join(theme.get('symbols', []))}\n"
        themes_text += f"  News Driver: {theme.get('news_driver', theme.get('narrative', 'N/A'))}\n"
        themes_text += f"  Drop Type: {theme.get('drop_type', theme.get('mean_reversion_outlook', 'N/A'))}\n"
    
    # Format symbol analyses summary
    symbols_text = ""
    for symbol, analysis in symbol_analyses.items():
        flag = analysis.get('flag', analysis.get('recommendation', '?'))
        symbols_text += f"- {symbol}: {flag}\n"
        symbols_text += f"  Reason: {analysis.get('flag_reason', analysis.get('narrative', 'N/A'))}\n"
        symbols_text += f"  Drop Assessment: {analysis.get('drop_assessment', 'N/A')}\n"
        symbols_text += f"  Key Concern: {analysis.get('key_concern', analysis.get('key_risk', 'N/A'))}\n"
    
    # Get news headlines for context
    news_context = []
    for symbol, articles in list(state.get("news_cache", {}).items())[:10]:
        for article in articles[:1]:
            if "error" not in article:
                news_context.append(f"[{symbol}] {article.get('title', '')}")
    
    prompt = f"""You are a SENIOR NEWS ANALYST reviewing a junior analyst's qualitative assessments.

IMPORTANT: Your review is QUALITATIVE ONLY. Do not second-guess the quantitative trade selection — that's handled by our quant system. Focus on whether the NEWS interpretation is sound.

=== THEMATIC ANALYSIS TO REVIEW ===
News Assessment: {thematic.get('overall_news_assessment', thematic.get('overall_sentiment', 'unknown'))}
Summary: {thematic.get('summary', 'N/A')}

Themes:
{themes_text if themes_text else "No themes identified"}

=== SYMBOL ASSESSMENTS TO REVIEW ===
{symbols_text if symbols_text else "No symbols analyzed"}

=== NEWS HEADLINES FOR REFERENCE ===
{chr(10).join(news_context) if news_context else "No news available"}

=== YOUR REVIEW TASK ===

Review the junior analyst's NEWS interpretations:
1. Are the flag assignments (GREEN/YELLOW/RED) justified by the news?
2. Did they miss any important news signals?
3. Are they being appropriately cautious about structural vs transient drops?
4. Which symbols have the clearest qualitative cases (best/worst)?

Respond in JSON format:
{{
    "overall_assessment": "APPROVE/NEEDS_REVISION",
    "quality_score": 1-10,
    "flag_adjustments": [
        {{
            "symbol": "XYZ",
            "original_flag": "GREEN",
            "adjusted_flag": "YELLOW",
            "reason": "Missed the regulatory news that suggests ongoing pressure"
        }}
    ],
    "missed_signals": ["Any news the junior analyst should have caught"],
    "strongest_candidates": ["SYM1", "SYM2"],
    "weakest_candidates": ["SYM3"],
    "reviewer_notes": "2-3 sentence summary of news landscape and key concerns"
}}

GUIDELINES:
- Only adjust flags if the NEWS clearly supports a different assessment
- "strongest_candidates" = GREEN flags with clearest transient drop signals
- "weakest_candidates" = RED flags or structural concerns
- Do NOT comment on quantitative factors (that's not your domain)
"""

    try:
        content = invoke_llm_with_tracking(llm, prompt, node="review_and_refine", role="reviewer")
        
        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        review = json.loads(content)
        state["review_results"] = review
        
        # Apply flag adjustments to symbol analyses
        adjustments = review.get("flag_adjustments", [])
        for adj in adjustments:
            symbol = adj.get("symbol")
            if symbol and symbol in state["symbol_analyses"]:
                original = state["symbol_analyses"][symbol].get("flag", "YELLOW")
                adjusted = adj.get("adjusted_flag", original)
                if original != adjusted:
                    state["symbol_analyses"][symbol]["flag_original"] = original
                    state["symbol_analyses"][symbol]["flag"] = adjusted
                    state["symbol_analyses"][symbol]["flag_adjusted_by_reviewer"] = True
                    state["symbol_analyses"][symbol]["adjustment_reason"] = adj.get("reason", "")
                    print(f"    {symbol}: flag adjusted {original} → {adjusted}")
        
        print(f"  Review complete: {review.get('overall_assessment', 'N/A')}")
        strongest = review.get('strongest_candidates', review.get('top_picks', []))
        weakest = review.get('weakest_candidates', review.get('avoid_list', []))
        if strongest:
            print(f"  Strongest (GREEN): {', '.join(strongest)}")
        if weakest:
            print(f"  Weakest (concerns): {', '.join(weakest)}")
        
    except Exception as e:
        state["errors"].append(f"Review error: {str(e)}")
        state["review_results"] = {
            "overall_assessment": "ERROR",
            "quality_score": 0,
            "reviewer_notes": f"Review failed: {str(e)}",
            "missed_signals": [],
            "strongest_candidates": [],
            "weakest_candidates": [],
        }
    
    if node_context:
        node_context.__exit__(None, None, None)
    
    return state


# =============================================================================
# Graph Definition
# =============================================================================

def build_analyst_graph():
    """
    Build the analyst workflow graph with Reflection Pattern.
    
    This creates a LangGraph StateGraph with the following structure:
    
        START → load → fetch_news → analyze_themes → analyze_symbols 
              → review_and_refine → END
    
    The review_and_refine node implements the REFLECTION PATTERN:
    - Reviews the initial analysis for consistency
    - Identifies red flags and potential errors
    - Adjusts conviction scores where warranted
    
    The graph uses a sequential (chain) pattern where each node processes
    the state and passes it to the next node. This is the simplest agentic
    pattern and is ideal for linear workflows.
    
    Returns:
        CompiledGraph: A compiled graph ready for execution via .invoke()
    """
    # Create graph with our state schema
    # The StateGraph manages state transitions automatically
    workflow = StateGraph(AnalystState)
    
    # Add nodes - each node is a function that transforms state
    # Node names are used for edges and debugging
    workflow.add_node("load", load_candidates)              # Initialize state
    workflow.add_node("fetch_news", fetch_news)             # Tool: web search
    workflow.add_node("analyze_themes", analyze_themes)     # LLM: thematic analysis
    workflow.add_node("analyze_symbols", analyze_symbols)   # LLM: per-symbol analysis
    workflow.add_node("review_and_refine", review_and_refine)  # LLM: REFLECTION NODE
    
    # Define edges (sequential flow with reflection)
    workflow.set_entry_point("load")
    workflow.add_edge("load", "fetch_news")
    workflow.add_edge("fetch_news", "analyze_themes")
    workflow.add_edge("analyze_themes", "analyze_symbols")
    workflow.add_edge("analyze_symbols", "review_and_refine")  # NEW: Add reflection
    workflow.add_edge("review_and_refine", END)
    
    # Compile the graph - this creates an executable workflow
    # The compiled graph handles state management automatically
    return workflow.compile()


def get_graph_for_visualization():
    """
    Get the uncompiled graph for visualization purposes.
    
    Returns the StateGraph before compilation so we can extract
    the structure for visualization.
    """
    workflow = StateGraph(AnalystState)
    workflow.add_node("load", load_candidates)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("analyze_themes", analyze_themes)
    workflow.add_node("analyze_symbols", analyze_symbols)
    workflow.add_node("review_and_refine", review_and_refine)  # REFLECTION NODE
    workflow.set_entry_point("load")
    workflow.add_edge("load", "fetch_news")
    workflow.add_edge("fetch_news", "analyze_themes")
    workflow.add_edge("analyze_themes", "analyze_symbols")
    workflow.add_edge("analyze_symbols", "review_and_refine")
    workflow.add_edge("review_and_refine", END)
    return workflow


def export_graph_visualization(
    output_path: str = "analyst_graph.png",
    include_mermaid: bool = True,
) -> Dict[str, str]:
    """
    Export the graph structure as PNG and Mermaid diagram.
    
    This generates visual representations of the directed graph,
    showing nodes and edges. The PNG uses the mermaid.ink service
    (requires network access). Also saves the raw Mermaid code.
    
    Args:
        output_path: Path for the output PNG file
        include_mermaid: Also save Mermaid markdown file
        
    Returns:
        Dict with paths to generated files:
        - "png": Path to PNG file (or None if failed)
        - "mermaid": Path to Mermaid markdown file
        - "ascii": ASCII representation
        
    Example:
        >>> from src.analyst.graph import export_graph_visualization
        >>> paths = export_graph_visualization("my_graph.png")
        >>> print(paths["png"])
        'my_graph.png'
    """
    graph = build_analyst_graph()
    results = {"png": None, "mermaid": None, "ascii": None}
    
    # Get mermaid diagram (always works)
    try:
        mermaid = graph.get_graph().draw_mermaid()
        mermaid_path = output_path.replace(".png", "_mermaid.md")
        with open(mermaid_path, "w") as f:
            f.write("# ETF Analyst Graph\n\n")
            f.write("This is a **Sequential Chain with Reflection** pattern.\n\n")
            f.write("## Graph Visualization\n\n")
            f.write("```mermaid\n")
            f.write(mermaid)
            f.write("\n```\n\n")
            f.write("## Node Descriptions\n\n")
            f.write("| Node | Purpose | Type |\n")
            f.write("|------|---------|------|\n")
            f.write("| `load` | Initialize state, validate inputs | Setup |\n")
            f.write("| `fetch_news` | Search web for ETF news | Tool (Tavily) |\n")
            f.write("| `analyze_themes` | Identify market themes | LLM Call |\n")
            f.write("| `analyze_symbols` | Score each candidate | LLM Calls |\n")
            f.write("| `review_and_refine` | **REFLECTION**: Review & adjust | LLM Call |\n")
        print(f"Mermaid diagram saved: {mermaid_path}")
        results["mermaid"] = mermaid_path
    except Exception as e:
        print(f"Mermaid export failed: {e}")
    
    # Try PNG export (requires network for mermaid.ink)
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"PNG graph saved: {output_path} ({len(png_data):,} bytes)")
        results["png"] = output_path
    except Exception as e:
        print(f"PNG export failed (may need network access): {e}")
    
    # ASCII fallback (always include)
    ascii_graph = """
┌─────────────────────────────────────────────────────────────┐
│     ETF ANALYST GRAPH (Sequential + Reflection Pattern)     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌──────────┐                                             │
│    │  START   │                                             │
│    └────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐                                             │
│    │   load   │  Initialize state                           │
│    └────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│    ┌──────────┐                                             │
│    │fetch_news│  Tool: Tavily web search                    │
│    └────┬─────┘                                             │
│         │                                                   │
│         ▼                                                   │
│    ┌───────────────┐                                        │
│    │analyze_themes │  LLM: Thematic analysis                │
│    └────┬──────────┘                                        │
│         │                                                   │
│         ▼                                                   │
│    ┌────────────────┐                                       │
│    │analyze_symbols │  LLM: Per-symbol conviction           │
│    └────┬───────────┘                                       │
│         │                                                   │
│         ▼                                                   │
│  ┌───────────────────┐                                      │
│  │ review_and_refine │  REFLECTION: Senior analyst review   │
│  │  • Check logic    │                                      │
│  │  • Spot red flags │                                      │
│  │  • Adjust scores  │                                      │
│  └────────┬──────────┘                                      │
│           │                                                 │
│           ▼                                                 │
│      ┌──────────┐                                           │
│      │   END    │                                           │
│      └──────────┘                                           │
│                                                             │
│  Pattern: Sequential Chain + Reflection                     │
│  Agent Type: Single Agent with Self-Review                  │
│  LLM Calls: 1 + N + 1 (themes + symbols + review)           │
└─────────────────────────────────────────────────────────────┘
"""
    results["ascii"] = ascii_graph
    
    return results


def print_graph_info():
    """
    Print information about the graph structure.
    
    Useful for debugging and understanding the workflow.
    """
    print("\n" + "=" * 60)
    print("ETF ANALYST GRAPH STRUCTURE")
    print("=" * 60)
    print("""
    Design Pattern: Sequential Chain + REFLECTION
    Agent Type:     Single Agent with Self-Review
    
    Nodes:
    ------
    1. load             - Initialize state, validate inputs
    2. fetch_news       - Web search via Tavily API (Tool)
    3. analyze_themes   - Thematic analysis via LLM
    4. analyze_symbols  - Per-symbol conviction scoring via LLM
    5. review_and_refine- REFLECTION: Senior analyst review (LLM)
    
    Edges:
    ------
    START → load → fetch_news → analyze_themes → analyze_symbols 
          → review_and_refine → END
    
    State Schema:
    -------------
    - candidates: List[Dict]           # Input ETF candidates
    - news_cache: Dict[str, List]      # Symbol → news articles
    - thematic_analysis: Dict          # Market themes output
    - symbol_analyses: Dict[str, Dict] # Symbol → conviction scores
    - review_results: Dict             # Reviewer's critique (NEW)
    - errors: List[str]                # Any errors encountered
    """)
    print("=" * 60)


# =============================================================================
# Main Interface
# =============================================================================

def analyze_candidates(
    candidates: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run the analyst on a list of trade candidates.
    
    Args:
        candidates: List of candidate dicts with keys:
            - symbol, etf_name, pct_return, limit_price, etc.
        output_dir: Optional directory to save logs and reports
            
    Returns:
        Dict with:
        - thematic_analysis: Dict of market themes
        - symbol_analyses: Dict of per-symbol conviction scores
        - errors: List of any errors encountered
        - usage: Token usage statistics
        - model: LLM model used
        - log_files: Paths to generated log files (if output_dir provided)
    """
    # Reset usage tracker for fresh run
    reset_usage_tracker()
    
    # Initialize instrumenter for detailed metrics
    instrumenter = None
    try:
        from src.analyst.instrumentation import get_instrumenter, reset_instrumenter
        reset_instrumenter()
        instrumenter = get_instrumenter()
        instrumenter.start_run(model=LLM_MODEL, candidates_count=len(candidates))
    except ImportError:
        pass
    
    # Initialize logging if output directory provided
    analyst_logger = None
    if output_dir:
        try:
            from src.analyst.logging_config import init_logger
            analyst_logger = init_logger(output_dir)
        except ImportError:
            pass
    
    print("\n" + "=" * 60)
    print("AGENTIC ETF TRADING ANALYST")
    print("=" * 60)
    print(f"  Analyst Model: {LLM_MODEL}")
    if REVIEWER_MODEL and REVIEWER_MODEL != LLM_MODEL:
        print(f"  Reviewer Model: {REVIEWER_MODEL}  ← Multi-model routing enabled")
    else:
        print(f"  Reviewer Model: {LLM_MODEL}  (same as analyst)")
    print(f"  Analyzing {len(candidates)} candidates...")
    
    # Log run start
    if analyst_logger:
        model_info = f"{LLM_MODEL}" + (f" + {REVIEWER_MODEL}" if REVIEWER_MODEL else "")
        analyst_logger.log_run_start(len(candidates), model_info)
    
    # Initialize state
    initial_state: AnalystState = {
        "candidates": candidates,
        "market_context": [],
        "news_cache": {},
        "thematic_analysis": None,
        "symbol_analyses": {},
        "review_results": None,
        "errors": [],
    }
    
    # Build and run graph
    graph = build_analyst_graph()
    final_state = graph.invoke(initial_state)
    
    # End instrumentation
    if instrumenter:
        instrumenter.end_run()
    
    # Get usage summary (basic)
    tracker = get_usage_tracker()
    print(tracker.summary(LLM_MODEL))
    
    # Print detailed instrumentation report
    if instrumenter:
        instrumenter.print_summary()
    
    # Log completion
    results_summary = {
        "symbols_analyzed": len(final_state.get("symbol_analyses", {})),
        "themes_found": len(final_state.get("thematic_analysis", {}).get("themes", [])),
        "errors_count": len(final_state.get("errors", [])),
    }
    if analyst_logger:
        analyst_logger.log_run_complete(tracker.to_dict(), results_summary)
    
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Build result
    result = {
        "thematic_analysis": final_state.get("thematic_analysis"),
        "symbol_analyses": final_state.get("symbol_analyses"),
        "review_results": final_state.get("review_results"),  # From reflection node
        "errors": final_state.get("errors", []),
        "timestamp": datetime.now().isoformat(),
        "usage": tracker.to_dict(),
        "metrics": instrumenter.generate_report() if instrumenter else None,
        "model": LLM_MODEL,
        "log_files": {},
    }
    
    # Save logs and reports if output_dir provided
    if output_dir:
        try:
            if analyst_logger:
                # Save full trace log (JSON with all prompts/responses)
                trace_path = analyst_logger.save_full_log(output_dir)
                result["log_files"]["trace"] = str(trace_path)
            
            # Save usage report (human-readable)
            from src.analyst.logging_config import generate_usage_report
            usage_path = generate_usage_report(tracker.to_dict(), LLM_MODEL, output_dir)
            result["log_files"]["usage_report"] = str(usage_path)
            print(f"  Usage report saved: {usage_path}")
            
            # Save detailed instrumentation report
            if instrumenter:
                metrics_json_path = instrumenter.save_report(output_dir)
                result["log_files"]["metrics_json"] = str(metrics_json_path)
                print(f"  Metrics (JSON) saved: {metrics_json_path}")
                
                metrics_txt_path = instrumenter.save_summary(output_dir)
                result["log_files"]["metrics_summary"] = str(metrics_txt_path)
                print(f"  Metrics (TXT) saved: {metrics_txt_path}")
            
        except Exception as e:
            print(f"  Warning: Could not save some log files: {e}")
    
    return result

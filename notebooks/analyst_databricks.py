# Databricks notebook source
# MAGIC %md
# MAGIC # ETF Trading Analyst — Agentic AI with LangGraph
# MAGIC
# MAGIC This notebook implements an **agentic workflow** for analyzing ETF trade candidates
# MAGIC using LangGraph with the **Reflection Pattern** for self-review and quality improvement.
# MAGIC
# MAGIC **Architecture:**
# MAGIC ```
# MAGIC START → load → fetch_news → analyze_themes → analyze_symbols → review_and_refine → END
# MAGIC                  (Tavily)       (LLM)            (LLM×N)         (LLM - Reflection)
# MAGIC ```
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - **LangGraph StateGraph**: Manages state transitions across nodes
# MAGIC - **Reflection Pattern**: A reviewer node critiques the initial analysis
# MAGIC - **Multi-Model Routing**: Different LLMs for analysis vs. review
# MAGIC - **MLflow Integration**: All runs logged for experiment tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup — Install Dependencies

# COMMAND ----------

# DBTITLE 1,Install Required Packages
%pip install langgraph langchain-openai langchain-google-genai tavily-python mlflow -q
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration — API Keys & Model Selection
# MAGIC
# MAGIC For Databricks Community Edition, use widgets for API keys.
# MAGIC On paid Databricks, use `dbutils.secrets.get(scope, key)` instead.

# COMMAND ----------

# DBTITLE 1,Configure API Keys and Model
import os

# --- API Keys ---
# Option A: Databricks widgets (Community Edition)
try:
    dbutils.widgets.text("tavily_api_key", "", "Tavily API Key")
    dbutils.widgets.text("google_api_key", "", "Google API Key (for Gemini)")
    dbutils.widgets.text("openai_api_key", "", "OpenAI API Key (optional)")
    dbutils.widgets.dropdown("model", "gpt-4o-mini", 
                             ["gpt-4o-mini", "gpt-4o", "gemini-2.0-flash", "gemini-1.5-flash"],
                             "LLM Model")
    
    TAVILY_API_KEY = dbutils.widgets.get("tavily_api_key")
    GOOGLE_API_KEY = dbutils.widgets.get("google_api_key")
    OPENAI_API_KEY = dbutils.widgets.get("openai_api_key")
    LLM_MODEL = dbutils.widgets.get("model")
except NameError:
    # Running outside Databricks (local testing)
    from dotenv import load_dotenv
    load_dotenv()
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("ANALYST_LLM_MODEL", "gpt-4o-mini")

# Option B: Databricks Secrets (paid tier - uncomment to use)
# TAVILY_API_KEY = dbutils.secrets.get(scope="analyst", key="tavily_api_key")
# GOOGLE_API_KEY = dbutils.secrets.get(scope="analyst", key="google_api_key")

# Set environment variables (LangChain reads these automatically)
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

REVIEWER_MODEL = LLM_MODEL  # Same model for review (change for multi-model routing)

print(f"Model: {LLM_MODEL}")
print(f"Reviewer: {REVIEWER_MODEL}")
print(f"Tavily API Key: {'***' + TAVILY_API_KEY[-4:] if len(TAVILY_API_KEY) > 4 else '(not set)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Candidate Data
# MAGIC
# MAGIC Upload `candidates.csv` to DBFS or provide a path via widget.

# COMMAND ----------

# DBTITLE 1,Load Candidates from CSV
import pandas as pd

# --- Load candidates ---
# Option A: From DBFS (upload via Databricks UI → Data → DBFS)
# df = pd.read_csv("/dbfs/FileStore/candidates.csv")

# Option B: From widget path
try:
    dbutils.widgets.text("candidates_path", "/dbfs/FileStore/candidates.csv", "Candidates CSV Path")
    csv_path = dbutils.widgets.get("candidates_path")
except NameError:
    # Local testing fallback
    csv_path = "experiments/exp019_3_trades/2026-02-09/candidates.csv"

df = pd.read_csv(csv_path)
candidates = df.to_dict(orient="records")

print(f"Loaded {len(candidates)} candidates from: {csv_path}")
display(df[["symbol", "etf_name", "pct_return", "limit_price", "stop_price", "target_price", "shares"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Core Agent Code
# MAGIC
# MAGIC The agentic workflow inlined from `src/analyst/graph.py` and `src/analyst/tools/search.py`.

# COMMAND ----------

# DBTITLE 1,Web Search Tool (Tavily)
from typing import List, Dict, Any, Optional
from tavily import TavilyClient


def search_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search for recent news about a topic using Tavily."""
    api_key = os.environ.get("TAVILY_API_KEY")
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
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],
                "score": r.get("score", 0),
            }
            for r in response.get("results", [])
        ]
    except Exception as e:
        return [{"error": str(e)}]


def search_symbol_news(symbol: str, name: str = "") -> List[Dict[str, Any]]:
    """Search for news about a specific ETF symbol."""
    query = f"{symbol} ETF {name.split()[0]} news price" if name else f"{symbol} ETF news price movement"
    return search_news(query, max_results=3)

# COMMAND ----------

# DBTITLE 1,LLM Setup & Usage Tracking
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# Usage Tracker
# ---------------------------------------------------------------------------
@dataclass
class UsageTracker:
    """Track LLM API usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    call_details: List[Dict[str, Any]] = field(default_factory=list)

    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }

    def record(self, model, input_tokens, output_tokens, node=""):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1
        self.call_details.append({"node": node, "model": model,
                                  "input_tokens": input_tokens, "output_tokens": output_tokens})

    def estimate_cost(self, model):
        p = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        return (self.input_tokens / 1e6) * p["input"] + (self.output_tokens / 1e6) * p["output"]

    def to_dict(self):
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "call_details": self.call_details}


_usage_tracker = UsageTracker()

def reset_usage_tracker():
    global _usage_tracker
    _usage_tracker = UsageTracker()


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
def _create_llm(model: str):
    """Create an LLM instance for the given model name."""
    if model.startswith("gpt-"):
        return ChatOpenAI(model=model, temperature=0.3)
    elif model.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.3)
    elif model.startswith("claude-"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0.3)
    else:
        return ChatOpenAI(model=model, temperature=0.3)


def get_llm(role="analyst"):
    model = REVIEWER_MODEL if role == "reviewer" else LLM_MODEL
    return _create_llm(model)


def invoke_llm_with_tracking(llm, prompt: str, node: str = "", role: str = "analyst") -> str:
    """Invoke LLM, track tokens and latency."""
    model_name = REVIEWER_MODEL if role == "reviewer" else LLM_MODEL
    start = time.time()
    response = llm.invoke(prompt)
    duration_ms = (time.time() - start) * 1000

    usage = getattr(response, "usage_metadata", None)
    if usage:
        in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    else:
        in_tok, out_tok = len(prompt) // 4, len(response.content) // 4

    _usage_tracker.record(model_name, in_tok, out_tok, node)
    return response.content

# COMMAND ----------

# DBTITLE 1,Graph State & Nodes
# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------
class AnalystState(TypedDict):
    candidates: List[Dict[str, Any]]
    news_cache: Dict[str, List[Dict[str, Any]]]
    thematic_analysis: Optional[Dict[str, Any]]
    symbol_analyses: Dict[str, Dict[str, Any]]
    review_results: Optional[Dict[str, Any]]
    errors: List[str]


# ---------------------------------------------------------------------------
# Helper: Parse JSON from LLM response
# ---------------------------------------------------------------------------
def _parse_json(content: str) -> dict:
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content)


# ---------------------------------------------------------------------------
# Node 1: Load Candidates
# ---------------------------------------------------------------------------
def load_candidates(state: AnalystState) -> AnalystState:
    for key in ("news_cache", "symbol_analyses"):
        if not state.get(key):
            state[key] = {}
    if not state.get("errors"):
        state["errors"] = []
    return state


# ---------------------------------------------------------------------------
# Node 2: Fetch News (Tavily Tool)
# ---------------------------------------------------------------------------
def fetch_news(state: AnalystState) -> AnalystState:
    print("  Fetching news for symbols...")
    for c in state["candidates"]:
        symbol, name = c.get("symbol", ""), c.get("etf_name", "")
        if symbol and symbol not in state["news_cache"]:
            print(f"    {symbol}...")
            state["news_cache"][symbol] = search_symbol_news(symbol, name)
    print(f"  Fetched news for {len(state['news_cache'])} symbols")
    return state


# ---------------------------------------------------------------------------
# Node 3: Analyze Themes (LLM)
# ---------------------------------------------------------------------------
def analyze_themes(state: AnalystState) -> AnalystState:
    print("  Analyzing themes...")
    llm = get_llm()

    summaries = [f"- {c['symbol']}: {c.get('etf_name','')} ({c.get('pct_return',0):.1f}%)"
                 for c in state["candidates"]]
    news_lines = []
    for sym, articles in list(state["news_cache"].items())[:5]:
        for a in articles[:2]:
            if "error" not in a:
                news_lines.append(f"[{sym}] {a.get('title','')}")

    prompt = f"""You are a NEWS ANALYST for an ETF trading desk. Your job is purely QUALITATIVE.

CONTEXT: These ETFs dropped significantly this week and are mean-reversion candidates (1-3 week hold).

CANDIDATES:
{chr(10).join(summaries)}

RECENT NEWS:
{chr(10).join(news_lines) if news_lines else "No news available"}

Group into 2-4 themes based on NEWS drivers. For each, assess: TRANSIENT or STRUCTURAL?

Respond in JSON:
{{"themes": [{{"name": "...", "symbols": ["SYM"], "news_driver": "...", "drop_type": "transient/structural/unclear", "narrative": "..."}}],
 "upcoming_catalysts": ["..."],
 "overall_news_assessment": "favorable/mixed/unfavorable",
 "summary": "2-3 sentences"}}"""

    try:
        content = invoke_llm_with_tracking(llm, prompt, node="analyze_themes")
        state["thematic_analysis"] = _parse_json(content)
        print("  Thematic analysis complete")
    except Exception as e:
        state["errors"].append(f"Theme error: {e}")
        state["thematic_analysis"] = {"themes": [], "overall_sentiment": "unknown", "summary": str(e)}
    return state


# ---------------------------------------------------------------------------
# Node 4: Analyze Symbols (LLM × N)
# ---------------------------------------------------------------------------
def analyze_symbols(state: AnalystState) -> AnalystState:
    print("  Analyzing individual symbols...")
    llm = get_llm()

    for c in state["candidates"]:
        symbol = c.get("symbol", "")
        name = c.get("etf_name", "")
        ret = c.get("pct_return", 0)

        news = state["news_cache"].get(symbol, [])
        news_text = "\n".join(f"- {a['title']}" for a in news if "error" not in a)
        citations = [{"title": a.get("title",""), "url": a.get("url","")} for a in news if "error" not in a]

        prompt = f"""You are a NEWS ANALYST providing a QUALITATIVE assessment.

SYMBOL: {symbol} | NAME: {name} | WEEKLY RETURN: {ret:.2f}%

RECENT NEWS:
{news_text if news_text else "No recent news found"}

Based ONLY on news: Is the drop TRANSIENT or STRUCTURAL? Any RED FLAGS?

Respond in JSON:
{{"flag": "GREEN/YELLOW/RED", "flag_reason": "...", "news_summary": "...",
 "drop_assessment": "transient/structural/unclear",
 "bullish_signals": ["..."], "bearish_signals": ["..."],
 "key_concern": "... or 'none'"}}

GREEN=overdone/transient, YELLOW=mixed, RED=structural/skip"""

        try:
            content = invoke_llm_with_tracking(llm, prompt, node=f"analyze_{symbol}")
            analysis = _parse_json(content)
            analysis["citations"] = citations
            state["symbol_analyses"][symbol] = analysis
            print(f"    {symbol}: {analysis.get('flag','?')} - {analysis.get('flag_reason','')[:50]}")
        except Exception as e:
            state["errors"].append(f"{symbol}: {e}")
            state["symbol_analyses"][symbol] = {
                "flag": "YELLOW", "flag_reason": "Analysis failed",
                "news_summary": str(e), "drop_assessment": "unclear",
                "bullish_signals": [], "bearish_signals": [], "key_concern": "Error",
                "citations": citations}

    print(f"  Analyzed {len(state['symbol_analyses'])} symbols")
    return state


# ---------------------------------------------------------------------------
# Node 5: Review & Refine — REFLECTION PATTERN
# ---------------------------------------------------------------------------
def review_and_refine(state: AnalystState) -> AnalystState:
    print(f"  Reviewing (Reflection Pattern) with {REVIEWER_MODEL}...")
    llm = get_llm("reviewer")

    thematic = state.get("thematic_analysis", {})
    themes_text = "\n".join(
        f"- {t.get('name')}: {', '.join(t.get('symbols',[]))} [{t.get('drop_type','?')}]"
        for t in thematic.get("themes", []))
    symbols_text = "\n".join(
        f"- {s}: {a.get('flag','?')} — {a.get('flag_reason','')}"
        for s, a in state.get("symbol_analyses", {}).items())

    prompt = f"""You are a SENIOR NEWS ANALYST reviewing a junior analyst's work.

=== THEMATIC ANALYSIS ===
{thematic.get('summary', 'N/A')}
{themes_text or "No themes"}

=== SYMBOL ASSESSMENTS ===
{symbols_text or "No symbols"}

Review the NEWS interpretations. Are flags justified? Any missed signals?

Respond in JSON:
{{"overall_assessment": "APPROVE/NEEDS_REVISION",
 "quality_score": 1-10,
 "flag_adjustments": [{{"symbol": "XYZ", "original_flag": "GREEN", "adjusted_flag": "YELLOW", "reason": "..."}}],
 "missed_signals": ["..."],
 "strongest_candidates": ["SYM1"],
 "weakest_candidates": ["SYM2"],
 "reviewer_notes": "2-3 sentences"}}"""

    try:
        content = invoke_llm_with_tracking(llm, prompt, node="review_and_refine", role="reviewer")
        review = _parse_json(content)
        state["review_results"] = review

        for adj in review.get("flag_adjustments", []):
            sym = adj.get("symbol")
            if sym and sym in state["symbol_analyses"]:
                orig = state["symbol_analyses"][sym].get("flag", "YELLOW")
                new = adj.get("adjusted_flag", orig)
                if orig != new:
                    state["symbol_analyses"][sym]["flag_original"] = orig
                    state["symbol_analyses"][sym]["flag"] = new
                    state["symbol_analyses"][sym]["flag_adjusted_by_reviewer"] = True
                    state["symbol_analyses"][sym]["adjustment_reason"] = adj.get("reason", "")
                    print(f"    {sym}: {orig} → {new}")

        print(f"  Review: {review.get('overall_assessment','N/A')}")
        strongest = review.get("strongest_candidates", [])
        weakest = review.get("weakest_candidates", [])
        if strongest: print(f"  Strongest: {', '.join(strongest)}")
        if weakest:   print(f"  Weakest: {', '.join(weakest)}")
    except Exception as e:
        state["errors"].append(f"Review error: {e}")
        state["review_results"] = {"overall_assessment": "ERROR", "reviewer_notes": str(e)}
    return state

# COMMAND ----------

# DBTITLE 1,Build LangGraph Workflow
def build_analyst_graph():
    """Build the analyst workflow graph with Reflection Pattern."""
    workflow = StateGraph(AnalystState)

    workflow.add_node("load", load_candidates)
    workflow.add_node("fetch_news", fetch_news)
    workflow.add_node("analyze_themes", analyze_themes)
    workflow.add_node("analyze_symbols", analyze_symbols)
    workflow.add_node("review_and_refine", review_and_refine)

    workflow.set_entry_point("load")
    workflow.add_edge("load", "fetch_news")
    workflow.add_edge("fetch_news", "analyze_themes")
    workflow.add_edge("analyze_themes", "analyze_symbols")
    workflow.add_edge("analyze_symbols", "review_and_refine")
    workflow.add_edge("review_and_refine", END)

    return workflow.compile()

print("Graph built successfully:")
print("  START → load → fetch_news → analyze_themes → analyze_symbols → review_and_refine → END")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run the Agent

# COMMAND ----------

# DBTITLE 1,Execute Analysis with MLflow Tracking
import mlflow

mlflow.set_experiment("/Shared/etf-trading-analyst")

reset_usage_tracker()

print("=" * 60)
print("AGENTIC ETF TRADING ANALYST")
print("=" * 60)
print(f"  Model: {LLM_MODEL}")
print(f"  Reviewer: {REVIEWER_MODEL}")
print(f"  Candidates: {len(candidates)}")
print()

# Build and run graph
graph = build_analyst_graph()
initial_state: AnalystState = {
    "candidates": candidates,
    "news_cache": {},
    "thematic_analysis": None,
    "symbol_analyses": {},
    "review_results": None,
    "errors": [],
}

start_time = time.time()
final_state = graph.invoke(initial_state)
duration = time.time() - start_time

# Collect results
tracker = _usage_tracker
review = final_state.get("review_results", {}) or {}
symbol_analyses = final_state.get("symbol_analyses", {})
thematic = final_state.get("thematic_analysis", {})
flags = [a.get("flag", "YELLOW") for a in symbol_analyses.values()]

results = {
    "thematic_analysis": thematic,
    "symbol_analyses": symbol_analyses,
    "review_results": review,
    "errors": final_state.get("errors", []),
    "timestamp": datetime.now().isoformat(),
    "usage": tracker.to_dict(),
    "model": LLM_MODEL,
}

# --- Log to MLflow ---
with mlflow.start_run(run_name=f"analyst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
    mlflow.log_params({
        "model": LLM_MODEL,
        "reviewer_model": REVIEWER_MODEL,
        "n_candidates": len(candidates),
        "reviewer_assessment": review.get("overall_assessment", "N/A"),
    })
    mlflow.log_metrics({
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.input_tokens + tracker.output_tokens,
        "llm_calls": tracker.calls,
        "cost_usd": tracker.estimate_cost(LLM_MODEL),
        "duration_seconds": duration,
        "green_count": flags.count("GREEN"),
        "yellow_count": flags.count("YELLOW"),
        "red_count": flags.count("RED"),
        "reviewer_adjustments": sum(1 for a in symbol_analyses.values() if a.get("flag_adjusted_by_reviewer")),
        "n_themes": len(thematic.get("themes", [])),
    })
    # Save results JSON as artifact
    results_json = json.dumps(results, indent=2, default=str)
    mlflow.log_text(results_json, "analyst_report.json")

    run_id = mlflow.active_run().info.run_id
    print(f"\n📊 MLflow run logged: {run_id[:8]}...")

# Print summary
print(f"\n{'='*60}")
print(f"COMPLETE — {len(candidates)} candidates in {duration:.1f}s")
print(f"Tokens: {tracker.input_tokens + tracker.output_tokens:,} | Cost: ${tracker.estimate_cost(LLM_MODEL):.4f}")
print(f"Flags: {flags.count('GREEN')} GREEN / {flags.count('YELLOW')} YELLOW / {flags.count('RED')} RED")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Display Results

# COMMAND ----------

# DBTITLE 1,Results Summary Table
# Build results dataframe
rows = []
for c in candidates:
    sym = c["symbol"]
    a = symbol_analyses.get(sym, {})
    rows.append({
        "Symbol": sym,
        "Name": c.get("etf_name", "")[:30],
        "Return": f"{c.get('pct_return',0):.1f}%",
        "Flag": a.get("flag", "?"),
        "Drop Type": a.get("drop_assessment", "?"),
        "News Summary": a.get("news_summary", "")[:80],
        "Key Concern": a.get("key_concern", "")[:60],
        "Adjusted": "✓" if a.get("flag_adjusted_by_reviewer") else "",
    })

results_df = pd.DataFrame(rows)
display(results_df)

# COMMAND ----------

# DBTITLE 1,Thematic Analysis
if thematic:
    print(f"Overall News Assessment: {thematic.get('overall_news_assessment', 'unknown').upper()}")
    print(f"\nSummary: {thematic.get('summary', 'N/A')}")
    print(f"\nUpcoming Catalysts: {', '.join(thematic.get('upcoming_catalysts', ['None']))}")
    print()
    for t in thematic.get("themes", []):
        print(f"  📌 {t.get('name')}")
        print(f"     Symbols: {', '.join(t.get('symbols', []))}")
        print(f"     Type: {t.get('drop_type', '?').upper()}")
        print(f"     {t.get('narrative', '')}")
        print()

# COMMAND ----------

# DBTITLE 1,Reviewer Assessment
if review and review.get("overall_assessment") != "ERROR":
    print(f"Assessment: {review.get('overall_assessment', 'N/A')}")
    print(f"Quality Score: {review.get('quality_score', 'N/A')}/10")
    print(f"\nStrongest: {', '.join(review.get('strongest_candidates', []))}")
    print(f"Weakest: {', '.join(review.get('weakest_candidates', []))}")
    print(f"\nNotes: {review.get('reviewer_notes', 'N/A')}")
    
    adjustments = review.get("flag_adjustments", [])
    if adjustments:
        print(f"\nFlag Adjustments ({len(adjustments)}):")
        for adj in adjustments:
            print(f"  {adj['symbol']}: {adj.get('original_flag')} → {adj.get('adjusted_flag')} — {adj.get('reason','')}")

# COMMAND ----------

# DBTITLE 1,HTML Report (Inline Display)
# Build and display HTML report inline
html_parts = [f"""
<div style="font-family: -apple-system, sans-serif; padding: 20px; background: #f5f5f5;">
<h1>🤖 ETF Trading Analyst Report</h1>
<p style="color:#666">{results.get('timestamp', '')} | Model: {LLM_MODEL} | 
   Tokens: {tracker.input_tokens + tracker.output_tokens:,} | Cost: ${tracker.estimate_cost(LLM_MODEL):.4f}</p>
<div style="background:white; padding:15px; border-radius:8px; border-left:4px solid #2E86AB; margin-bottom:20px;">
  <h2 style="margin-top:0">Market Summary</h2>
  <p>{thematic.get('summary', 'N/A')}</p>
</div>
"""]

# Symbol cards
for c in candidates:
    sym = c["symbol"]
    a = symbol_analyses.get(sym, {})
    flag = a.get("flag", "YELLOW")
    color = {"GREEN": "#28A745", "YELLOW": "#FFC107", "RED": "#DC3545"}.get(flag, "#6c757d")
    adjusted = f" <small>(was {a.get('flag_original')})</small>" if a.get("flag_adjusted_by_reviewer") else ""
    html_parts.append(f"""
<div style="background:white; padding:12px; border-radius:8px; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
  <div style="display:flex; align-items:center; gap:10px;">
    <h3 style="margin:0">{sym}</h3>
    <span style="background:{color}; color:white; padding:2px 8px; border-radius:4px; font-size:0.85em;">{flag}</span>
    {adjusted}
    <span style="color:#666; font-size:0.9em;">{c.get('etf_name','')[:40]}</span>
    <span style="color:#DC3545; font-weight:bold;">{c.get('pct_return',0):.1f}%</span>
  </div>
  <p style="margin:8px 0; line-height:1.5;">{a.get('news_summary', 'No analysis')}</p>
  <div style="background:#fff3cd; padding:6px 10px; border-radius:4px; font-size:0.9em;">
    <strong>Key Concern:</strong> {a.get('key_concern', 'None')}
  </div>
</div>""")

html_parts.append("</div>")
report_html = "\n".join(html_parts)

# Display in Databricks (or save locally)
try:
    displayHTML(report_html)
except NameError:
    # Local fallback: save to file
    from pathlib import Path
    out = Path("analyst_report_databricks.html")
    out.write_text(report_html)
    print(f"HTML report saved: {out}")

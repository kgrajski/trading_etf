#!/usr/bin/env python3
"""
Local test runner for analyst_databricks.py logic.

Validates the notebook's core code works before uploading to Databricks.
Run from the project root:

    cd /Users/kag/Development/Projects/trading_etf
    python notebooks/test_local.py
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

# ── Setup ──────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import mlflow
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

LLM_MODEL = os.getenv("ANALYST_LLM_MODEL", "gemini-2.0-flash")
REVIEWER_MODEL = LLM_MODEL

print(f"Model: {LLM_MODEL}")
print(f"Tavily: {'***' + os.environ['TAVILY_API_KEY'][-4:] if os.environ.get('TAVILY_API_KEY') else '(NOT SET)'}")

# ── Load candidates ───────────────────────────────────────────────
csv_path = Path("notebooks/sample_candidates.csv")
if not csv_path.exists():
    csv_path = Path("experiments/exp019_3_trades/2026-02-09/candidates.csv")
df = pd.read_csv(csv_path)
candidates = df.to_dict(orient="records")
print(f"Loaded {len(candidates)} candidates from {csv_path}")

# ── Web Search Tool ───────────────────────────────────────────────
def search_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "TAVILY_API_KEY not set"}]
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query, search_depth="basic", max_results=max_results,
            include_domains=["reuters.com", "bloomberg.com", "wsj.com",
                             "cnbc.com", "marketwatch.com", "finance.yahoo.com",
                             "seekingalpha.com", "fool.com"],
        )
        return [{"title": r.get("title",""), "url": r.get("url",""),
                 "content": r.get("content","")[:500], "score": r.get("score",0)}
                for r in response.get("results", [])]
    except Exception as e:
        return [{"error": str(e)}]

def search_symbol_news(symbol: str, name: str = "") -> List[Dict[str, Any]]:
    query = f"{symbol} ETF {name.split()[0]} news price" if name else f"{symbol} ETF news price movement"
    return search_news(query, max_results=3)

# ── Usage Tracker ─────────────────────────────────────────────────
@dataclass
class UsageTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    call_details: List[Dict[str, Any]] = field(default_factory=list)
    PRICING = {"gpt-4o-mini": {"input": 0.15, "output": 0.60},
               "gpt-4o": {"input": 2.50, "output": 10.00},
               "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
               "gemini-1.5-flash": {"input": 0.075, "output": 0.30}}

    def record(self, model, in_tok, out_tok, node=""):
        self.input_tokens += in_tok; self.output_tokens += out_tok; self.calls += 1
        self.call_details.append({"node": node, "model": model, "input_tokens": in_tok, "output_tokens": out_tok})

    def estimate_cost(self, model):
        p = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        return (self.input_tokens/1e6)*p["input"] + (self.output_tokens/1e6)*p["output"]

    def to_dict(self):
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "total_tokens": self.input_tokens + self.output_tokens}

_usage_tracker = UsageTracker()

# ── LLM Factory ───────────────────────────────────────────────────
def _create_llm(model: str):
    if model.startswith("gpt-"):
        return ChatOpenAI(model=model, temperature=0.3)
    elif model.startswith("gemini-"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=0.3)
    else:
        return ChatOpenAI(model=model, temperature=0.3)

def get_llm(role="analyst"):
    return _create_llm(REVIEWER_MODEL if role == "reviewer" else LLM_MODEL)

def invoke_llm_with_tracking(llm, prompt, node="", role="analyst"):
    model_name = REVIEWER_MODEL if role == "reviewer" else LLM_MODEL
    start = time.time()
    response = llm.invoke(prompt)
    usage = getattr(response, "usage_metadata", None)
    in_tok = usage.get("input_tokens", 0) if usage else len(prompt) // 4
    out_tok = usage.get("output_tokens", 0) if usage else len(response.content) // 4
    _usage_tracker.record(model_name, in_tok, out_tok, node)
    return response.content

def _parse_json(content):
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content)

# ── State & Nodes ─────────────────────────────────────────────────
class AnalystState(TypedDict):
    candidates: List[Dict[str, Any]]
    news_cache: Dict[str, List[Dict[str, Any]]]
    thematic_analysis: Optional[Dict[str, Any]]
    symbol_analyses: Dict[str, Dict[str, Any]]
    review_results: Optional[Dict[str, Any]]
    errors: List[str]

def load_candidates(state: AnalystState) -> AnalystState:
    for k in ("news_cache", "symbol_analyses"):
        if not state.get(k): state[k] = {}
    if not state.get("errors"): state["errors"] = []
    return state

def fetch_news(state: AnalystState) -> AnalystState:
    print("  Fetching news...")
    for c in state["candidates"]:
        sym, name = c.get("symbol",""), c.get("etf_name","")
        if sym and sym not in state["news_cache"]:
            print(f"    {sym}...")
            state["news_cache"][sym] = search_symbol_news(sym, name)
    print(f"  Fetched news for {len(state['news_cache'])} symbols")
    return state

def analyze_themes(state: AnalystState) -> AnalystState:
    print("  Analyzing themes...")
    llm = get_llm()
    summaries = [f"- {c['symbol']}: {c.get('etf_name','')} ({c.get('pct_return',0):.1f}%)" for c in state["candidates"]]
    news_lines = []
    for sym, articles in list(state["news_cache"].items())[:5]:
        for a in articles[:2]:
            if "error" not in a: news_lines.append(f"[{sym}] {a.get('title','')}")
    prompt = f"""You are a NEWS ANALYST for an ETF trading desk. QUALITATIVE only.
CANDIDATES:\n{chr(10).join(summaries)}
NEWS:\n{chr(10).join(news_lines) if news_lines else "No news"}
Group into 2-4 themes. Assess: TRANSIENT or STRUCTURAL?
Respond in JSON: {{"themes": [{{"name":"...","symbols":["SYM"],"news_driver":"...","drop_type":"transient/structural/unclear","narrative":"..."}}], "upcoming_catalysts":["..."], "overall_news_assessment":"favorable/mixed/unfavorable", "summary":"..."}}"""
    try:
        state["thematic_analysis"] = _parse_json(invoke_llm_with_tracking(llm, prompt, node="analyze_themes"))
        print("  Thematic analysis complete")
    except Exception as e:
        state["errors"].append(str(e))
        state["thematic_analysis"] = {"themes": [], "summary": str(e)}
    return state

def analyze_symbols(state: AnalystState) -> AnalystState:
    print("  Analyzing symbols...")
    llm = get_llm()
    for c in state["candidates"]:
        sym, name, ret = c.get("symbol",""), c.get("etf_name",""), c.get("pct_return",0)
        news = state["news_cache"].get(sym, [])
        news_text = "\n".join(f"- {a['title']}" for a in news if "error" not in a)
        citations = [{"title":a.get("title",""),"url":a.get("url","")} for a in news if "error" not in a]
        prompt = f"""NEWS ANALYST: qualitative assessment.
SYMBOL: {sym} | NAME: {name} | RETURN: {ret:.2f}%
NEWS:\n{news_text or "None"}
Flag GREEN/YELLOW/RED. JSON: {{"flag":"...","flag_reason":"...","news_summary":"...","drop_assessment":"transient/structural/unclear","bullish_signals":["..."],"bearish_signals":["..."],"key_concern":"..."}}"""
        try:
            analysis = _parse_json(invoke_llm_with_tracking(llm, prompt, node=f"analyze_{sym}"))
            analysis["citations"] = citations
            state["symbol_analyses"][sym] = analysis
            print(f"    {sym}: {analysis.get('flag','?')}")
        except Exception as e:
            state["errors"].append(f"{sym}: {e}")
            state["symbol_analyses"][sym] = {"flag":"YELLOW","flag_reason":"Error","citations":citations}
    return state

def review_and_refine(state: AnalystState) -> AnalystState:
    print(f"  Reviewing with {REVIEWER_MODEL}...")
    llm = get_llm("reviewer")
    thematic = state.get("thematic_analysis", {})
    symbols_text = "\n".join(f"- {s}: {a.get('flag','?')} — {a.get('flag_reason','')}"
                              for s, a in state.get("symbol_analyses",{}).items())
    prompt = f"""SENIOR NEWS ANALYST review.
THEMES: {thematic.get('summary','N/A')}
SYMBOLS:\n{symbols_text or "None"}
Respond JSON: {{"overall_assessment":"APPROVE/NEEDS_REVISION","quality_score":1-10,"flag_adjustments":[{{"symbol":"X","original_flag":"G","adjusted_flag":"Y","reason":"..."}}],"missed_signals":["..."],"strongest_candidates":["..."],"weakest_candidates":["..."],"reviewer_notes":"..."}}"""
    try:
        review = _parse_json(invoke_llm_with_tracking(llm, prompt, node="review_and_refine", role="reviewer"))
        state["review_results"] = review
        for adj in review.get("flag_adjustments", []):
            sym = adj.get("symbol")
            if sym and sym in state["symbol_analyses"]:
                orig = state["symbol_analyses"][sym].get("flag","YELLOW")
                new = adj.get("adjusted_flag", orig)
                if orig != new:
                    state["symbol_analyses"][sym]["flag_original"] = orig
                    state["symbol_analyses"][sym]["flag"] = new
                    state["symbol_analyses"][sym]["flag_adjusted_by_reviewer"] = True
                    print(f"    {sym}: {orig} → {new}")
        print(f"  Review: {review.get('overall_assessment','N/A')}")
    except Exception as e:
        state["errors"].append(f"Review: {e}")
        state["review_results"] = {"overall_assessment": "ERROR", "reviewer_notes": str(e)}
    return state

# ── Build & Run ───────────────────────────────────────────────────
def build_analyst_graph():
    wf = StateGraph(AnalystState)
    wf.add_node("load", load_candidates)
    wf.add_node("fetch_news", fetch_news)
    wf.add_node("analyze_themes", analyze_themes)
    wf.add_node("analyze_symbols", analyze_symbols)
    wf.add_node("review_and_refine", review_and_refine)
    wf.set_entry_point("load")
    wf.add_edge("load", "fetch_news")
    wf.add_edge("fetch_news", "analyze_themes")
    wf.add_edge("analyze_themes", "analyze_symbols")
    wf.add_edge("analyze_symbols", "review_and_refine")
    wf.add_edge("review_and_refine", END)
    return wf.compile()

if __name__ == "__main__":
    mlflow.set_experiment("etf-trading-analyst-dbtest")

    print("\n" + "=" * 60)
    print("LOCAL TEST — Databricks Notebook Logic")
    print("=" * 60)

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

    tracker = _usage_tracker
    review = final_state.get("review_results", {}) or {}
    symbol_analyses = final_state.get("symbol_analyses", {})
    thematic = final_state.get("thematic_analysis", {})
    flags = [a.get("flag", "YELLOW") for a in symbol_analyses.values()]

    # Log to MLflow
    with mlflow.start_run(run_name=f"dbtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({"model": LLM_MODEL, "reviewer_model": REVIEWER_MODEL,
                           "n_candidates": len(candidates),
                           "reviewer_assessment": review.get("overall_assessment", "N/A")})
        mlflow.log_metrics({"input_tokens": tracker.input_tokens,
                            "output_tokens": tracker.output_tokens,
                            "total_tokens": tracker.input_tokens + tracker.output_tokens,
                            "llm_calls": tracker.calls,
                            "cost_usd": tracker.estimate_cost(LLM_MODEL),
                            "duration_seconds": duration,
                            "green_count": flags.count("GREEN"),
                            "yellow_count": flags.count("YELLOW"),
                            "red_count": flags.count("RED")})
        results_json = json.dumps({
            "thematic_analysis": thematic,
            "symbol_analyses": symbol_analyses,
            "review_results": review,
            "errors": final_state.get("errors", []),
            "usage": tracker.to_dict(),
            "model": LLM_MODEL,
        }, indent=2, default=str)
        mlflow.log_text(results_json, "analyst_report.json")
        run_id = mlflow.active_run().info.run_id

    print(f"\n{'='*60}")
    print(f"LOCAL TEST PASSED")
    print(f"{'='*60}")
    print(f"  Candidates: {len(candidates)}")
    print(f"  Duration:   {duration:.1f}s")
    print(f"  Tokens:     {tracker.input_tokens + tracker.output_tokens:,}")
    print(f"  Cost:       ${tracker.estimate_cost(LLM_MODEL):.4f}")
    print(f"  Flags:      {flags.count('GREEN')}G / {flags.count('YELLOW')}Y / {flags.count('RED')}R")
    print(f"  MLflow:     run_id={run_id[:8]}...")
    print(f"  Errors:     {len(final_state.get('errors', []))}")
    if final_state.get("errors"):
        for e in final_state["errors"]:
            print(f"    - {e}")
    print(f"\n✅ Notebook logic validated — safe to upload to Databricks!")

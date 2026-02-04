"""
Comprehensive Instrumentation for ETF Trading Analyst.

This module provides detailed metrics tracking for developing intuitions
around LLM workloads including:
- Token consumption (input, output, efficiency)
- Cost estimation (per call, per node, total)
- Latency (per call, per node, pipeline)
- Throughput (tokens/sec, symbols/min)

Usage:
    from src.analyst.instrumentation import Instrumenter, get_instrumenter

    # Start tracking
    inst = get_instrumenter()
    inst.start_run(model="gpt-4o-mini", candidates_count=15)
    
    # Track a node
    with inst.track_node("analyze_themes"):
        # ... node work ...
        inst.record_llm_call(input_tokens=500, output_tokens=200, duration_ms=1500)
    
    # Get report
    report = inst.generate_report()
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import json


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    node: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_ms: float
    timestamp: str
    prompt_chars: int = 0
    response_chars: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def token_efficiency(self) -> float:
        """Output tokens per input token (higher = more efficient)."""
        return self.output_tokens / max(self.input_tokens, 1)
    
    @property
    def tokens_per_second(self) -> float:
        """Processing speed in tokens per second."""
        if self.duration_ms <= 0:
            return 0
        return (self.total_tokens / self.duration_ms) * 1000


@dataclass
class NodeMetrics:
    """Metrics for a graph node execution."""
    name: str
    start_time: float = 0
    end_time: float = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_duration_ms: float = 0
    tool_duration_ms: float = 0
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000
    
    @property
    def overhead_ms(self) -> float:
        """Time spent on non-LLM, non-tool work."""
        return self.duration_ms - self.llm_duration_ms - self.tool_duration_ms


@dataclass 
class ToolCallMetrics:
    """Metrics for a tool call (e.g., web search)."""
    tool: str
    node: str
    duration_ms: float
    success: bool
    result_size: int = 0  # e.g., number of results returned


class Instrumenter:
    """
    Comprehensive metrics tracker for LLM agent workloads.
    
    Tracks tokens, costs, latency, and throughput to help develop
    intuitions about LLM application performance.
    """
    
    # Pricing per 1M tokens (Jan 2025)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
        "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new run."""
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model: str = ""
        self.candidates_count: int = 0
        self.start_time: float = 0
        self.end_time: float = 0
        
        # Detailed metrics
        self.llm_calls: List[LLMCallMetrics] = []
        self.tool_calls: List[ToolCallMetrics] = []
        self.node_metrics: Dict[str, NodeMetrics] = {}
        
        # Current tracking state
        self._current_node: Optional[str] = None
        self._node_start_time: float = 0
    
    def start_run(self, model: str, candidates_count: int):
        """Start tracking a new analysis run."""
        self.reset()
        self.model = model
        self.candidates_count = candidates_count
        self.start_time = time.time()
    
    def end_run(self):
        """Mark the end of the analysis run."""
        self.end_time = time.time()
    
    @contextmanager
    def track_node(self, node_name: str):
        """Context manager to track a node's execution."""
        self._current_node = node_name
        self._node_start_time = time.time()
        
        if node_name not in self.node_metrics:
            self.node_metrics[node_name] = NodeMetrics(name=node_name)
        
        self.node_metrics[node_name].start_time = self._node_start_time
        
        try:
            yield
        finally:
            self.node_metrics[node_name].end_time = time.time()
            self._current_node = None
    
    def record_llm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        node: Optional[str] = None,
        prompt_chars: int = 0,
        response_chars: int = 0,
    ):
        """Record metrics for an LLM call."""
        node = node or self._current_node or "unknown"
        
        call = LLMCallMetrics(
            node=node,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            prompt_chars=prompt_chars,
            response_chars=response_chars,
        )
        self.llm_calls.append(call)
        
        # Update node metrics
        if node in self.node_metrics:
            self.node_metrics[node].llm_calls += 1
            self.node_metrics[node].input_tokens += input_tokens
            self.node_metrics[node].output_tokens += output_tokens
            self.node_metrics[node].llm_duration_ms += duration_ms
    
    def record_tool_call(
        self,
        tool: str,
        duration_ms: float,
        success: bool = True,
        result_size: int = 0,
        node: Optional[str] = None,
    ):
        """Record metrics for a tool call."""
        node = node or self._current_node or "unknown"
        
        call = ToolCallMetrics(
            tool=tool,
            node=node,
            duration_ms=duration_ms,
            success=success,
            result_size=result_size,
        )
        self.tool_calls.append(call)
        
        # Update node metrics
        if node in self.node_metrics:
            self.node_metrics[node].tool_calls += 1
            self.node_metrics[node].tool_duration_ms += duration_ms
    
    # =========================================================================
    # Computed Metrics
    # =========================================================================
    
    @property
    def total_duration_ms(self) -> float:
        """Total pipeline duration in milliseconds."""
        if self.end_time <= 0:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
    
    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)
    
    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)
    
    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def total_llm_duration_ms(self) -> float:
        return sum(c.duration_ms for c in self.llm_calls)
    
    @property
    def total_tool_duration_ms(self) -> float:
        return sum(c.duration_ms for c in self.tool_calls)
    
    @property
    def overhead_ms(self) -> float:
        """Time not spent on LLM or tool calls."""
        return self.total_duration_ms - self.total_llm_duration_ms - self.total_tool_duration_ms
    
    @property
    def tokens_per_second(self) -> float:
        """Overall token processing rate."""
        if self.total_llm_duration_ms <= 0:
            return 0
        return (self.total_tokens / self.total_llm_duration_ms) * 1000
    
    @property
    def symbols_per_minute(self) -> float:
        """Analysis throughput."""
        if self.total_duration_ms <= 0:
            return 0
        return (self.candidates_count / self.total_duration_ms) * 60000
    
    def estimate_cost(self, model: Optional[str] = None) -> float:
        """Estimate total cost for the run."""
        model = model or self.model
        pricing = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def cost_per_symbol(self) -> float:
        """Average cost per symbol analyzed."""
        if self.candidates_count <= 0:
            return 0
        return self.estimate_cost() / self.candidates_count
    
    # =========================================================================
    # Reporting
    # =========================================================================
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report."""
        return {
            "run_id": self.run_id,
            "model": self.model,
            "candidates_count": self.candidates_count,
            
            # Token metrics
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
                "efficiency_ratio": self.total_output_tokens / max(self.total_input_tokens, 1),
                "per_symbol_avg": self.total_tokens / max(self.candidates_count, 1),
            },
            
            # Cost metrics
            "cost": {
                "total_usd": self.estimate_cost(),
                "per_symbol_usd": self.cost_per_symbol(),
                "comparison": {
                    m: self.estimate_cost(m) for m in self.PRICING.keys()
                },
            },
            
            # Latency metrics
            "latency": {
                "total_ms": self.total_duration_ms,
                "total_seconds": self.total_duration_ms / 1000,
                "llm_ms": self.total_llm_duration_ms,
                "tool_ms": self.total_tool_duration_ms,
                "overhead_ms": self.overhead_ms,
                "breakdown_pct": {
                    "llm": (self.total_llm_duration_ms / max(self.total_duration_ms, 1)) * 100,
                    "tool": (self.total_tool_duration_ms / max(self.total_duration_ms, 1)) * 100,
                    "overhead": (self.overhead_ms / max(self.total_duration_ms, 1)) * 100,
                },
            },
            
            # Throughput metrics
            "throughput": {
                "tokens_per_second": self.tokens_per_second,
                "symbols_per_minute": self.symbols_per_minute,
            },
            
            # Call counts
            "calls": {
                "llm_total": len(self.llm_calls),
                "tool_total": len(self.tool_calls),
                "llm_success_rate": 1.0,  # Could track failures
                "tool_success_rate": sum(1 for t in self.tool_calls if t.success) / max(len(self.tool_calls), 1),
            },
            
            # Per-node breakdown
            "nodes": {
                name: {
                    "duration_ms": node.duration_ms,
                    "llm_calls": node.llm_calls,
                    "tool_calls": node.tool_calls,
                    "input_tokens": node.input_tokens,
                    "output_tokens": node.output_tokens,
                    "llm_duration_ms": node.llm_duration_ms,
                    "tool_duration_ms": node.tool_duration_ms,
                    "overhead_ms": node.overhead_ms,
                }
                for name, node in self.node_metrics.items()
            },
            
            # Per-call details (for deep analysis)
            "llm_call_details": [
                {
                    "node": c.node,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "duration_ms": c.duration_ms,
                    "tokens_per_second": c.tokens_per_second,
                    "efficiency": c.token_efficiency,
                }
                for c in self.llm_calls
            ],
        }
    
    def print_summary(self):
        """Print a human-readable summary to console."""
        report = self.generate_report()
        
        print("\n" + "=" * 70)
        print("                    INSTRUMENTATION REPORT")
        print("=" * 70)
        
        print(f"\n📊 RUN OVERVIEW")
        print(f"   Model:          {report['model']}")
        print(f"   Candidates:     {report['candidates_count']}")
        print(f"   Total Duration: {report['latency']['total_seconds']:.1f}s")
        
        print(f"\n🔢 TOKEN METRICS")
        print(f"   Input Tokens:   {report['tokens']['input']:,}")
        print(f"   Output Tokens:  {report['tokens']['output']:,}")
        print(f"   Total Tokens:   {report['tokens']['total']:,}")
        print(f"   Efficiency:     {report['tokens']['efficiency_ratio']:.2f} (output/input)")
        print(f"   Per Symbol:     {report['tokens']['per_symbol_avg']:.0f} tokens avg")
        
        print(f"\n💰 COST METRICS")
        print(f"   Total Cost:     ${report['cost']['total_usd']:.4f}")
        print(f"   Per Symbol:     ${report['cost']['per_symbol_usd']:.4f}")
        print(f"   Cost Comparison:")
        for model, cost in sorted(report['cost']['comparison'].items(), key=lambda x: x[1]):
            marker = " ← current" if model == report['model'] else ""
            print(f"      {model:25} ${cost:.4f}{marker}")
        
        print(f"\n⏱️  LATENCY BREAKDOWN")
        print(f"   Total:          {report['latency']['total_ms']:.0f}ms ({report['latency']['total_seconds']:.1f}s)")
        print(f"   LLM Time:       {report['latency']['llm_ms']:.0f}ms ({report['latency']['breakdown_pct']['llm']:.1f}%)")
        print(f"   Tool Time:      {report['latency']['tool_ms']:.0f}ms ({report['latency']['breakdown_pct']['tool']:.1f}%)")
        print(f"   Overhead:       {report['latency']['overhead_ms']:.0f}ms ({report['latency']['breakdown_pct']['overhead']:.1f}%)")
        
        print(f"\n🚀 THROUGHPUT")
        print(f"   Tokens/Second:  {report['throughput']['tokens_per_second']:.1f}")
        print(f"   Symbols/Minute: {report['throughput']['symbols_per_minute']:.2f}")
        
        print(f"\n📞 CALL COUNTS")
        print(f"   LLM Calls:      {report['calls']['llm_total']}")
        print(f"   Tool Calls:     {report['calls']['tool_total']}")
        
        print(f"\n📍 PER-NODE BREAKDOWN")
        print(f"   {'Node':<20} {'Duration':>10} {'LLM Calls':>10} {'Tokens':>10}")
        print(f"   {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        for name, node in report['nodes'].items():
            tokens = node['input_tokens'] + node['output_tokens']
            print(f"   {name:<20} {node['duration_ms']:>8.0f}ms {node['llm_calls']:>10} {tokens:>10,}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, output_dir: Path) -> Path:
        """Save full report as JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"instrumentation_{self.run_id}.json"
        
        with open(report_path, "w") as f:
            json.dump(self.generate_report(), f, indent=2, default=str)
        
        return report_path
    
    def save_summary(self, output_dir: Path) -> Path:
        """Save human-readable summary as text."""
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"metrics_summary_{self.run_id}.txt"
        
        report = self.generate_report()
        
        lines = [
            "=" * 70,
            "              ETF TRADING ANALYST - METRICS SUMMARY",
            "=" * 70,
            "",
            f"Run ID:        {report['run_id']}",
            f"Model:         {report['model']}",
            f"Candidates:    {report['candidates_count']}",
            "",
            "TOKEN METRICS",
            "-" * 40,
            f"  Input Tokens:   {report['tokens']['input']:,}",
            f"  Output Tokens:  {report['tokens']['output']:,}",
            f"  Total Tokens:   {report['tokens']['total']:,}",
            f"  Efficiency:     {report['tokens']['efficiency_ratio']:.2f}",
            f"  Per Symbol:     {report['tokens']['per_symbol_avg']:.0f}",
            "",
            "COST METRICS",
            "-" * 40,
            f"  Total Cost:     ${report['cost']['total_usd']:.4f}",
            f"  Per Symbol:     ${report['cost']['per_symbol_usd']:.4f}",
            "",
            "  Model Comparison:",
        ]
        
        for model, cost in sorted(report['cost']['comparison'].items(), key=lambda x: x[1]):
            marker = " ← USED" if model == report['model'] else ""
            lines.append(f"    {model:25} ${cost:.4f}{marker}")
        
        lines.extend([
            "",
            "LATENCY METRICS",
            "-" * 40,
            f"  Total Duration: {report['latency']['total_seconds']:.1f}s",
            f"  LLM Time:       {report['latency']['llm_ms']:.0f}ms ({report['latency']['breakdown_pct']['llm']:.1f}%)",
            f"  Tool Time:      {report['latency']['tool_ms']:.0f}ms ({report['latency']['breakdown_pct']['tool']:.1f}%)",
            f"  Overhead:       {report['latency']['overhead_ms']:.0f}ms ({report['latency']['breakdown_pct']['overhead']:.1f}%)",
            "",
            "THROUGHPUT",
            "-" * 40,
            f"  Tokens/Second:  {report['throughput']['tokens_per_second']:.1f}",
            f"  Symbols/Minute: {report['throughput']['symbols_per_minute']:.2f}",
            "",
            "NODE BREAKDOWN",
            "-" * 40,
            f"  {'Node':<20} {'Duration':>10} {'LLM':>6} {'Tokens':>10}",
        ])
        
        for name, node in report['nodes'].items():
            tokens = node['input_tokens'] + node['output_tokens']
            lines.append(f"  {name:<20} {node['duration_ms']:>8.0f}ms {node['llm_calls']:>6} {tokens:>10,}")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        summary_path.write_text("\n".join(lines))
        return summary_path


# Global instrumenter instance
_instrumenter: Optional[Instrumenter] = None


def get_instrumenter() -> Instrumenter:
    """Get or create the global instrumenter."""
    global _instrumenter
    if _instrumenter is None:
        _instrumenter = Instrumenter()
    return _instrumenter


def reset_instrumenter():
    """Reset the global instrumenter."""
    global _instrumenter
    _instrumenter = Instrumenter()

"""
Logging and Traceability for ETF Trading Analyst.

This module provides comprehensive logging to ensure full traceability
of all analyst operations, LLM calls, and outputs.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnalystLogger:
    """
    Comprehensive logger for ETF analyst operations.
    
    Captures:
    - All LLM prompts and responses
    - Tool calls (web searches)
    - State transitions
    - Errors and warnings
    - Token usage
    - Timing information
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        
        # Log entries
        self.entries: List[Dict[str, Any]] = []
        
        # Setup Python logger for console output
        self.logger = logging.getLogger("analyst")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler if output_dir specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f"analyst_run_{self.run_id}.log"
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def log_run_start(self, candidates_count: int, model: str):
        """Log the start of an analysis run."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "run_start",
            "run_id": self.run_id,
            "candidates_count": candidates_count,
            "model": model,
        }
        self.entries.append(entry)
        self.logger.info(f"Run started: {self.run_id}")
        self.logger.info(f"Model: {model}")
        self.logger.info(f"Candidates: {candidates_count}")
    
    def log_node_enter(self, node_name: str, state_keys: List[str]):
        """Log entering a graph node."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "node_enter",
            "node": node_name,
            "state_keys": state_keys,
        }
        self.entries.append(entry)
        self.logger.debug(f"Entering node: {node_name}")
    
    def log_node_exit(self, node_name: str, duration_ms: float):
        """Log exiting a graph node."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "node_exit",
            "node": node_name,
            "duration_ms": duration_ms,
        }
        self.entries.append(entry)
        self.logger.debug(f"Exiting node: {node_name} ({duration_ms:.0f}ms)")
    
    def log_tool_call(self, tool_name: str, inputs: Dict[str, Any], outputs: Any):
        """Log a tool invocation (e.g., web search)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "tool_call",
            "tool": tool_name,
            "inputs": inputs,
            "outputs_summary": str(outputs)[:500],  # Truncate for readability
        }
        self.entries.append(entry)
        self.logger.debug(f"Tool call: {tool_name}")
    
    def log_llm_call(
        self, 
        node: str, 
        prompt: str, 
        response: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ):
        """Log an LLM invocation with full prompt and response."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "llm_call",
            "node": node,
            "model": model,
            "prompt": prompt,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.entries.append(entry)
        self.logger.debug(f"LLM call: {node} ({input_tokens}+{output_tokens} tokens)")
    
    def log_error(self, node: str, error: str, details: Optional[Dict] = None):
        """Log an error."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "error",
            "node": node,
            "error": error,
            "details": details or {},
        }
        self.entries.append(entry)
        self.logger.error(f"Error in {node}: {error}")
    
    def log_run_complete(self, usage: Dict[str, Any], results_summary: Dict[str, Any]):
        """Log run completion with final statistics."""
        duration = (datetime.now() - self.start_time).total_seconds()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": "run_complete",
            "run_id": self.run_id,
            "duration_seconds": duration,
            "usage": usage,
            "results_summary": results_summary,
        }
        self.entries.append(entry)
        self.logger.info(f"Run complete in {duration:.1f}s")
    
    def save_full_log(self, output_dir: Path) -> Path:
        """Save complete log as JSON for full traceability."""
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"analyst_trace_{self.run_id}.json"
        
        full_log = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "entries": self.entries,
        }
        
        with open(log_path, "w") as f:
            json.dump(full_log, f, indent=2, default=str)
        
        self.logger.info(f"Full trace saved: {log_path}")
        return log_path
    
    def get_llm_calls_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all LLM calls for audit."""
        return [e for e in self.entries if e["event"] == "llm_call"]


# Global logger instance
_analyst_logger: Optional[AnalystLogger] = None


def get_logger() -> Optional[AnalystLogger]:
    """Get the global analyst logger."""
    return _analyst_logger


def init_logger(output_dir: Optional[Path] = None) -> AnalystLogger:
    """Initialize the global analyst logger."""
    global _analyst_logger
    _analyst_logger = AnalystLogger(output_dir)
    return _analyst_logger


def generate_usage_report(
    usage: Dict[str, Any],
    model: str,
    output_dir: Path,
) -> Path:
    """
    Generate a standalone usage report file.
    
    This creates a human-readable report of token usage and costs.
    """
    from src.analyst.graph import UsageTracker
    
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "usage_report.txt"
    
    # Calculate costs for comparison
    pricing = UsageTracker.PRICING
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    lines = [
        "=" * 60,
        "ETF TRADING ANALYST - USAGE REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "TOKEN USAGE",
        "-" * 30,
        f"  Model Used:      {model}",
        f"  LLM Calls:       {usage.get('calls', 0)}",
        f"  Input Tokens:    {input_tokens:,}",
        f"  Output Tokens:   {output_tokens:,}",
        f"  Total Tokens:    {input_tokens + output_tokens:,}",
        "",
        "COST COMPARISON (for this run)",
        "-" * 30,
    ]
    
    for m, prices in sorted(pricing.items(), key=lambda x: x[1]["input"]):
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost
        marker = " <-- USED" if m == model else ""
        lines.append(f"  {m:25} ${total_cost:.4f}{marker}")
    
    lines.extend([
        "",
        "CALL DETAILS",
        "-" * 30,
    ])
    
    for detail in usage.get("call_details", []):
        node = detail.get("node", "unknown")
        inp = detail.get("input_tokens", 0)
        out = detail.get("output_tokens", 0)
        lines.append(f"  {node:25} {inp:>5} in / {out:>5} out")
    
    lines.extend([
        "",
        "=" * 60,
    ])
    
    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    
    return report_path

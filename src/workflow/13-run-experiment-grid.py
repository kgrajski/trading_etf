#!/usr/bin/env python3
"""
13-run-experiment-grid.py

Master experiment orchestrator that runs a structured grid of experiments
with kill criteria to efficiently explore the hypothesis space.

This implements the research grid defined in RESEARCH_LOG.md.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.workflow.workflow_utils import PROJECT_ROOT, setup_logging, workflow_script

# =============================================================================
# Research Grid Definition
# =============================================================================

EXPERIMENTS = [
    {
        "id": "exp011_regime_features",
        "name": "Regime as Feature",
        "hypothesis": "Adding regime indicators improves IC",
        "script": "12-run-regime-feature-experiment.py",
        "depends_on": None,
        "kill_if": {"ic_below": 0.02, "dir_acc_below": 0.50},
    },
    {
        "id": "exp012_classification",
        "name": "Classification in Best Regime",
        "hypothesis": "Classification achieves Dir.Acc > 55% when VIX=medium",
        "script": None,  # Will be created if exp011 passes
        "depends_on": "exp011_regime_features",
        "kill_if": {"dir_acc_below": 0.55},
    },
    {
        "id": "exp013_regime_specific",
        "name": "Regime-Specific Models",
        "hypothesis": "Separate models per regime outperform pooled",
        "script": None,
        "depends_on": "exp011_regime_features",
        "kill_if": {"ic_below": 0.05},
    },
    {
        "id": "exp014_momentum_baseline",
        "name": "Naive Momentum Baseline",
        "hypothesis": "Simple buy-last-weeks-winners beats ML",
        "script": None,  # Simple analysis
        "depends_on": None,
        "kill_if": None,
    },
]

# Kill criteria thresholds
GLOBAL_IC_THRESHOLD = 0.02
GLOBAL_DIR_ACC_THRESHOLD = 0.50
IMPROVEMENT_THRESHOLD = 0.02  # Must improve by 2% to continue branch

# Baseline from exp009
BASELINE = {
    "dir_acc": 0.509,
    "ic": 0.030,
}


def load_experiment_results(exp_id: str) -> dict | None:
    """Load results from a completed experiment."""
    exp_dir = PROJECT_ROOT / "experiments" / exp_id
    
    # Try weekly_metrics.csv first (rolling experiments)
    weekly_path = exp_dir / "weekly_metrics.csv"
    if weekly_path.exists():
        df = pd.read_csv(weekly_path)
        return {
            "dir_acc": df["directional_accuracy"].mean(),
            "ic": df["information_coefficient"].mean(),
            "n_weeks": len(df),
        }
    
    # Try test_results.json (static experiments)
    results_path = exp_dir / "test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            # Get best model
            best_model = max(results.items(), key=lambda x: x[1].get("information_coefficient", 0))
            return {
                "dir_acc": best_model[1].get("directional_accuracy", 0),
                "ic": best_model[1].get("information_coefficient", 0),
            }
    
    return None


def check_kill_criteria(results: dict, kill_if: dict | None) -> tuple[bool, str]:
    """
    Check if experiment should be killed based on criteria.
    
    Returns: (should_kill, reason)
    """
    if kill_if is None:
        return False, ""
    
    if "ic_below" in kill_if and results["ic"] < kill_if["ic_below"]:
        return True, f"IC {results['ic']:.3f} < {kill_if['ic_below']}"
    
    if "dir_acc_below" in kill_if and results["dir_acc"] < kill_if["dir_acc_below"]:
        return True, f"Dir.Acc {results['dir_acc']:.1%} < {kill_if['dir_acc_below']:.0%}"
    
    return False, ""


def check_improvement(results: dict, baseline: dict) -> tuple[bool, str]:
    """Check if there's meaningful improvement over baseline."""
    ic_delta = results["ic"] - baseline["ic"]
    dir_acc_delta = results["dir_acc"] - baseline["dir_acc"]
    
    if ic_delta > IMPROVEMENT_THRESHOLD or dir_acc_delta > IMPROVEMENT_THRESHOLD:
        return True, f"IC +{ic_delta:.3f}, Dir.Acc +{dir_acc_delta:.1%}"
    
    return False, f"IC +{ic_delta:.3f}, Dir.Acc +{dir_acc_delta:.1%} (below threshold)"


def run_experiment(exp: dict) -> dict:
    """Run a single experiment and return results."""
    exp_id = exp["id"]
    
    # Check if already completed
    results = load_experiment_results(exp_id)
    if results is not None:
        logging.info(f"[{exp_id}] Already completed, loading results")
        return {
            "status": "completed",
            "results": results,
            "skipped": True,
        }
    
    # Check dependencies
    if exp["depends_on"]:
        dep_results = load_experiment_results(exp["depends_on"])
        if dep_results is None:
            logging.warning(f"[{exp_id}] Dependency {exp['depends_on']} not completed")
            return {"status": "blocked", "reason": f"Waiting on {exp['depends_on']}"}
        
        # Check if dependency was killed
        dep_exp = next((e for e in EXPERIMENTS if e["id"] == exp["depends_on"]), None)
        if dep_exp:
            killed, reason = check_kill_criteria(dep_results, dep_exp.get("kill_if"))
            if killed:
                logging.info(f"[{exp_id}] Skipping - dependency killed: {reason}")
                return {"status": "skipped", "reason": f"Dependency killed: {reason}"}
    
    # Run the experiment
    if exp["script"] is None:
        logging.warning(f"[{exp_id}] No script defined yet")
        return {"status": "not_implemented"}
    
    script_path = PROJECT_ROOT / "src" / "workflow" / exp["script"]
    if not script_path.exists():
        logging.error(f"[{exp_id}] Script not found: {script_path}")
        return {"status": "error", "reason": "Script not found"}
    
    logging.info(f"[{exp_id}] Running {exp['script']}...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
        )
        
        if result.returncode != 0:
            logging.error(f"[{exp_id}] Script failed:\n{result.stderr}")
            return {"status": "error", "reason": result.stderr[:500]}
        
        # Load results
        results = load_experiment_results(exp_id)
        if results is None:
            return {"status": "error", "reason": "No results found after run"}
        
        # Check kill criteria
        killed, reason = check_kill_criteria(results, exp.get("kill_if"))
        
        return {
            "status": "killed" if killed else "completed",
            "results": results,
            "kill_reason": reason if killed else None,
        }
        
    except subprocess.TimeoutExpired:
        logging.error(f"[{exp_id}] Timeout after 30 minutes")
        return {"status": "timeout"}
    except Exception as e:
        logging.error(f"[{exp_id}] Exception: {e}")
        return {"status": "error", "reason": str(e)}


def update_research_log(grid_results: list[dict]) -> None:
    """Update RESEARCH_LOG.md with grid results."""
    log_path = PROJECT_ROOT / "RESEARCH_LOG.md"
    
    # Read existing log
    content = log_path.read_text()
    
    # Find the experiment queue section and update it
    lines = content.split("\n")
    new_lines = []
    in_queue = False
    
    for line in lines:
        if "## Experiment Queue" in line:
            in_queue = True
            new_lines.append(line)
            new_lines.append("")
            new_lines.append("| Priority | ID | Hypothesis | Status | Result |")
            new_lines.append("|----------|-----|------------|--------|--------|")
            
            for i, (exp, result) in enumerate(zip(EXPERIMENTS, grid_results), 1):
                status = result.get("status", "PENDING")
                if status == "completed" and "results" in result:
                    res = result["results"]
                    result_str = f"IC={res['ic']:.3f}, Dir={res['dir_acc']:.1%}"
                elif status == "killed":
                    result_str = result.get("kill_reason", "Killed")
                else:
                    result_str = "-"
                
                new_lines.append(
                    f"| {i} | {exp['id']} | {exp['hypothesis']} | {status.upper()} | {result_str} |"
                )
            continue
        
        if in_queue and line.startswith("| "):
            continue  # Skip old queue rows
        
        if in_queue and line.startswith("---"):
            in_queue = False
        
        if not in_queue:
            new_lines.append(line)
    
    # Update last updated timestamp
    new_content = "\n".join(new_lines)
    new_content = new_content.replace(
        "*Last updated: 2026-01-23*",
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
    )
    
    log_path.write_text(new_content)
    logging.info(f"Updated {log_path}")


@workflow_script("13-run-experiment-grid")
def main() -> None:
    """Run the experiment grid."""
    logging.info("=" * 60)
    logging.info("EXPERIMENT GRID RUNNER")
    logging.info("=" * 60)
    
    grid_results = []
    
    for exp in EXPERIMENTS:
        logging.info(f"\n{'='*60}")
        logging.info(f"EXPERIMENT: {exp['id']}")
        logging.info(f"Hypothesis: {exp['hypothesis']}")
        logging.info("=" * 60)
        
        result = run_experiment(exp)
        grid_results.append(result)
        
        # Log result
        status = result.get("status", "unknown")
        if status == "completed" and "results" in result:
            res = result["results"]
            logging.info(f"Result: IC={res['ic']:.3f}, Dir.Acc={res['dir_acc']:.1%}")
            
            # Check improvement over baseline
            improved, reason = check_improvement(res, BASELINE)
            if improved:
                logging.info(f"✓ Improvement: {reason}")
            else:
                logging.info(f"✗ No improvement: {reason}")
        
        elif status == "killed":
            logging.info(f"KILLED: {result.get('kill_reason', 'Unknown')}")
        
        else:
            logging.info(f"Status: {status}")
    
    # Update research log
    update_research_log(grid_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GRID SUMMARY")
    print("=" * 60)
    
    for exp, result in zip(EXPERIMENTS, grid_results):
        status = result.get("status", "?")
        if status == "completed" and "results" in result:
            res = result["results"]
            print(f"  {exp['id']}: IC={res['ic']:.3f}, Dir.Acc={res['dir_acc']:.1%}")
        else:
            print(f"  {exp['id']}: {status.upper()}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

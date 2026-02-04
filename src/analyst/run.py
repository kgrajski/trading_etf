#!/usr/bin/env python3
"""
Run the Agentic ETF Trading Analyst.

Usage:
    python -m src.analyst.run <candidates_csv>
    python -m src.analyst.run experiments/exp019_3_trades/2026-02-03/candidates.csv
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analyst.graph import (
    analyze_candidates, 
    UsageTracker,
    export_graph_visualization,
    print_graph_info,
)


def _generate_usage_footer(results: dict) -> str:
    """Generate HTML footer with usage stats and cost comparison."""
    usage = results.get("usage", {})
    model = results.get("model", "gpt-4o-mini")
    
    if not usage or usage.get("calls", 0) == 0:
        return ""
    
    total_tokens = usage.get("total_tokens", 0)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    # Calculate costs for all models
    pricing = UsageTracker.PRICING
    cost_rows = ""
    for m, prices in sorted(pricing.items()):
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        total_cost = input_cost + output_cost
        highlight = 'style="background: #e8f5e9; font-weight: bold;"' if m == model else ""
        cost_rows += f'<tr {highlight}><td>{m}</td><td>${total_cost:.4f}</td></tr>\n'
    
    return f'''
    <div class="usage-footer">
        <h3>API Usage</h3>
        <div class="usage-stats">
            <div><strong>Model:</strong> {model}</div>
            <div><strong>LLM Calls:</strong> {usage.get("calls", 0)}</div>
            <div><strong>Input Tokens:</strong> {input_tokens:,}</div>
            <div><strong>Output Tokens:</strong> {output_tokens:,}</div>
            <div><strong>Total Tokens:</strong> {total_tokens:,}</div>
        </div>
        <h4>Cost Comparison (for this run)</h4>
        <table class="cost-table">
            <tr><th>Model</th><th>Est. Cost</th></tr>
            {cost_rows}
        </table>
        <p class="note">* Highlighted row shows current model. Costs are estimates based on Jan 2025 pricing.</p>
    </div>
    <style>
        .usage-footer {{ 
            margin-top: 30px; 
            padding: 15px; 
            background: #f8f9fa; 
            border-radius: 8px;
            font-size: 0.9em;
        }}
        .usage-footer h3 {{ margin-top: 0; }}
        .usage-stats {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); 
            gap: 10px;
            margin-bottom: 15px;
        }}
        .cost-table {{ 
            border-collapse: collapse; 
            width: 100%; 
            max-width: 400px;
        }}
        .cost-table th, .cost-table td {{ 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: left; 
        }}
        .cost-table th {{ background: #e9ecef; }}
        .note {{ color: #666; font-size: 0.85em; margin-top: 10px; }}
    </style>
    '''


def _generate_review_section(review: dict) -> str:
    """Generate HTML for the reviewer's assessment section."""
    if not review or review.get("overall_assessment") == "ERROR":
        return ""
    
    assessment = review.get("overall_assessment", "N/A")
    confidence = review.get("confidence_in_analysis", 0)
    
    # Badge color based on assessment
    badge_class = {
        "APPROVE": "approve",
        "NEEDS_REVISION": "revision",
        "REJECT": "reject",
    }.get(assessment, "")
    
    # Build lists
    top_picks = review.get("top_picks", [])
    avoid_list = review.get("avoid_list", [])
    red_flags = review.get("red_flags", [])
    
    top_picks_html = "<ul>" + "".join([f"<li>⭐ {s}</li>" for s in top_picks]) + "</ul>" if top_picks else "<p>None specified</p>"
    avoid_html = "<ul>" + "".join([f"<li>⛔ {s}</li>" for s in avoid_list]) + "</ul>" if avoid_list else "<p>None specified</p>"
    red_flags_html = "<ul>" + "".join([f"<li>🚨 {f}</li>" for f in red_flags]) + "</ul>" if red_flags else "<p>None identified</p>"
    
    return f'''
    <div class="review-section">
        <h2>🔍 Senior Analyst Review (Reflection)</h2>
        <div class="review-header">
            <span class="review-badge {badge_class}">{assessment}</span>
            <span>Confidence: {confidence}/10</span>
        </div>
        <div class="review-content">
            <div class="review-box">
                <h4>Top Picks</h4>
                {top_picks_html}
            </div>
            <div class="review-box">
                <h4>Avoid List</h4>
                {avoid_html}
            </div>
            <div class="review-box">
                <h4>Red Flags</h4>
                {red_flags_html}
            </div>
        </div>
        <div class="review-notes">
            <strong>Reviewer Notes:</strong> {review.get("reviewer_notes", "No notes provided")}
        </div>
    </div>
    '''


def load_candidates_csv(csv_path: str) -> list:
    """Load candidates from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Convert to list of dicts
    candidates = df.to_dict(orient="records")
    
    return candidates


def save_results(results: dict, output_dir: Path) -> Path:
    """Save analysis results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "analyst_report.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return output_path


def generate_html_report(results: dict, candidates: list, output_dir: Path) -> Path:
    """Generate HTML report with conviction scores and review."""
    
    thematic = results.get("thematic_analysis", {})
    symbol_analyses = results.get("symbol_analyses", {})
    review = results.get("review_results", {}) or {}
    
    # Build theme cards
    theme_cards = ""
    for theme in thematic.get("themes", []):
        symbols_str = ", ".join(theme.get("symbols", []))
        outlook_color = {
            "favorable": "#28A745",
            "neutral": "#FFC107", 
            "unfavorable": "#DC3545"
        }.get(theme.get("mean_reversion_outlook", "neutral"), "#6c757d")
        
        theme_cards += f"""
        <div class="theme-card">
            <h3>{theme.get('name', 'Unknown Theme')}</h3>
            <div class="symbols">{symbols_str}</div>
            <p>{theme.get('narrative', '')}</p>
            <span class="outlook" style="background: {outlook_color}">
                {theme.get('mean_reversion_outlook', 'unknown').upper()}
            </span>
        </div>
        """
    
    # Build symbol cards
    symbol_cards = ""
    top_picks = review.get("top_picks", [])
    avoid_list = review.get("avoid_list", [])
    
    for candidate in candidates:
        symbol = candidate.get("symbol", "")
        analysis = symbol_analyses.get(symbol, {})
        
        conviction = analysis.get("conviction", 5)
        recommendation = analysis.get("recommendation", "HOLD")
        
        # Check if conviction was adjusted by reviewer
        was_adjusted = analysis.get("conviction_adjusted_by_reviewer", False)
        original_conviction = analysis.get("conviction_original", conviction)
        adjustment_reason = analysis.get("adjustment_reason", "")
        
        rec_color = {
            "BUY": "#28A745",
            "HOLD": "#FFC107",
            "AVOID": "#DC3545"
        }.get(recommendation, "#6c757d")
        
        conv_color = "#28A745" if conviction >= 7 else "#FFC107" if conviction >= 4 else "#DC3545"
        
        # Add badge if in top picks or avoid list
        badge = ""
        if symbol in top_picks:
            badge = '<span class="badge top-pick">⭐ TOP PICK</span>'
        elif symbol in avoid_list:
            badge = '<span class="badge avoid">⛔ AVOID</span>'
        
        # Show adjustment info
        adjustment_html = ""
        if was_adjusted:
            direction = "↓" if conviction < original_conviction else "↑"
            adjustment_html = f'''
            <div class="adjustment">
                <strong>Reviewer Adjusted:</strong> {original_conviction} → {conviction} {direction}
                <br><em>{adjustment_reason}</em>
            </div>
            '''
        
        pros = "".join([f"<li>✅ {p}</li>" for p in analysis.get("pros", [])])
        cons = "".join([f"<li>⚠️ {c}</li>" for c in analysis.get("cons", [])])
        
        citations = ""
        for cite in analysis.get("citations", []):
            citations += f'<a href="{cite.get("url", "#")}" target="_blank">{cite.get("title", "Source")[:50]}...</a><br>'
        
        symbol_cards += f"""
        <div class="symbol-card" id="card-{symbol}">
            <div class="symbol-header">
                <h3>{symbol}</h3>
                <span class="conviction" style="background: {conv_color}">{conviction}/10</span>
                <span class="recommendation" style="background: {rec_color}">{recommendation}</span>
                {badge}
            </div>
            <div class="symbol-name">{candidate.get('etf_name', '')[:50]}</div>
            <div class="return">Weekly Return: {candidate.get('pct_return', 0):.1f}%</div>
            {adjustment_html}
            <p class="narrative">{analysis.get('narrative', 'No analysis available')}</p>
            <div class="pros-cons">
                <div class="pros"><strong>Pros:</strong><ul>{pros if pros else '<li>None identified</li>'}</ul></div>
                <div class="cons"><strong>Cons:</strong><ul>{cons if cons else '<li>None identified</li>'}</ul></div>
            </div>
            <div class="key-risk"><strong>Key Risk:</strong> {analysis.get('key_risk', 'Unknown')}</div>
            <div class="citations"><strong>Sources:</strong><br>{citations if citations else 'No sources'}</div>
        </div>
        """
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ETF Trading Analyst Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif; 
            margin: 0; padding: 20px; 
            background: #f5f5f5; 
        }}
        h1 {{ margin-bottom: 5px; }}
        .timestamp {{ color: #666; margin-bottom: 20px; }}
        .summary {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 20px;
            border-left: 4px solid #2E86AB;
        }}
        .summary h2 {{ margin-top: 0; }}
        .themes {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 15px; 
            margin-bottom: 30px; 
        }}
        .theme-card {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .theme-card h3 {{ margin-top: 0; color: #2E86AB; }}
        .theme-card .symbols {{ color: #666; font-size: 0.9em; margin-bottom: 10px; }}
        .theme-card .outlook {{ 
            display: inline-block;
            color: white; 
            padding: 2px 8px; 
            border-radius: 4px; 
            font-size: 0.8em;
        }}
        .symbols-grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 15px; 
        }}
        .symbol-card {{ 
            background: white; 
            padding: 15px; 
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .symbol-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }}
        .symbol-header h3 {{ margin: 0; }}
        .conviction, .recommendation {{ 
            color: white; 
            padding: 2px 8px; 
            border-radius: 4px; 
            font-size: 0.85em;
            font-weight: bold;
        }}
        .symbol-name {{ color: #666; font-size: 0.9em; }}
        .return {{ font-weight: bold; color: #DC3545; margin: 5px 0; }}
        .narrative {{ margin: 10px 0; line-height: 1.5; }}
        .pros-cons {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 10px 0; }}
        .pros-cons ul {{ margin: 5px 0; padding-left: 20px; font-size: 0.9em; }}
        .key-risk {{ background: #fff3cd; padding: 8px; border-radius: 4px; font-size: 0.9em; margin: 10px 0; }}
        .citations {{ font-size: 0.8em; color: #666; margin-top: 10px; }}
        .citations a {{ color: #2E86AB; }}
        .errors {{ background: #f8d7da; padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
        /* Review section styles */
        .review-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .review-section h2 {{ margin-top: 0; color: white; }}
        .review-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .review-badge {{
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .review-badge.approve {{ background: #28A745; }}
        .review-badge.revision {{ background: #FFC107; color: #333; }}
        .review-badge.reject {{ background: #DC3545; }}
        .review-content {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .review-box {{
            background: rgba(255,255,255,0.1);
            padding: 12px;
            border-radius: 6px;
        }}
        .review-box h4 {{ margin: 0 0 8px 0; font-size: 0.9em; opacity: 0.9; }}
        .review-box ul {{ margin: 0; padding-left: 20px; }}
        .review-notes {{
            margin-top: 15px;
            font-style: italic;
            opacity: 0.9;
        }}
        /* Badge styles */
        .badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .badge.top-pick {{ background: #FFD700; color: #333; }}
        .badge.avoid {{ background: #DC3545; color: white; }}
        /* Adjustment styles */
        .adjustment {{
            background: #e3f2fd;
            border-left: 3px solid #2196F3;
            padding: 8px;
            margin: 8px 0;
            font-size: 0.9em;
        }}
        .adjustment em {{ color: #666; }}
    </style>
</head>
<body>
    <h1>🤖 ETF Trading Analyst Report</h1>
    <div class="timestamp">Generated: {results.get('timestamp', datetime.now().isoformat())}</div>
    
    <div class="summary">
        <h2>Market Summary</h2>
        <p>{thematic.get('summary', 'No summary available')}</p>
        <strong>Overall Sentiment:</strong> {thematic.get('overall_sentiment', 'unknown').upper()}
    </div>
    
    <h2>Thematic Analysis</h2>
    <div class="themes">
        {theme_cards if theme_cards else '<p>No themes identified</p>'}
    </div>
    
    {_generate_review_section(review)}
    
    <h2>Symbol Analysis ({len(candidates)} candidates)</h2>
    <div class="symbols-grid">
        {symbol_cards}
    </div>
    
    {f'<div class="errors"><strong>Errors:</strong> {", ".join(results.get("errors", []))}</div>' if results.get("errors") else ''}
    
    {_generate_usage_footer(results)}
</body>
</html>
"""
    
    output_path = output_dir / "analyst_report.html"
    output_path.write_text(html)
    
    return output_path


def main():
    """
    Main entry point for the ETF Trading Analyst.
    
    Usage:
        python -m src.analyst.run <candidates_csv>     # Run full analysis
        python -m src.analyst.run --graph              # Export graph visualization
        python -m src.analyst.run --info               # Print graph structure info
    """
    import subprocess
    import platform
    
    # Handle special flags
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--graph":
            # Export graph visualization only
            output_path = sys.argv[2] if len(sys.argv) > 2 else "analyst_graph.png"
            print("Exporting graph visualization...")
            results = export_graph_visualization(output_path)
            print(f"\nGraph exported:")
            for key, path in results.items():
                if path:
                    print(f"  {key}: {path}")
            if results.get("png") and platform.system() == "Darwin":
                subprocess.run(["open", results["png"]])
            return
        
        if sys.argv[1] == "--info":
            # Print graph info only
            print_graph_info()
            return
    
    # Normal operation: run analysis
    if len(sys.argv) < 2:
        print("Usage: python -m src.analyst.run <candidates_csv>")
        print("       python -m src.analyst.run --graph [output.png]  # Export graph diagram")
        print("       python -m src.analyst.run --info                # Show graph structure")
        print("")
        print("Example: python -m src.analyst.run experiments/exp019_3_trades/2026-02-03/candidates.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Determine output directory
    output_dir = csv_path.parent
    
    print(f"Loading candidates from: {csv_path}")
    candidates = load_candidates_csv(csv_path)
    print(f"Loaded {len(candidates)} candidates")
    print(f"Output directory: {output_dir}")
    
    # Run analysis (with logging enabled)
    results = analyze_candidates(candidates, output_dir=output_dir)
    
    # Save JSON results
    json_path = save_results(results, output_dir)
    print(f"\nJSON saved: {json_path}")
    
    # Generate HTML report
    html_path = generate_html_report(results, candidates, output_dir)
    print(f"HTML saved: {html_path}")
    
    # Export graph visualization alongside report
    graph_path = output_dir / "analyst_graph.png"
    graph_results = export_graph_visualization(str(graph_path))
    
    # Print summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    thematic = results.get("thematic_analysis", {})
    print(f"\nOverall Sentiment: {thematic.get('overall_sentiment', 'unknown').upper()}")
    print(f"Summary: {thematic.get('summary', 'N/A')}")
    
    print("\nConviction Scores:")
    for symbol, analysis in results.get("symbol_analyses", {}).items():
        conv = analysis.get("conviction", "?")
        rec = analysis.get("recommendation", "?")
        print(f"  {symbol}: {conv}/10 ({rec})")
    
    # Summary of all generated files
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)
    print(f"  Report (HTML):    {html_path}")
    print(f"  Report (JSON):    {json_path}")
    if graph_results.get("png"):
        print(f"  Graph (PNG):      {graph_results['png']}")
    if graph_results.get("mermaid"):
        print(f"  Graph (Mermaid):  {graph_results['mermaid']}")
    
    # Log files from analysis
    for log_name, log_path in results.get("log_files", {}).items():
        print(f"  {log_name.replace('_', ' ').title()}: {log_path}")
    
    print("=" * 60)
    
    # Open HTML report
    if platform.system() == "Darwin":
        subprocess.run(["open", str(html_path)])
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", str(html_path)])
    
    print(f"\n✅ Report opened in browser")


if __name__ == "__main__":
    main()

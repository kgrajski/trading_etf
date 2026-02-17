#!/bin/bash
# =============================================================================
# Weekly Data Update Script
# =============================================================================
#
# This script performs a complete data refresh:
# 1. Fetches new daily data from Alpaca (incremental mode)
# 2. Analyzes data coverage (sweet spot analysis)
# 3. Generates ETF retrospective report
# 4. Regenerates weekly features
# 5. Regenerates visualizations
# 6. Rebuilds the feature matrix
#
# SCHEDULE:
#   Recommended: Saturday 8am (after Friday data is fully available)
#   Alternative: Sunday morning (safer for data availability)
#
# USAGE:
#   ./scripts/weekly_update.sh
#   ./scripts/weekly_update.sh 2>&1 | tee logs/update_$(date +%Y%m%d).log
#
# =============================================================================

set -e  # Exit immediately on error

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================================================="
echo " Weekly Data Update"
echo " Started: $(date)"
echo " Project: ${PROJECT_ROOT}"
echo "============================================================================="
echo

# Change to project root
cd "${PROJECT_ROOT}"

# Activate virtual environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "${LOG_PREFIX} Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
else
    echo -e "${RED}${LOG_PREFIX} ERROR: Virtual environment not found at ${VENV_PATH}${NC}"
    exit 1
fi

# Check Python
echo "${LOG_PREFIX} Python: $(python --version)"
echo

# Step 1: Fetch new daily data
echo "============================================================================="
echo " Step 1: Fetch Daily Data (Incremental Mode)"
echo "============================================================================="
python src/workflow/pipeline/02-fetch-daily-data.py
echo

# Step 2: Analyze data coverage
echo "============================================================================="
echo " Step 2: Analyze Data Coverage"
echo "============================================================================="
python src/workflow/pipeline/02a-analyze-data-coverage.py
echo

# Step 3: ETF retrospective report
echo "============================================================================="
echo " Step 3: ETF Retrospective Report"
echo "============================================================================="
python src/workflow/pipeline/02b-etf-retrospective.py
echo

# Step 4: Generate weekly features
echo "============================================================================="
echo " Step 4: Generate Weekly Features"
echo "============================================================================="
python src/workflow/pipeline/03-generate-features.py
echo

# Step 5a: Generate visualizations
echo "============================================================================="
echo " Step 5a: Generate Visualizations"
echo "============================================================================="
python src/workflow/pipeline/04-visualize-features.py
echo

# Step 5b: Scatter visualizations
echo "============================================================================="
echo " Step 5b: Scatter Visualizations"
echo "============================================================================="
python src/workflow/pipeline/04b-visualize-scatter.py
echo

# Step 6: Build feature matrix
echo "============================================================================="
echo " Step 6: Build Feature Matrix"
echo "============================================================================="
python src/workflow/pipeline/05-build-feature-matrix.py
echo

# Step 7: Generate trade candidates (pre-production)
echo "============================================================================="
echo " Step 7: Generate Trade Candidates (pre-production)"
echo "============================================================================="
python src/workflow/pipeline/21b-gen-new-trades.py
echo

# Step 8: Research trade candidates with AI-enriched dashboard (19.3)
echo "============================================================================="
echo " Step 8a: Generate Research Trade Candidates"
echo "============================================================================="
python src/workflow/research/19.3-gen-trades.py
echo

# Find today's candidates CSV for the AI analyst
TODAY=$(date +%Y-%m-%d)
CANDIDATES_CSV="experiments/exp019_3_trades/${TODAY}/candidates.csv"

if [ -f "${CANDIDATES_CSV}" ]; then
    echo "============================================================================="
    echo " Step 8b: Run AI Analyst on Research Candidates"
    echo "============================================================================="
    python -m src.analyst.run "${CANDIDATES_CSV}" || echo -e "${YELLOW}  AI analyst failed (non-fatal) — continuing without AI enrichment${NC}"
    echo

    echo "============================================================================="
    echo " Step 8c: Rebuild Research Dashboard with AI Analysis"
    echo "============================================================================="
    python src/workflow/research/19.3-gen-trades.py
    echo
else
    echo -e "${YELLOW}  No candidates generated — skipping AI analyst${NC}"
    echo
fi

# Done
echo "============================================================================="
echo -e "${GREEN} Weekly Update Complete!${NC}"
echo " Finished: $(date)"
echo "============================================================================="

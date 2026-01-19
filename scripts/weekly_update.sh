#!/bin/bash
# =============================================================================
# Weekly Data Update Script
# =============================================================================
#
# This script performs a complete data refresh:
# 1. Fetches new daily data from Alpaca (incremental mode)
# 2. Regenerates weekly features
# 3. Regenerates visualizations
# 4. Rebuilds the feature matrix
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
python src/workflow/02-fetch-daily-data.py
echo

# Step 2: Generate weekly features
echo "============================================================================="
echo " Step 2: Generate Weekly Features"
echo "============================================================================="
python src/workflow/03-generate-features.py
echo

# Step 3: Generate visualizations
echo "============================================================================="
echo " Step 3: Generate Visualizations"
echo "============================================================================="
python src/workflow/04-visualize-features.py
echo

# Step 4: Build feature matrix
echo "============================================================================="
echo " Step 4: Build Feature Matrix"
echo "============================================================================="
python src/workflow/05-build-feature-matrix.py
echo

# Done
echo "============================================================================="
echo -e "${GREEN} Weekly Update Complete!${NC}"
echo " Finished: $(date)"
echo "============================================================================="

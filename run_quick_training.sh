#!/bin/bash
#
# Quick Training Pipeline
# ========================
# Same as full training but uses only 5 subjects for fast testing
#
# Usage: ./run_quick_training.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS_FILE="$PROJECT_DIR/requirements-lock.txt"
DATASET_DIR="${DATASET_DIR:-$PROJECT_DIR/../datasets/openneuro/ds006623}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

require_supported_python() {
    local interpreter="$1"
    "$interpreter" - <<'PY'
import sys

required = (3, 11)
current = sys.version_info[:2]
if current < required:
    raise SystemExit(
        f"Python {required[0]}.{required[1]}+ is required for this release; "
        f"found {current[0]}.{current[1]}"
    )
PY
}

echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN} Quick Test Training (5 subjects only)${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}Checking Python version...${NC}"
if require_supported_python "$PYTHON_BIN"; then
    echo -e "${GREEN}Python 3.11+ detected${NC}"
else
    echo -e "${RED}Python 3.11+ is required for this release${NC}"
    exit 1
fi

# Activate venv (create if missing)
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Check if required packages are installed
if python -c "import numpy, sklearn, xgboost, pandas, imblearn" 2>/dev/null; then
    echo -e "${GREEN}Dependencies already installed${NC}"
else
    echo -e "${YELLOW}Installing dependencies into venv...${NC}"
    python -m pip install --no-cache-dir -r "$REQUIREMENTS_FILE"
fi

# Check dataset
if [ ! -d "$DATASET_DIR/derivatives/xcp_d_without_GSR_bandpass_output" ]; then
    echo -e "${YELLOW}Downloading dataset first...${NC}"
    python src/download_dataset.py --output-dir "$DATASET_DIR" --max-subjects 5
fi

echo -e "${CYAN}Running training on first 5 subjects...${NC}"
echo -e "${YELLOW}Estimated time: 2-3 minutes${NC}"
echo ""

# Run quick training (5 subjects only)
python src/train.py --max-subjects 5 2>&1 | tee "quick_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo -e "${GREEN}✓ Quick training complete!${NC}"
echo -e "${CYAN}For full training (25 subjects): ${NC}${GREEN}./run_full_training.sh${NC}"

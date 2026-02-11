#!/usr/bin/env bash
# Run test suite for QM Multi Agent System
# Usage: bash scripts/run_tests.sh [pytest-args]

set -euo pipefail

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
fi

echo "=== Running QM System Tests ==="
echo "Python: $(python --version)"
echo ""

# Default: verbose with short tracebacks
PYTEST_ARGS="${*:--v --tb=short}"

python -m pytest tests/ $PYTEST_ARGS

echo ""
echo "=== Tests complete ==="

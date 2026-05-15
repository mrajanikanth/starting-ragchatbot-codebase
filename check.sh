#!/usr/bin/env bash
# Run code quality checks. Pass --fix to auto-format instead of just checking.
set -euo pipefail

FIX=false
for arg in "$@"; do
    [[ "$arg" == "--fix" ]] && FIX=true
done

if $FIX; then
    echo "==> Formatting with black..."
    uv run black backend/ main.py
else
    echo "==> Checking formatting with black..."
    uv run black --check backend/ main.py
fi

echo ""
echo "==> Running tests..."
uv run pytest backend/tests/ -v

echo ""
echo "All checks passed."
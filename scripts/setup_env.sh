#!/usr/bin/env bash
# =====================================================================
# One-command environment setup (macOS / Linux / Git-Bash)
# Creates an ISOLATED virtual environment in .venv so this project's
# packages never disturb your other projects or global Python.
#
# Usage (from the project root):
#   bash scripts/setup_env.sh
# =====================================================================
set -e

echo "==> Creating virtual environment in .venv ..."
python -m venv .venv

# venv python path differs between Windows (Git-Bash) and Unix
if [ -f ".venv/Scripts/python.exe" ]; then
    PY=".venv/Scripts/python.exe"
else
    PY=".venv/bin/python"
fi

echo "==> Upgrading pip ..."
"$PY" -m pip install --upgrade pip

echo "==> Installing dependencies (requirements.txt) ..."
"$PY" -m pip install -r requirements.txt

echo ""
echo "Done. Activate the environment with:"
echo "    source .venv/bin/activate        # macOS/Linux"
echo "    source .venv/Scripts/activate    # Git-Bash on Windows"
echo "Then run:"
echo "    python scripts/run_week1_2_pipeline.py --demo"

#!/bin/bash
# Wrapper script for running single_query.py with proper environment isolation
# This script is called by the gateway to avoid subprocess networking issues

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the agent directory
cd "$SCRIPT_DIR"

# Activate the virtual environment
source .venv/bin/activate

# Ensure unbuffered Python output
export PYTHONUNBUFFERED=1

# Run the query script with all passed arguments
exec python scripts/single_query.py "$@"

#!/bin/bash
# Interactive chat using Modal Qwen-7B endpoint
# Usage: ./scripts/chat_modal.sh [--verbose]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$AGENT_DIR"
source .venv/bin/activate

echo "Starting interactive chat with Modal Qwen-7B..."
echo ""

python scripts/interactive_auto_search.py \
    --config workflows/auto_search_modal_deep.yaml \
    --dataset-name long_form \
    "$@"

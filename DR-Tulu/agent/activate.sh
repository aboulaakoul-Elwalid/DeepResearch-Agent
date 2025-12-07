#!/bin/bash
# Activate DR-Tulu agent environment with API keys

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Python venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Load .env file if it exists
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(cat "$SCRIPT_DIR/.env" | grep -v '^#' | xargs)
    echo "✓ Loaded API keys from .env"
else
    echo "⚠ Warning: .env file not found. Please create it with your API keys."
    echo "  See .env.example for template"
fi

# Ensure Arabic Books MCP tool env vars are set (fallback if not in .env)
export ARABIC_BOOKS_CHROMA_PATH="${ARABIC_BOOKS_CHROMA_PATH:-/home/elwalid/projects/parallax_project/chroma_db}"
export ARABIC_BOOKS_COLLECTION="${ARABIC_BOOKS_COLLECTION:-arabic_books}"

echo "✓ DR-Tulu environment activated"
echo ""
echo "Available commands:"
echo "  chat_modal            - Interactive chat with Modal Qwen-7B (recommended)"
echo "  launch_chat           - Run interactive demo"
echo "  mcp_server            - Start MCP backend server"
echo "  mcp_http              - Start MCP backend over HTTP (for Open WebUI External Tools)"
echo "  auto_search           - Run auto-search workflow"
echo ""
echo "Quick start:"
echo "  chat_modal            # Uses Modal GPU endpoint"
echo "  python scripts/interactive_auto_search.py -c workflows/auto_search_modal_deep.yaml"
echo ""

# Create convenience functions
chat_modal() {
    bash scripts/chat_modal.sh "$@"
}

launch_chat() {
    python scripts/launch_chat.py "$@"
}

mcp_server() {
    MCP_CACHE_DIR=".cache-$(hostname)" python -m dr_agent.mcp_backend.main --port 8000
}

mcp_http() {
    MCP_CACHE_DIR=".cache-$(hostname)" PATH="${PATH:-/mcp}" PORT="${PORT:-8765}" \
      bash "$SCRIPT_DIR/scripts/mcp_http.sh"
}

auto_search() {
    bash scripts/auto_search.sh
}

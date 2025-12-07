#!/usr/bin/env bash
# Start the MCP backend over HTTP so Open WebUI (or other MCP clients) can
# consume tools like `search_arabic_books` via a streamable HTTP endpoint.
#
# Usage:
#   ./scripts/mcp_http.sh            # binds 0.0.0.0:8765, path /mcp
#   PORT=9000 PATH=/mcp ./scripts/mcp_http.sh
#
# Requires:
#   - .env with ARABIC_BOOKS_CHROMA_PATH / ARABIC_BOOKS_COLLECTION if using the Arabic RAG tool
#   - .venv activated (or run via activate.sh which exports .env)

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

: "${PORT:=8765}"
: "${PATH:=/mcp}"

cd "${ROOT_DIR}"

exec python -m dr_agent.mcp_backend.main \
  --transport http \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --path "${PATH}" \
  --log-level info

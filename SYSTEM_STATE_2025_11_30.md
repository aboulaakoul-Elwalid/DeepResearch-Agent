# Parallax System State - 2025-11-30

## Executive Summary

**Status**: ✅ FULLY WORKING - All components operational

The Parallax system is fully operational with Open WebUI + DR-Tulu Gateway + Gemini 2.5 Flash. DR-Tulu successfully executes research tools (ChromaDB search, Google search, Exa search) and returns cited answers.

## System Architecture

```
User Browser (localhost:3005)
    ↓
Open WebUI (Docker, port 3005)
    ↓
DR-Tulu Gateway (Python, localhost:3001)
    ├─ Route: dr-tulu model → Agent Workflow
    ├─ Route: Qwen model → Direct inference to Modal
    └─ Route: gemini/* → Fallback (disabled)
    ↓
Modal Qwen Endpoint (https://aboulaakoul-elwalid...modal.run/v1)
    ↓
Responses (SSE streaming)
```

## Component Status

### ✅ Working

1. **Open WebUI Container**
   - Running: Yes
   - Port: 3005
   - Status: Healthy
   - Access: http://localhost:3005

2. **DR-Tulu Gateway**
   - Running: Yes
   - Port: 3001
   - Status: Healthy, accepting connections
   - Startup: See "Starting the Server" section below

3. **Gemini 2.5 Flash (Search Agent)**
   - Status: ✅ Working
   - Model: gemini/gemini-2.5-flash
   - API: Google AI API via LiteLLM
   - Tool calling: ✅ Follows XML instructions correctly

4. **ChromaDB (Arabic Books Search)**
   - Status: ✅ Working
   - Location: `/home/elwalid/projects/parallax_project/tmp_chroma/`
   - Tool: `search_arabic_books`
   - Contains: ~24,000 book embeddings

5. **Web Search Tools**
   - google_search: ✅ Working (via Serper API)
   - exa_search: ✅ Working (via Exa API)
   - browse_webpage: ✅ Working (via Jina API)

6. **Streaming**
   - SSE format: ✅ Correct
   - Tool calls: ✅ Streamed to client
   - Response chunks: ✅ Streaming properly
   - Open WebUI integration: ✅ Displays responses

7. **Model Routing**
   - Gateway model discovery: ✅ Working
   - Model selector in Open WebUI: ✅ Shows available models
   - Model switching: ✅ Can select different models

## Configuration

### Open WebUI Docker Command
```bash
docker run -d \
  --name open-webui \
  -p 3005:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  -e ENABLE_RAG=false \
  -e ENABLE_RAG_HYBRID_SEARCH=false \
  -e EMBEDDING_MODEL=none \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:3001/v1 \
  -e OPENAI_API_KEYS=dummy-key \
  -e ENABLE_SIGNUP=true \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

### DR-Tulu Gateway Startup

**IMPORTANT**: Must install dr_agent package and set environment variables correctly.

```bash
# 1. Install dr_agent package (one-time)
pip install -e /home/elwalid/projects/parallax_project/DR-Tulu/agent/

# 2. Start the server
export PYTHONPATH="/home/elwalid/projects/parallax_project/DR-Tulu/agent:$PYTHONPATH"
cd /home/elwalid/projects/parallax_project
set -a; source DR-Tulu/agent/.env; set +a
export GEMINI_API_KEY="$GOOGLE_AI_API_KEY"
python dr_tulu_agent_server.py &
```

### Required Environment Variables (DR-Tulu/agent/.env)

```bash
GOOGLE_AI_API_KEY=xxx      # For Gemini via LiteLLM
GEMINI_API_KEY=xxx         # Must match GOOGLE_AI_API_KEY
SERPER_API_KEY=xxx         # For Google search
EXA_API_KEY=xxx            # For Exa search
JINA_API_KEY=xxx           # For webpage browsing
```

### Critical Configuration: auto_search_deep.yaml
**Location**: `DR-Tulu/agent/workflows/auto_search_deep.yaml`

**Key Changes Made**:
```yaml
# Changed from Modal Qwen to Gemini (Qwen didn't follow XML tool instructions)
search_agent_model_name: "gemini/gemini-2.5-flash"
search_agent_max_tool_calls: 12
```

## Available Models

1. **dr-tulu** (Full research agent with tools)
   - Model selector: Shows as "dr-tulu"
   - Backend: Gemini 2.5 Flash via LiteLLM
   - Tools: search_arabic_books, google_search, exa_search, browse_webpage
   - Capability: Full research with citations

2. **Qwen/Qwen2.5-0.5B-Instruct** (Pure inference, proxied to Modal)
   - Model selector: Shows full name
   - Backend: Modal Qwen direct
   - Capability: Chat/completion only (no tools)

3. **gemini/gemini-2.5-flash** (Direct Gemini, no tools)
   - Backend: Google AI API
   - Capability: Fast chat without research tools

## Documentation Files

- `docs/dr_tulu_modal_qwen_working.md` - Working solution guide
- `docs/qwen_modal_setup.md` - Complete setup instructions
- `docs/dr_tulu_litellm_issue.md` - Technical debugging reference
- `docs/stuff_updates_openwebui.md` - Initial Open WebUI notes

## Verification Tests

**Test 1: Direct curl to gateway (with tools)**
```bash
curl -s -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dr-tulu","stream":true,"messages":[{"role":"user","content":"What are the best books on Tawhid?"}]}'
```
Expected: tool_calls for search_arabic_books and google_search, then final answer with citations
Actual: ✅ Working correctly

**Test 2: Models endpoint**
```bash
curl -s http://localhost:3001/v1/models
```
Expected: List of available models
Actual: ✅ Returns dr-tulu, Qwen, gemini models

## Recovery / Restart

If system is down:

```bash
# 1. Kill old processes
pkill -f "dr_tulu_agent_server"

# 2. Start DR-Tulu Gateway
export PYTHONPATH="/home/elwalid/projects/parallax_project/DR-Tulu/agent:$PYTHONPATH"
cd /home/elwalid/projects/parallax_project
set -a; source DR-Tulu/agent/.env; set +a
export GEMINI_API_KEY="$GOOGLE_AI_API_KEY"
python dr_tulu_agent_server.py > /tmp/dr_tulu_server.log 2>&1 &

sleep 5

# 3. Start Open WebUI if needed
docker start open-webui  # or docker run command above

# 4. Verify
curl -s http://localhost:3001/v1/models | jq .
curl -s http://localhost:3005/ | head -20
```

## Issues Resolved

### 1. dr_agent Module Import Error
**Problem**: MCP subprocess couldn't import `dr_agent`
**Solution**: Install as editable package: `pip install -e DR-Tulu/agent/`

### 2. Modal Qwen Not Following Tool Instructions
**Problem**: Qwen 8B output free-form text like "Call tool: exa_search" instead of proper XML
**Solution**: Switched to Gemini 2.5 Flash which follows instructions correctly

### 3. Missing Dependencies
**Problem**: Various import errors (fastapi, litellm, fastmcp, cohere)
**Solution**: Installed all required packages

### 4. Environment Variables Not Loaded
**Problem**: Gateway didn't load API keys from .env file
**Solution**: Added python-dotenv loading and explicit GEMINI_API_KEY export

## System Restart Checklist

- [x] DR-Tulu gateway running on port 3001
- [x] Open WebUI running on port 3005
- [x] Can access http://localhost:3005
- [x] Can log in to Open WebUI
- [x] Model selector shows "dr-tulu"
- [x] Can send messages and get responses
- [x] Responses stream correctly
- [x] Tool calls appear in responses
- [x] ChromaDB search working
- [x] Google search working

---

**Generated**: 2025-11-30 22:55
**Status**: ✅ Fully Working
**Key Fix**: Switched from Modal Qwen to Gemini 2.5 Flash for tool calling

# Qwen Modal Setup & DR-Tulu Integration Status

## Current Architecture

```
Browser (localhost:3005)
    ↓
Open WebUI (Docker container)
    ↓
DR-Tulu Gateway (localhost:3001/v1)
    ├── Route 1: Qwen/Qwen2.5-0.5B-Instruct → Modal endpoint ✅
    └── Route 2: dr-tulu model → Local agent workflow ⚠️
```

## Working ✅

### Modal Qwen Inference
- **Status**: Fully working
- **Endpoint**: `https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1`
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Test**: `curl -X POST http://localhost:3001/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","stream":true,"messages":[{"role":"user","content":"hi"}]}'`
- **From Open WebUI**: Fully functional, streams responses correctly

### Open WebUI Integration
- **Status**: Operational
- **URL**: `http://localhost:3005`
- **Connected to**: `http://host.docker.internal:3001/v1` (from container)
- **Models visible**: `Qwen/Qwen2.5-0.5B-Instruct` appears in model selector
- **Functionality**: Chat, streaming, message history all working

## Issues ⚠️

### DR-Tulu Local Agent
- **Problem**: `RetryError` when attempting to use `dr-tulu` model
- **Root cause**: Workflow attempts to call Qwen via Modal, but `litellm` isn't properly configured
- **Error detail**: "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
- **Attempted fix**: Updated `auto_search_deep.yaml` to:
  - Use `Qwen/Qwen2.5-0.5B-Instruct` as reasoning model
  - Set `search_agent_provider: "openai"`
  - Set base_url to `null` to use env variables
  - Restarted gateway with `OPENAI_API_BASE_URL` and `OPENAI_API_KEY` env vars
- **Status**: Still failing - needs deeper debugging into DR-Tulu's `dr_agent/client.py` litellm integration

## Workaround: Use Qwen Instead

Until DR-Tulu's internal agent workflow is fixed, use Qwen model directly:

### In Open WebUI:
1. Select model: `Qwen/Qwen2.5-0.5B-Instruct`
2. Ask questions normally
3. Responses stream correctly

**Note**: Qwen doesn't have tool capabilities like the original DR-Tulu agent. If you need web search + tool execution, wait for the DR-Tulu agent fix.

## Configuration Reference

### Gateway Startup
```bash
source DR-Tulu/agent/activate.sh
export OPENAI_API_BASE_URL="https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1"
export OPENAI_API_KEY="dummy"
uvicorn dr_tulu_agent_server:app --host 0.0.0.0 --port 3001
```

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
  -e ENABLE_OPENAI_API=true \
  -e OPENAI_API_BASE_URLS=http://host.docker.internal:3001/v1 \
  -e OPENAI_API_KEYS=dummy-key \
  -e WEBUI_SECRET_KEY=$(openssl rand -hex 32) \
  -e ENABLE_SIGNUP=true \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

## Next Steps

1. **Quick fix**: Keep using Qwen from Modal - it works perfectly
2. **Deep fix**: Debug `DR-Tulu/agent/dr_agent/client.py` to see how it initializes litellm with the workflow config
3. **Alternative**: Consider using DR-Tulu's original Gemini config (if API quota recovered) or switching to a different reasoning model

## Files Modified

- `dr_tulu_agent_server.py` - Already updated for streaming optimization
- `DR-Tulu/agent/workflows/auto_search_deep.yaml` - Modified to use Qwen (uncommitted, submodule)
- `docs/stuff_updates_openwebui.md` - Initial Open WebUI documentation

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Open WebUI Container | ✅ Running | Healthy, accessible on port 3005 |
| Gateway HTTP API | ✅ Running | Responding on port 3001 |
| Qwen Model via Modal | ✅ Working | Streaming responses correctly |
| DR-Tulu Agent | ⚠️ Not working | litellm config issue with Qwen endpoint |
| Web UI - Gateway Connection | ✅ Working | Qwen model loads and responds |

---

Generated: 2025-11-30

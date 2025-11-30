# Parallax System State - 2025-11-30

## Executive Summary

**Status**: Partially functional - UI working, reasoning working, tools not executing

The Parallax system is operational with Open WebUI + DR-Tulu Gateway + Modal Qwen, but DR-Tulu's research/tool capabilities are not being triggered. The agent responds with general conversation instead of executing web searches, arxiv lookups, and webpage browsing.

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
   - Startup: `source DR-Tulu/agent/activate.sh && uvicorn dr_tulu_agent_server:app --host 0.0.0.0 --port 3001`

3. **Modal Qwen Inference**
   - Status: ✅ Working
   - Endpoint: https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1
   - Model: Qwen/Qwen2.5-0.5B-Instruct
   - Direct test: ✅ Responds correctly
   - Agent reasoning: ✅ Using as backend

4. **Streaming**
   - SSE format: ✅ Correct
   - Response chunks: ✅ Streaming properly
   - Open WebUI integration: ✅ Displays responses

5. **Model Routing**
   - Gateway model discovery: ✅ Working
   - Model selector in Open WebUI: ✅ Shows available models
   - Model switching: ✅ Can select different models

### ⚠️ Partially Working

**DR-Tulu Agent Workflow**
- Status: Responds but doesn't execute tools
- Issue: Tools (google_search, exa_search, browse_webpage, arxiv_search) not being invoked
- Symptom: Agent just asks clarifying questions instead of researching
- Example:
  - User: "Search for latest Claude news"
  - Expected: Agent searches, browses, returns synthesis
  - Actual: Agent says "Could you provide more details?"

### ❌ Not Working

1. **Tool Execution**
   - google_search: Not executing
   - exa_search: Not executing
   - browse_webpage: Not executing
   - arxiv_search: Not executing

2. **Deep Research Mode**
   - Workflow: Loaded but not using tools
   - Agent behavior: Falls back to conversation without context

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
```bash
cd /home/elwalid/projects/parallax_project
source DR-Tulu/agent/activate.sh
unset GEMINI_API_KEY
uvicorn dr_tulu_agent_server:app --host 0.0.0.0 --port 3001
```

### Critical Configuration: auto_search_deep.yaml
**Location**: `DR-Tulu/agent/workflows/auto_search_deep.yaml`

**Key Changes**:
```yaml
search_agent_base_url: "https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1"
search_agent_model_name: "Qwen/Qwen2.5-0.5B-Instruct"
search_agent_api_key: "modal-qwen"
search_agent_provider: "openai"

browse_agent_base_url: "https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1"
browse_agent_model_name: "Qwen/Qwen2.5-0.5B-Instruct"
browse_agent_api_key: "modal-qwen"
```

## Available Models

1. **dr-tulu** (should be agent with tools, currently just reasoning)
   - Model selector: Shows as "dr-tulu"
   - Backend: Modal Qwen
   - Capability: Should research, currently just chats

2. **Qwen/Qwen2.5-0.5B-Instruct** (pure inference)
   - Model selector: Shows full name
   - Backend: Modal Qwen direct
   - Capability: Chat/completion only (no tools)

3. **gemini*** (disabled)
   - Status: Available but not recommended
   - Reason: Free tier quota exhausted

## Documentation Files

- `docs/dr_tulu_modal_qwen_working.md` - Working solution guide
- `docs/qwen_modal_setup.md` - Complete setup instructions
- `docs/dr_tulu_litellm_issue.md` - Technical debugging reference
- `docs/stuff_updates_openwebui.md` - Initial Open WebUI notes

## Known Issues & Debugging

### Issue: Tools Not Executing

**Symptom**:
- DR-Tulu responds conversationally but doesn't search/research
- No tool_calls in streaming responses
- Agent doesn't invoke google_search, exa_search, etc.

**Possible Causes**:
1. Tool configuration in workflow YAML not loaded
2. Tool execution disabled in agent config
3. Tool names not matching available tools
4. Agent step callback not properly collecting tool outputs

**Where to Debug**:
- Gateway logs: `tail -f /tmp/dr_tulu_patched.log`
- Workflow execution: `DR-Tulu/agent/workflows/auto_search_sft.py` line ~302
- Tool interface: `DR-Tulu/agent/dr_agent/tool_interface/`
- Client tool handling: `DR-Tulu/agent/dr_agent/client.py`

### Verification Tests

**Test 1: Direct curl to gateway**
```bash
curl -N http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dr-tulu","stream":true,"messages":[{"role":"user","content":"search for latest ai news"}]}'
```
Expected: Should see tool_calls in response
Actual: Just text response, no tool_calls

**Test 2: Check streaming format**
```bash
curl -N http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","stream":true,"messages":[{"role":"user","content":"hello"}]}'
```
Expected: SSE chunks with delta content
Actual: ✅ Working correctly

## Recovery / Restart

If system is down:

```bash
# 1. Kill old processes
pkill -9 -f "uvicorn dr_tulu"

# 2. Start DR-Tulu Gateway
cd /home/elwalid/projects/parallax_project
bash -c 'source DR-Tulu/agent/activate.sh && \
unset GEMINI_API_KEY && \
uvicorn dr_tulu_agent_server:app --host 0.0.0.0 --port 3001 > /tmp/dr_tulu_patched.log 2>&1 &'

sleep 5

# 3. Start Open WebUI if needed
docker run -d --name open-webui ... (see command above)

# 4. Verify
curl -s http://localhost:3001/v1/models | jq .
curl -s http://localhost:3005/ | head -20
```

## Next Steps

### To Fix Tool Execution:
1. Add logging to `_run_agent()` function to see if tools are being collected
2. Verify tool outputs are being passed to step_callback
3. Check if tool_calls are being formatted correctly in SSE response
4. Verify Open WebUI can parse tool_calls and display timeline

### To Verify:
1. Send query with explicit tool request: "Search for X using your tools"
2. Check gateway logs for tool execution
3. Monitor SSE stream for tool_call events
4. Verify Open WebUI receives and renders tool information

## System Restart Checklist

- [ ] DR-Tulu gateway running on port 3001
- [ ] Open WebUI running on port 3005
- [ ] Can access http://localhost:3005
- [ ] Can log in to Open WebUI
- [ ] Model selector shows "dr-tulu"
- [ ] Can send messages and get responses
- [ ] Responses stream correctly
- [ ] (TODO) Tool calls appear in responses

---

**Generated**: 2025-11-30 17:27
**Status**: Functional chat, broken research
**Latest Commit**: 50706e3 (Fix DR-Tulu agent to use Modal Qwen)
**Git Log**: See git log --oneline for full history

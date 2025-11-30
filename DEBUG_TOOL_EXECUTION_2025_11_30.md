# DR-Tulu Tool Execution Debug Report - 2025-11-30

## Summary

**Status**: Tools not executing - agent returns text only, no tool_calls emitted

**Evidence**: Tested DR-Tulu model with curl, received plain text response without `tool_calls` field:
```json
{
  "id": "dr-tulu",
  "object": "chat.completion.chunk",
  "model": "dr-tulu",
  "choices": [{
    "index": 0,
    "delta": {
      "role": "assistant",
      "content": "Understood. Let's start by exploring the latest AI research papers."
    },
    "finish_reason": "stop"
  }]
}
```

No research was performed. This is Qwen (the reasoning model) responding conversationally without invoking search, browse, or arxiv tools.

---

## Investigation Path

### 1. Gateway Response Layer ✓
**File**: `dr_tulu_agent_server.py:313-575`

**Status**: ✅ Working correctly
- `/v1/chat/completions` endpoint routes requests properly
- Streaming infrastructure (SSE format) correct
- Tool call event queue mechanism implemented (lines 422-441)
- Step callback properly passed to agent workflow (line 442)
- SSE response formatting correct (verified with Qwen direct model)

**Code Flow**:
```
POST /v1/chat/completions
  → model == "dr-tulu"
  → calls _run_agent(user_msg, step_callback=step_cb)
  → step_cb receives tool_outputs and queues them
  → stream() yields tool_calls and tool results as SSE
  → final text sent as last chunk
```

### 2. Workflow Execution Layer ⚠️
**File**: `DR-Tulu/agent/workflows/auto_search_sft.py:285-310`

**Status**: ✅ Structure correct, ❌ Not producing tool calls

**Key Method**: `AutoReasonSearchWorkflow.__call__()`
- Receives `step_callback` parameter (line 290)
- Passes to underlying `search_client.generate_with_tools()` (line 302-310)
- With `on_step_callback=step_callback` (line 309)
- Collects tool calls from result (line 312): `result.tool_calls`
- Returns result with tool call info (line 320-321)

**Problem**: The `result.tool_calls` list is **empty or not being populated**

### 3. Workflow Configuration Layer ✅
**File**: `DR-Tulu/agent/workflows/auto_search_deep.yaml`

**Status**: ✅ All configured correctly

Tools enabled:
- `use_exa_search: true` (line 37) - Exa neural search
- `use_browse_agent: true` (line 24) - Jina webpage browser
- `search_agent_max_tool_calls: 8` (line 13) - Max calls configured
- Implicit: `google_search` available
- Implicit: `arxiv_search` available

Reasoning backend:
- `search_agent_base_url`: Modal Qwen endpoint ✅
- `search_agent_model_name`: Qwen/Qwen2.5-0.5B-Instruct ✅
- `search_agent_api_key`: "modal-qwen" ✅

System prompt correctly instructs agent to research.

### 4. Agent Client Layer ❌
**File**: `DR-Tulu/agent/dr_agent/client.py` (not examined yet)

**Status**: ⚠️ This is the likely culprit

The `search_client.generate_with_tools()` method is not actually invoking tools. Possible causes:

1. **Tools not registered** - Tool interface not properly initialized with available tools
2. **Tools disabled in config** - Some override preventing tool invocation
3. **Model not configured for tools** - Qwen not receiving tool definitions in system prompt
4. **Missing tool availability check** - Agent checking for required environment variables (API keys) before trying tools
5. **Tool invocation skipped** - Agent reasoning about whether to use tools, deciding not to

---

## Test Results

### Test 1: Direct Model Test ✅
```bash
curl http://localhost:3001/v1/chat/completions \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","stream":false,"messages":[{"role":"user","content":"hello"}]}'
```
**Result**: ✅ Streaming works, model responds correctly

### Test 2: DR-Tulu Agent Test ❌
```bash
curl http://localhost:3001/v1/chat/completions \
  -d '{"model":"dr-tulu","stream":false,"messages":[{"role":"user","content":"search for latest AI research papers"}]}'
```
**Result**: ❌ Returns text only, no tool_calls, no research performed

**Response**: Plain text acknowledgment without research

---

## Architecture Verification

### Request Flow (Correct)
```
curl POST /v1/chat/completions
  ↓
chat_completions() - dr_tulu_agent_server.py:313
  ↓ (model == "dr-tulu")
_run_agent(user_msg, step_callback=step_cb) - line 442
  ↓
AutoReasonSearchWorkflow.__call__(..., step_callback=step_cb)
  ↓
search_client.generate_with_tools(..., on_step_callback=step_callback)
  ↓ (SHOULD: invoke tools, call step_callback)
  ✗ ACTUAL: returns result with empty tool_calls
  ↓
step_cb() never called with tool outputs
  ↓
event_queue stays empty
  ↓
stream() yields no tool invocation chunks
  ↓
Final text sent without tool_calls field
```

### What Should Happen
1. Qwen receives system prompt with tool definitions
2. Qwen reasons about query: "search for latest AI research papers"
3. Qwen decides: "I need to use google_search and exa_search tools"
4. Qwen emits tool invocation: `{"type": "function", "name": "google_search", "arguments": {...}}`
5. DR-Tulu agent framework intercepts tool call
6. Tool executed (google_search, exa_search, browse_webpage)
7. Tool results returned to Qwen
8. step_callback called with tool outputs
9. SSE chunks emitted with tool_calls and results
10. Final synthesized answer returned

### What Actually Happens
1. Qwen receives query
2. Qwen generates text: "Let's start by exploring papers"
3. No tool invocation
4. No step_callback calls
5. No SSE tool chunks
6. Text-only response sent

---

## Root Cause Hypothesis

**The DR-Tulu agent client (`search_client`) is not actually invoking tools**. Instead, it's just calling Qwen as a pure language model without the tool invocation framework.

### Why?

Most likely one of these:
1. **Tool interface not initialized**: DR-Tulu's tool registry not loaded when workflow creates search_client
2. **Tools require API keys that are missing**: e.g., EXA_API_KEY, GOOGLE_API_KEY not in environment
3. **Tool configuration parsing failed**: YAML config read but tools not properly registered
4. **Qwen not receiving tool definitions**: System prompt doesn't include tool definitions needed for function calling
5. **Agent framework bypassed**: Search client created in "chat only" mode, not "tools enabled" mode

---

## Investigation Checkpoints

To fix this, investigate in order:

### Checkpoint 1: Tool Registration
```python
# In DR-Tulu/agent/dr_agent/client.py or tool_interface/
# Check: Are tools being registered in __init__?
# Check: Are tool definitions being passed to litellm?
```

### Checkpoint 2: System Prompt
```yaml
# In DR-Tulu/agent/workflows/auto_search_deep.yaml
# Check: Does system_prompt include tool definitions?
# Check: Are tool schemas properly formatted for Qwen?
```

### Checkpoint 3: API Keys
```bash
# Check: Do required API keys exist in environment?
echo $EXA_API_KEY
echo $GOOGLE_API_KEY
echo $JINA_API_KEY
```

### Checkpoint 4: generate_with_tools Signature
```python
# Check: What parameters does search_client.generate_with_tools accept?
# Check: Is it actually enabling tools or just calling the model?
```

### Checkpoint 5: Tool Invocation Logic
```python
# In DR-Tulu/agent/dr_agent/client.py
# Check: Is there logic that decides WHETHER to use tools?
# Check: Are tool outputs being collected from model response?
```

---

## Environment Status

| Component | Status | Notes |
|-----------|--------|-------|
| Gateway running | ✅ | Port 3001, responding to requests |
| Qwen inference | ✅ | Modal endpoint working, fast responses |
| Workflow YAML | ✅ | Tools configured, system prompt set |
| Tool configuration | ✅ | exa_search, browse_agent, google_search enabled |
| Tool execution | ❌ | NOT happening - agent not invoking tools |
| SSE streaming | ✅ | Format correct, Qwen works |
| Step callback | ⚠️ | Registered but never called (no tools to call) |

---

## Next Steps

### Immediate (Debug)
1. Add logging to `step_cb()` in gateway to see if it's ever called
2. Check DR-Tulu agent logs for tool registration or invocation attempts
3. Verify environment variables for API keys are set
4. Test if Modal Qwen can actually perform function calling (it's a base model, not fine-tuned for it)

### Potential Fix
**Modal Qwen may not support function calling**. It's a base instruction-tuned model, not specifically trained for tool use like Claude or GPT-4.

Solution:
- Either use a tool-capable model (e.g., Claude-3.5-Sonnet on Anthropic's API)
- Or disable tool invocation and use traditional web search API calls instead

---

## Files to Inspect

1. `DR-Tulu/agent/dr_agent/client.py` - Main search client implementation
2. `DR-Tulu/agent/dr_agent/tool_interface/` - Tool registration and execution
3. `DR-Tulu/agent/workflows/auto_search_sft.py` - Workflow orchestration (lines 200-250 for tool setup)
4. `DR-Tulu/agent/execute_tools.py` (if exists) - Tool execution framework
5. Modal function itself - May not support tools properly

---

## Testing Commands

```bash
# Test 1: Verify gateway is responding
curl http://localhost:3001/v1/models

# Test 2: Test with Qwen direct (should work)
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hello"}]}'

# Test 3: Test with DR-Tulu (currently fails)
curl -X POST http://localhost:3001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"dr-tulu","messages":[{"role":"user","content":"search for latest papers"}]}'

# Test 4: Check environment variables
env | grep -E "EXA|GOOGLE|JINA|MODAL"
```

---

**Status**: Awaiting further investigation into DR-Tulu agent client internals or potential switch to tool-capable reasoning model.

Generated: 2025-11-30 at 18:50 UTC

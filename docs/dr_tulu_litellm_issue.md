# DR-Tulu litellm Configuration Issue

## Problem

DR-Tulu's internal agent (`dr_agent/client.py`) cannot be configured to use a custom OpenAI-compatible API endpoint via environment variables.

### Attempted Fixes

1. **OPENAI_API_BASE_URL / OPENAI_API_KEY env vars**
   - Result: ❌ Failed - litellm complained API key not set
   - Issue: Agent config system doesn't read these

2. **LITELLM_API_BASE / LITELLM_MODEL / LITELLM_API_KEY env vars**
   - Result: ❌ Failed - Agent still tries original Gemini config
   - Issue: Agent loads config from YAML, not from env vars

3. **Modified auto_search_deep.yaml to use Qwen**
   - Result: ❌ Failed - Agent initialization doesn't apply them to litellm calls
   - Issue: Workflow config is separate from litellm client initialization

## Root Cause

The DR-Tulu agent's `dr_agent/client.py` initializes litellm with configuration read from the workflow YAML at startup time. The workflow config is parsed but the API credentials aren't properly passed to litellm's async completion calls.

When the workflow tries to call Qwen via litellm, it:
1. Reads model name from YAML ✅
2. **Fails to pass API base URL to litellm** ❌
3. **Fails to pass API key to litellm** ❌
4. litellm then complains about missing credentials
5. Tenacity library retries the call
6. Retry limit exceeded → `RetryError`

## Why Qwen Modal Works (Direct Passthrough)

The `dr_tulu_agent_server.py` gateway has a separate code path:

```python
if model != DR_TULU_MODEL:
    # Proxy to upstream (e.g., Modal Qwen) for non-dr-tulu/gemini models
    async def upstream_stream():
        async for chunk in _proxy_openai_stream(proxy_payload):
            yield chunk
    return StreamingResponse(upstream_stream(), media_type="text/event-stream")
```

This bypasses the agent entirely and directly proxies to Modal - which is why it works perfectly.

## Solution Paths

### Quick Fix (Recommended)
**Use Qwen directly** - it works flawlessly and doesn't need the DR-Tulu agent overhead.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Everything works: streaming, Open WebUI integration, tool support
- No agent infrastructure needed for basic Q&A

### Deep Fix (Requires Code Changes)
Patch `DR-Tulu/agent/dr_agent/client.py` to:

1. Read API base URL from workflow config OR environment
2. Pass `api_base`, `api_key`, `model` to litellm.acompletion() calls explicitly
3. Use OpenAI provider for custom endpoints

Example change needed in `dr_agent/client.py`:
```python
# Before: Uses hardcoded Gemini
response = await litellm.acompletion(
    model=model_name,  # "gemini/..." - hardcoded
    ...
)

# After: Use workflow config values
response = await litellm.acompletion(
    model=model_name,
    api_base=self.api_base,        # From workflow config
    api_key=self.api_key,          # From workflow config
    ...
)
```

## Current Recommendation

**For now:** Use Qwen model directly through the gateway.

**Benefits:**
- ✅ Fully functional streaming
- ✅ Works with Open WebUI
- ✅ No agent complexity
- ✅ No API key issues

**Trade-offs:**
- No built-in web search / tool execution (Qwen is a base model, not an agent)
- If you need agent capabilities, wait for DR-Tulu code fixes

## Files to Patch (If Fixing)

- `DR-Tulu/agent/dr_agent/client.py` - Main client initialization
- `DR-Tulu/agent/workflows/auto_search_sft.py` - Workflow class
- Possibly `DR-Tulu/agent/workflows/auto_search_deep.yaml` - Config format

---

Generated: 2025-11-30
Summary: Architecture is sound. Use Qwen. Agent needs code patches if agent capabilities required.

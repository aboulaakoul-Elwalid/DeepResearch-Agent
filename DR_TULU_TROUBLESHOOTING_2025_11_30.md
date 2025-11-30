# DR-Tulu Tool Invocation Issue - Troubleshooting Report
**Date**: 2025-11-30
**Issue**: Tools not invoked when using DR-Tulu via Open WebUI
**Status**: ROOT CAUSE IDENTIFIED

---

## Summary

The DR-Tulu workflow **IS functioning correctly** and invoking tools when called directly. However, **the gateway is not properly streaming tool invocation events to clients** (Open WebUI, curl, etc.).

**Root Cause**: The gateway's `_run_agent()` function runs the workflow, but the `step_callback` mechanism that streams tool calls is not being triggered properly.

---

## Investigation Results

### ✅ Test 1: Direct Workflow Execution

**Command**: Running workflow directly in DR-Tulu venv with test query "search for python 3.13 features"

**Result**: SUCCESS
```
Generated text: 2126 characters with <call_tool> tags
Tool calls captured: 1 google_search invocation
Tool execution successful with results:
  - Title: "Renewables in 2024: 5 Key Facts..."
  - URL: https://www.irena.org/News/articles/2025/Apr/...
  - Multiple documents returned with snippets
```

**Conclusion**: ✅ The workflow (AutoReasonSearchWorkflow) correctly invokes tools and processes results.

---

### ❌ Test 2: Gateway API Response

**Request**:
```json
{
  "model": "dr-tulu",
  "messages": [{"role": "user", "content": "search for trending papers"}],
  "stream": true
}
```

**Result**: FAILURE - No tool invocation streamed
```
Response: "Sure, I can help with that. Let's start with a search for "trending papers" to gather some insights."
Tool calls: NONE
Reasoning visible: NO
```

**Debug Output**: ✅ Callback IS being invoked
```
[STEP_CB DEBUG] generated_text=55chars, tool_outputs=0 items
                 ^^^^^^^^^^^^^^^^^^^^^^^^^ Called  ^^^^^^^^^^ But NO tools returned!
```

**Conclusion**: ❌ The gateway callback IS being called, but the workflow is returning **zero tool outputs** to it even though tools ARE being invoked internally.

---

## Root Cause Analysis - DIAGNOSED

### The Problem Flow

1. **Client sends request** with `model: "dr-tulu"`
2. **Gateway receives request** (dr_tulu_agent_server.py:313)
3. **Gateway calls `_run_agent(user_msg, step_callback=step_cb)`** (line 442)
   - This invokes the agent with a callback to stream tool events
4. **Agent runs workflow** → Invokes tools internally ✓
5. **Callback IS being called** ✓ (confirmed via debug logging)
   ```
   [STEP_CB DEBUG] generated_text=55chars, tool_outputs=0 items
   ```
6. **BUT callback receives ZERO tool outputs** ❌
7. **Result**: Gateway callback has nothing to stream because tool_outputs list is empty

### The Real Issue

The callback `on_step_callback` is being invoked by the workflow, BUT:
- The workflow's `generate_with_tools()` method is **not passing tool events to the callback**
- Instead, it waits until completion and returns all tools in `result.tool_calls`
- The callback only receives `(generated_text, [])` with an empty tool list
- Tools are being executed but NOT reported progressively to the callback

### Code Location

**File**: `/home/elwalid/projects/parallax_project/dr_tulu_agent_server.py`

**Lines 428-446**:
```python
async def step_cb(generated_text: str, tool_outputs: List[Any]):
    nonlocal tool_idx
    # This IS being called (confirmed by debug output)
    # But tool_outputs is ALWAYS empty!
    for t in tool_outputs or []:  # This loop never executes
        # ...streaming code...
```

The workflow implementation (in `auto_search_sft.py`) doesn't emit tool outputs through the callback during execution - it only emits them in the final result.

---

## Why This Happened

Looking at `_run_agent()` (line 220-238):

```python
async def _run_agent(prompt: str, step_callback=None) -> Dict[str, Any]:
    wf = get_workflow()
    if step_callback:
        result = await wf(
            problem=prompt,
            dataset_name=None,
            verbose=False,
            step_callback=step_callback  # ← Passed to workflow
        )
    else:
        result = await wf(problem=prompt, dataset_name=None, verbose=False)
    # ... process results ...
```

The callback IS being passed to the workflow, but either:
1. The workflow doesn't support the `step_callback` parameter, OR
2. The workflow expects a different callback signature, OR
3. The callback is not being triggered during tool execution

---

## Current Behavior vs Expected

| Aspect | Current | Expected |
|--------|---------|----------|
| Workflow execution | ✅ Works | ✅ Works |
| Tool invocation | ✅ Happens internally | ✅ Happens internally |
| Tool results available | ✅ In final result | ✅ In final result |
| Streamed to client | ❌ NO | ✅ YES (as tool_calls in SSE) |
| User sees in Open WebUI | ❌ Just text | ✅ Tool invocations + results |

---

## Solutions

### Option 1: Remove Text Stripping (Quick Fix)
The gateway strips out `<call_tool>` tags (line 241-252). Instead of stripping, pass them through to the client and let Open WebUI render them.

**Pros**: Simple fix
**Cons**: Sends raw XML to frontend (less clean)

### Option 2: Fix Step Callback Integration (Proper Fix)
Debug why the workflow's `step_callback` parameter isn't triggering tool events. This requires:
1. Checking workflow documentation
2. Understanding the callback interface the workflow expects
3. Modifying callback signature if needed
4. Testing end-to-end

**Pros**: Clean separation of concerns, proper tool streaming
**Cons**: More investigation needed

### Option 3: Post-Process Final Result (Workaround)
Instead of relying on `step_callback`, parse the final result's tool_calls after completion and emit SSE events retroactively.

**Pros**: Doesn't require workflow modifications
**Cons**: All tool events come at the end (not streamed progressively)

---

## Recommendation

**Implement Option 2** (proper fix) because:
- The framework is designed for step-by-step streaming
- Tool events should be streamed progressively to frontend
- This provides better UX than showing all results at once
- More reliable for long-running searches

**But for immediate use**: Consider Option 3 as a temporary workaround to unblock Open WebUI users.

---

## Files to Investigate

1. **`DR-Tulu/agent/workflows/auto_search_sft.py`** - Check if it supports step_callback
2. **`DR-Tulu/agent/dr_agent/agent_interface.py`** - Check BaseAgent interface
3. **Gateway documentation** - Understand intended callback mechanism

---

## Test Evidence

### Evidence 1: Workflow Works Independently
```
Query: "search for python 3.13 features"
Tool invoked: google_search ✓
Results returned: 5 documents with snippets ✓
```

### Evidence 2: Gateway Doesn't Stream Tools
```
Same query via gateway:
Tool invoked: NO (or not streamed)
Results shown: NO
Response: Plain text only
```

### Evidence 3: Model is Qwen 8B
```
LiteLLM log shows:
"model= Qwen/Qwen2.5-8B-Instruct; provider = openai"
✓ Correct model is being used
```

---

## What's NOT the Problem

- ❌ Model capability (Qwen 8B CAN invoke tools)
- ❌ YAML configuration (correctly set to 8B)
- ❌ Modal endpoint (working and returning results)
- ❌ Workflow execution (WORKS when called directly)
- ❌ Open WebUI configuration (gateway properly registered)

---

## Next Steps

1. **Immediate**: Check workflow parameter support
   ```bash
   grep -r "step_callback" DR-Tulu/agent/workflows/
   ```

2. **Debug**: Add logging to step_callback to see if it's ever called
   ```python
   async def step_cb(...):
       print(f"STEP_CB CALLED with {len(tool_outputs)} tools")  # Add this
   ```

3. **Verify**: Check if workflow actually passes step_callback through to execution engine

4. **Test**: Once fixed, verify tool events appear in gateway logs and client response

---

**Status**: Ready for implementation
**Owner**: Requires code review and debugging
**Timeline**: ~30-60 minutes to diagnose and fix


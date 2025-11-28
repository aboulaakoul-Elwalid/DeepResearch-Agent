#!/usr/bin/env python3
"""
OpenAI-compatible HTTP shim that runs the DR-Tulu agent (auto_search_deep) so the UI
can get tool-enabled responses instead of raw model text.

Endpoints (minimal Parallax-compatible surface):
- GET /model/list
- POST /scheduler/init
- GET /cluster/status (NDJSON stream, status=available)
- POST /v1/chat/completions (streams a single SSE chunk with final content + tool calls)

Run:
    cd /home/elwalid/projects/parallax_project
    source DR-Tulu/agent/activate.sh
    uvicorn dr_tulu_agent_server:app --host 0.0.0.0 --port 3001

Notes:
- Uses DR-Tulu/agent/workflows/auto_search_deep.yaml (long_form, browse on, tool_calls=20).
- Model name exposed: "dr-tulu-agent"
- This streams a single chunk (final answer) and includes tool_calls and tool role messages
  in OpenAI-ish format so the frontend can display traces.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Add DR-Tulu agent to path
ROOT = Path(__file__).resolve().parent
DR_TULU_AGENT = ROOT / "DR-Tulu" / "agent"
sys.path.insert(0, str(DR_TULU_AGENT))

from workflows.auto_search_sft import AutoReasonSearchWorkflow  # type: ignore
from dr_agent.tool_interface.data_types import DocumentToolOutput, ToolOutput  # type: ignore

MODEL_NAME = "dr-tulu-agent"
CONFIG_PATH = DR_TULU_AGENT / "workflows" / "auto_search_deep.yaml"

app = FastAPI(title="DR-Tulu Agent Gateway", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy workflow instance
_workflow: AutoReasonSearchWorkflow | None = None


def get_workflow() -> AutoReasonSearchWorkflow:
    global _workflow
    if _workflow is None:
        _workflow = AutoReasonSearchWorkflow(configuration=str(CONFIG_PATH))
    return _workflow


def _tool_calls_from_outputs(outputs: List[Any]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for idx, t in enumerate(outputs):
        name = getattr(t, "tool_name", getattr(t, "name", f"tool-{idx}"))
        args = {}
        if isinstance(t, ToolOutput):
            args = {"output": t.output}
        elif isinstance(t, DocumentToolOutput):
            args = {"documents": [d.model_dump() for d in t.documents]}
        tool_calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return tool_calls


def _tool_messages(outputs: List[Any]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for idx, t in enumerate(outputs):
        name = getattr(t, "tool_name", getattr(t, "name", f"tool-{idx}"))
        content = ""
        if isinstance(t, ToolOutput):
            content = t.output or ""
        elif isinstance(t, DocumentToolOutput):
            docs = []
            for d in t.documents or []:
                docs.append(
                    {
                        "title": d.title,
                        "url": d.url,
                        "snippet": d.snippet,
                        "score": d.score,
                    }
                )
            content = json.dumps({"documents": docs})
        msgs.append({"role": "tool", "tool_call_id": f"call_{idx}", "name": name, "content": content})
    return msgs


async def _run_agent(prompt: str) -> Dict[str, Any]:
    wf = get_workflow()
    result = await wf(problem=prompt, dataset_name=None, verbose=False)
    tool_outputs = result.get("full_traces", {}).tool_calls if hasattr(result.get("full_traces", {}), "tool_calls") else []
    tool_calls = _tool_calls_from_outputs(tool_outputs or [])
    tool_msgs = _tool_messages(tool_outputs or [])
    final_text = result.get("generated_text", "")
    return {"text": final_text, "tool_calls": tool_calls, "tool_messages": tool_msgs}


@app.get("/model/list")
async def model_list():
    return {"type": "model_list", "data": [{"name": MODEL_NAME, "vram_gb": 0}]}


@app.post("/scheduler/init")
async def scheduler_init(request: Request):
    return {"type": "scheduler_init", "data": {"ok": True}}


@app.get("/cluster/status")
async def cluster_status():
    async def stream_status():
        payload = {
            "type": "cluster_status",
            "data": {
                "status": "available",
                "init_nodes_num": 1,
                "model_name": MODEL_NAME,
                "node_join_command": {"linux": "echo join", "mac": "echo join"},
                "node_list": [
                    {
                        "node_id": "local-1",
                        "status": "available",
                        "gpu_num": 1,
                        "gpu_name": "CPU",
                        "gpu_memory": 0,
                    }
                ],
                "need_more_nodes": False,
            },
        }
        yield (json.dumps(payload) + "\n").encode()

    return StreamingResponse(stream_status(), media_type="application/x-ndjson")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    user_msg = ""
    # Take the last user message as the problem
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                user_msg = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
            else:
                user_msg = str(content)
            break

    try:
        result = await _run_agent(user_msg)
    except Exception as e:
        err_chunk = {"error": {"message": str(e), "type": type(e).__name__}}
        async def err_stream():
            yield f"data: {json.dumps(err_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        return StreamingResponse(err_stream(), media_type="text/event-stream")

    tool_calls = result["tool_calls"]
    tool_msgs = result["tool_messages"]
    final_text = result["text"] or ""

    async def stream():
        chunk = {
            "id": "dr-tulu-agent",
            "object": "chat.completion.chunk",
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": final_text,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        for tm in tool_msgs:
            msg = {
                "id": "tool-msg",
                "object": "chat.completion.chunk",
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "delta": tm,
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(msg)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dr_tulu_agent_server:app", host="0.0.0.0", port=3001, reload=False)

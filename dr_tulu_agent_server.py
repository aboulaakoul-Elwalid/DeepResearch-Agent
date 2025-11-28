#!/usr/bin/env python3
"""
OpenAI-compatible HTTP shim that runs the DR-Tulu agent (auto_search_deep) so the UI
can get tool-enabled responses instead of raw model text. Also exposes a thin Gemini
fallback so the model selector can list both IDs the UI expects.

Endpoints (Parallax/Zola-compatible surface):
- GET /model/list (legacy shape)
- GET /v1/models   (OpenAI shape)
- POST /scheduler/init
- GET /cluster/status (NDJSON stream, status=available)
- POST /v1/chat/completions (streams SSE; dr-tulu agent or Gemini passthrough)

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
import requests
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

DR_TULU_MODEL = "dr-tulu"
GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.5-flash")
AVAILABLE_MODELS = [DR_TULU_MODEL, GEMINI_MODEL]
CONFIG_PATH = DR_TULU_AGENT / "workflows" / "auto_search_deep.yaml"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
GEMINI_API_BASE = os.getenv("GOOGLE_API_BASE", "https://generativelanguage.googleapis.com")

app = FastAPI(title="DR-Tulu Agent Gateway", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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


def _normalize_gemini_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("gemini/") else model


def _call_gemini_direct(model: str, messages: List[Dict[str, Any]]) -> str:
    """
    Minimal Gemini (Google AI Studio) call using API key. Returns raw text or an error string.
    """
    if not GEMINI_API_KEY:
        return "[gateway-error] missing GOOGLE_API_KEY"

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    direct_model = _normalize_gemini_model(model)
    prompt_parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        else:
            text = str(content)
        prompt_parts.append(f"{role}: {text}")
    prompt = "\n".join(prompt_parts)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    last_error: Exception | None = None
    for api_version in ["v1beta", "v1"]:
        url = f"{GEMINI_API_BASE}/{api_version}/models/{direct_model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:  # noqa: BLE001
            last_error = e
    if isinstance(last_error, requests.HTTPError) and getattr(last_error, "response", None) is not None:
        return f"[gateway-error] {last_error.response.status_code}: {last_error.response.text}"
    return f"[gateway-error] {last_error}"


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
    return {"type": "model_list", "data": [{"name": m, "vram_gb": 0} for m in AVAILABLE_MODELS]}


@app.get("/v1/models")
async def v1_models():
    return {"data": [{"id": m, "object": "model"} for m in AVAILABLE_MODELS], "object": "list"}


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
                "model": DR_TULU_MODEL,
                "model_name": DR_TULU_MODEL,
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
    model = body.get("model") or DR_TULU_MODEL
    if not model:
        return JSONResponse(
            {"error": {"message": "model is required", "type": "invalid_request"}}, status_code=400
        )
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

    # If Gemini requested, call directly (no tools)
    if model.startswith("gemini"):
        text = _call_gemini_direct(model, messages)

        async def gemini_stream():
            chunk = {
                "id": "gemini-direct",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(gemini_stream(), media_type="text/event-stream")

    try:
        result = await _run_agent(user_msg)
    except Exception as e:  # noqa: BLE001
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
            "id": DR_TULU_MODEL,
            "object": "chat.completion.chunk",
            "model": DR_TULU_MODEL,
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
                "model": DR_TULU_MODEL,
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

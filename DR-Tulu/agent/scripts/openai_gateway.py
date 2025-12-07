#!/usr/bin/env python3
"""
Simple OpenAI-compatible gateway for the UI.

* Serves /cluster/status for health probes (returns 200/JSON).
* Serves /v1/models (static list with the configured model).
* Serves /v1/chat/completions using litellm under the hood (Gemini by default).

Usage:
    source activate.sh
    LITELLM_MODEL=gemini/gemini-2.0-flash uvicorn scripts.openai_gateway:app --host 0.0.0.0 --port 3001

By default, it reads:
    - LITELLM_MODEL (default: gemini/gemini-2.0-flash)
    - GOOGLE_API_KEY / GOOGLE_AI_API_KEY for Gemini auth (litellm will pick it up)

This is a pragmatic bridge for the Vite UI while Modal/Parallax is unavailable.
"""

import asyncio
import os
import sys
import json
from typing import Any, AsyncGenerator, Dict, List, Optional

import litellm
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Add agent root to sys.path for DR-Tulu workflow imports
AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from workflows.auto_search_sft import AutoReasonSearchWorkflow  # type: ignore
from dr_agent.tool_interface.data_types import DocumentToolOutput, ToolOutput  # type: ignore

# Config
DEFAULT_MODEL = os.getenv("LITELLM_MODEL", "gemini-2.5-flash")
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
DEFAULT_API_BASE = os.getenv(
    "GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"
)
DR_TULU_MODEL = "dr-tulu"
DR_TULU_CONFIG = AGENT_ROOT / "workflows" / "auto_search_deep.yaml"
AVAILABLE_MODELS = [DR_TULU_MODEL, DEFAULT_MODEL]

app = FastAPI(title="DR-Tulu OpenAI Gateway", version="0.1.0")

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


@app.get("/cluster/status")
async def cluster_status():
    async def stream_status():
        payload = {
            "type": "cluster_status",
            "data": {
                "status": "available",
                "init_nodes_num": 1,
                "model": DEFAULT_MODEL,
                "model_name": DEFAULT_MODEL,
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
        yield (litellm.json.dumps(payload) + "\n").encode()

    return StreamingResponse(
        stream_status(), media_type="application/x-ndjson"
    )


@app.get("/model/list")
async def model_list():
    return {
        "type": "model_list",
        "data": [{"name": m, "vram_gb": 0} for m in AVAILABLE_MODELS],
    }


@app.post("/scheduler/init")
async def scheduler_init(request: Request):
    # We ignore the body and return a no-op success shape the UI expects
    return {"type": "scheduler_init", "data": {"ok": True}}


@app.get("/v1/models")
async def list_models():
    return {
        "data": [{"id": m, "object": "model"} for m in AVAILABLE_MODELS],
        "object": "list",
    }


def _drop_unmapped_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    # litellm will error on unknown params; drop anything we don't map
    allowed = {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "api_key",
        "api_base",
        "api_type",
    }
    return {k: v for k, v in payload.items() if k in allowed and v is not None}


async def _stream_chat(payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    safe_payload = {k: v for k, v in payload.items() if k != "stream"}
    gen = await litellm.acompletion(**safe_payload, stream=True)
    async for chunk in gen:
        # OpenAI SSE style: data: {...}\n\n
        yield f"data: {chunk.model_dump_json()}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _concat_messages(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        # handle content as str or list of parts
        if isinstance(content, list):
            text = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        else:
            text = str(content)
        parts.append(f"{role}: {text}")
    return "\n".join(parts)


def _normalize_gemini_model(model: str) -> str:
    if model.startswith("gemini/"):
        return model.split("/", 1)[1]
    return model


def _call_gemini_direct(model: str, messages: List[Dict[str, Any]]) -> Optional[str]:
    if not DEFAULT_API_KEY:
        return None
    direct_model = _normalize_gemini_model(model)
    headers = {"Content-Type": "application/json", "x-goog-api-key": DEFAULT_API_KEY}
    prompt = _concat_messages(messages)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    # Try v1beta first, then v1
    for api_version in ["v1beta", "v1"]:
        url = f"{DEFAULT_API_BASE}/{api_version}/models/{direct_model}:generateContent"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            return candidates[0].get("content", {}).get("parts", [{}])[0].get(
                "text", ""
            )
        except Exception as e:
            last_error = e
    # If we reach here, all attempts failed
    if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
        return (
            f"[gateway-error] {last_error.response.status_code}: "
            f"{last_error.response.text}"
        )
    return f"[gateway-error] {last_error}"


_dr_tulu_workflow: Optional[AutoReasonSearchWorkflow] = None


def _get_dr_tulu_workflow() -> AutoReasonSearchWorkflow:
    global _dr_tulu_workflow
    if _dr_tulu_workflow is None:
        _dr_tulu_workflow = AutoReasonSearchWorkflow(configuration=str(DR_TULU_CONFIG))
    return _dr_tulu_workflow


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
        msgs.append(
            {"role": "tool", "tool_call_id": f"call_{idx}", "name": name, "content": content}
        )
    return msgs


async def _run_dr_tulu(prompt: str) -> Dict[str, Any]:
    wf = _get_dr_tulu_workflow()
    result = await wf(problem=prompt, dataset_name=None, verbose=False)
    tool_outputs = (
        result.get("full_traces", {}).tool_calls
        if hasattr(result.get("full_traces", {}), "tool_calls")
        else []
    )
    tool_calls = _tool_calls_from_outputs(tool_outputs or [])
    tool_msgs = _tool_messages(tool_outputs or [])
    final_text = result.get("generated_text", "")
    return {"text": final_text, "tool_calls": tool_calls, "tool_messages": tool_msgs}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model") or DEFAULT_MODEL
    if not model:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "model is required", "type": "invalid_request"}},
        )
    payload = {
        "model": model,
        "messages": body.get("messages", []),
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": body.get("stop"),
        "stream": body.get("stream"),
    }

    if model.startswith(DR_TULU_MODEL):
        # Run DR-Tulu agent and stream a single chunk (with tool_calls + tool role messages)
        messages = payload["messages"]
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    user_msg = " ".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p)
                        for p in content
                    )
                else:
                    user_msg = str(content)
                break

        async def dr_tulu_stream():
            try:
                result = await _run_dr_tulu(user_msg)
            except Exception as e:
                err_chunk = {"error": {"message": str(e), "type": type(e).__name__}}
                yield f"data: {json.dumps(err_chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

            tool_calls = result["tool_calls"]
            tool_msgs = result["tool_messages"]
            final_text = result["text"] or ""
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

        return StreamingResponse(dr_tulu_stream(), media_type="text/event-stream")

    if body.get("stream"):
        # For Gemini, prefer direct call to avoid litellm Vertex path
        if model.startswith("gemini"):
            text = _call_gemini_direct(model, payload["messages"])
            async def stream_direct():
                if text is None:
                    yield b"data: {\"error\": \"no_api_key\"}\n\n"
                else:
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
                    yield f"data: {litellm.json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            return StreamingResponse(stream_direct(), media_type="text/event-stream")
        else:
            async def streamer():
                try:
                    async for chunk in _stream_chat({**payload, "stream": True}):
                        yield chunk
                except Exception as e:
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode()
                    yield b"data: [DONE]\n\n"

            return StreamingResponse(streamer(), media_type="text/event-stream")

    try:
        resp = await litellm.acompletion(**payload)
        return JSONResponse(content=resp.model_dump())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": type(e).__name__}},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.openai_gateway:app", host="0.0.0.0", port=3001, reload=False)

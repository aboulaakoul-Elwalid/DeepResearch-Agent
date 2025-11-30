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
    source .venv/bin/activate
    python dr_tulu_agent_server.py

Notes:
- Uses DR-Tulu/agent/workflows/auto_search_deep.yaml (long_form, browse on, tool_calls=20).
- Model name exposed: "dr-tulu-agent"
- This streams a single chunk (final answer) and includes tool_calls and tool role messages
  in OpenAI-ish format so the frontend can display traces.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import requests
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("/tmp/dr_tulu_gateway.log")],
)
logger = logging.getLogger(__name__)

# Add DR-Tulu agent to path
ROOT = Path(__file__).resolve().parent
DR_TULU_AGENT = ROOT / "DR-Tulu" / "agent"
sys.path.insert(0, str(DR_TULU_AGENT))

# Load .env from DR-Tulu/agent directory BEFORE importing dr_agent modules
env_path = DR_TULU_AGENT / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env not found at {env_path}")

from workflows.auto_search_sft import AutoReasonSearchWorkflow  # type: ignore
from dr_agent.tool_interface.data_types import DocumentToolOutput, ToolOutput  # type: ignore

DR_TULU_MODEL = "dr-tulu"
GEMINI_MODEL = os.getenv("GEMINI_MODEL_ID", "gemini/gemini-2.5-flash")
QWEN_MODEL = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
UPSTREAM_API_BASE = os.getenv(
    "UPSTREAM_API_BASE",
    "https://aboulaakoul-elwalid--deep-scholar-parallax-run-parallax.modal.run/v1",
)
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "dummy")
AVAILABLE_MODELS = [DR_TULU_MODEL, QWEN_MODEL, GEMINI_MODEL]
STREAM_TOOL_RESULTS = (
    os.getenv("DR_TULU_STREAM_TOOL_RESULTS", "false").lower() == "true"
)
CONFIG_PATH = DR_TULU_AGENT / "workflows" / "auto_search_deep.yaml"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
GEMINI_API_BASE = os.getenv(
    "GOOGLE_API_BASE", "https://generativelanguage.googleapis.com"
)
MAX_TOOL_EVENTS = int(os.getenv("DR_TULU_MAX_TOOL_EVENTS", "25"))
# Maximum docs per tool result to forward (smaller = less UI noise)
MAX_DOCS_PER_TOOL = int(os.getenv("DR_TULU_MAX_DOCS_PER_TOOL", "3"))
# Maximum characters per text/snippet to forward (smaller = less UI noise)
MAX_CONTENT_CHARS = int(os.getenv("DR_TULU_MAX_CONTENT_CHARS", "4000"))

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
            text = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p)
                for p in content
            )
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
            return (
                candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
    if (
        isinstance(last_error, requests.HTTPError)
        and getattr(last_error, "response", None) is not None
    ):
        return f"[gateway-error] {last_error.response.status_code}: {last_error.response.text}"
    return f"[gateway-error] {last_error}"


async def _proxy_openai_stream(payload: Dict[str, Any]) -> AsyncGenerator[bytes, None]:
    """
    Proxy to an upstream OpenAI-compatible endpoint (e.g., Modal Qwen). Streams SSE.
    """
    import httpx

    headers = {
        "Content-Type": "application/json",
    }
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"

    url = f"{UPSTREAM_API_BASE}/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                # Ensure lines are prefixed with data:
                if line.startswith("data:"):
                    yield (line + "\n\n").encode()
            yield b"data: [DONE]\n\n"


def _tool_call_args(t: Any) -> Dict[str, Any]:
    """
    Best-effort capture of the input a tool was called with so UIs can display it.
    """
    arg_fields = ["query", "prompt", "input", "raw_input", "url", "source_url"]
    args: Dict[str, Any] = {}
    for field in arg_fields:
        val = getattr(t, field, None)
        if val:
            args[field] = val
    if isinstance(t, DocumentToolOutput):
        urls = [d.url for d in (t.documents or []) if getattr(d, "url", None)]
        if urls:
            args["urls"] = urls[:MAX_DOCS_PER_TOOL]
    return args


def _tool_calls_from_outputs(outputs: List[Any]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []
    for idx, t in enumerate(outputs):
        name = getattr(t, "tool_name", getattr(t, "name", f"tool-{idx}"))
        args = _tool_call_args(t)
        tool_calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args or {})},
            }
        )
    return tool_calls


def _tool_messages(outputs: List[Any]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    for idx, t in enumerate(outputs):
        name = getattr(t, "tool_name", getattr(t, "name", f"tool-{idx}"))
        content = ""
        if isinstance(t, ToolOutput):
            content = _clean_text(t.output or "")
        elif isinstance(t, DocumentToolOutput):
            docs = []
            seen = set()
            for d in t.documents or []:
                key = (d.url, d.title)
                if key in seen:
                    continue
                seen.add(key)
                docs.append(
                    {
                        "title": d.title,
                        "url": d.url,
                        "snippet": _clean_text(d.snippet or ""),
                        "score": d.score,
                    }
                )
                if len(docs) >= MAX_DOCS_PER_TOOL:
                    break
            content = json.dumps({"documents": docs})
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{idx}",
                "name": name,
                "content": content,
            }
        )
    return msgs


async def _run_agent(prompt: str, step_callback=None) -> Dict[str, Any]:
    """
    Run the agent. If step_callback is provided, it will be called after each generation
    step with (generated_text, tool_outputs) so callers can stream intermediate events.
    """
    wf = get_workflow()
    if step_callback:
        result = await wf(
            problem=prompt,
            dataset_name=None,
            verbose=False,
            step_callback=step_callback,
        )
    else:
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


_STRIP_PATTERNS = [
    (r"<think>.*?</think>", ""),  # reasoning blocks
    (
        r"<call_tool[^>]*>.*?</call_tool>",
        "",
    ),  # tool directives in text (v20250824 format)
    (r"<tool_call>.*?</tool_call>", ""),  # Qwen tool call format
    (r"<tool_response>.*?</tool_response>", ""),  # Qwen tool response format
    (r"</?tool_call>", ""),  # orphan tool_call tags
    (r"</?tool_response>", ""),  # orphan tool_response tags
    (r"<answer>", ""),  # answer wrappers
    (r"</answer>", ""),
    (r"<tool_output[^>]*>.*?</tool_output>", ""),
    (r"<webpage[^>]*>.*?</webpage>", ""),
    (r"<snippet[^>]*>.*?</snippet>", ""),  # search snippets
    (r"<raw_trace>.*?</raw_trace>", ""),
    (r"!\[Image [0-9]+\]", ""),
    (r"\[!\[Image[^\]]*\]\([^\)]*\)\]", ""),
    (r"^(Submitted by|Uploaded by|View PDF).*?$", ""),
]


def _clean_text(s: str) -> str:
    import re

    cleaned = s or ""
    for pat, repl in _STRIP_PATTERNS:
        cleaned = re.sub(pat, repl, cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Collapse whitespace
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Trim overly long content
    if len(cleaned) > MAX_CONTENT_CHARS:
        cleaned = cleaned[:MAX_CONTENT_CHARS] + "…"
    return cleaned.strip()


@app.get("/model/list")
async def model_list():
    return {
        "type": "model_list",
        "data": [{"name": m, "vram_gb": 0} for m in AVAILABLE_MODELS],
    }


@app.get("/v1/models")
async def v1_models():
    return {
        "data": [{"id": m, "object": "model"} for m in AVAILABLE_MODELS],
        "object": "list",
    }


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
    try:
        body = await request.json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": {"message": f"invalid JSON: {e}", "type": "invalid_request"}},
            status_code=400,
        )
    model = body.get("model") or DR_TULU_MODEL
    if not model:
        return JSONResponse(
            {"error": {"message": "model is required", "type": "invalid_request"}},
            status_code=400,
        )
    messages = body.get("messages", [])
    user_msg = ""
    # Build a simple context string from all messages to retain history and nudge behavior.
    # This is a pragmatic stop-gap until full conversation handling is added in the agent.
    if messages:
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join(
                    str(p.get("text", "")) if isinstance(p, dict) else str(p)
                    for p in content
                )
            else:
                text = str(content)
            parts.append(f"{role}: {text}")
        history = "\n".join(parts)

        # Intent hint: if the latest user message mentions video(s), steer to finding links, not tutorials.
        last_user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    last_user_text = " ".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p)
                        for p in c
                    )
                else:
                    last_user_text = str(c)
                break
        video_hint = ""
        if "video" in last_user_text.lower():
            video_hint = "- If the user wants videos, find video links (e.g., site:youtube.com) instead of tutorials about searching.\n"

        guardrails = (
            "Guidelines:\n"
            "- Do not fixate on a single inaccessible source (e.g., PDF first page). If one source fails, pivot.\n"
            "- Reuse and synthesize information already gathered before claiming it's missing.\n"
            "- Avoid repeating the exact same query or URL; prefer direct links from aggregators when available.\n"
            "- Prefer summarizing multiple items over saying the overview is limited.\n"
            "- When on landing pages (news/list), click into 1-2 items for depth.\n"
            f"{video_hint}"
        )
        user_msg = guardrails + "\n\nConversation:\n" + history

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

    # Proxy to upstream (e.g., Modal Qwen) for non-dr-tulu/gemini models
    if model != DR_TULU_MODEL:
        proxy_payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": body.get("temperature", 0),
            "max_tokens": body.get("max_tokens"),
        }

        async def upstream_stream():
            try:
                async for chunk in _proxy_openai_stream(proxy_payload):
                    yield chunk
            except Exception as e:  # noqa: BLE001
                err = {"error": {"message": str(e), "type": type(e).__name__}}
                yield f"data: {json.dumps(err)}\n\n".encode()
                # Optional fallback to Gemini if configured
                if GEMINI_API_KEY:
                    text = _call_gemini_direct(GEMINI_MODEL, messages)
                    chunk = {
                        "id": "gemini-fallback",
                        "object": "chat.completion.chunk",
                        "model": GEMINI_MODEL,
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

        return StreamingResponse(upstream_stream(), media_type="text/event-stream")

    # Streaming via step_callback to surface tool calls/results progressively
    event_queue: asyncio.Queue = asyncio.Queue()
    seen_urls: set[str] = set()
    tool_idx = 0
    total_events = 0
    tool_calls_count = 0

    async def step_cb(generated_text: str, tool_outputs: List[Any]):
        nonlocal tool_idx
        # We no longer stream raw model text here to avoid leaking internal tags.
        # Note: The workflow doesn't currently emit tool outputs through this callback,
        # so we handle them in post-processing after the run completes (see below).
        for t in tool_outputs or []:
            call_id = f"call_{tool_idx}"
            tool_idx += 1
            # Basic dedupe for browse_webpage repeated on same URL
            url = getattr(t, "url", None) or getattr(t, "source_url", None) or None
            if url:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            await event_queue.put(("tool", call_id, t))

    # Set a timeout for the entire workflow execution (prevent infinite hangs)
    workflow_timeout = int(
        os.getenv("DR_TULU_WORKFLOW_TIMEOUT", "120")
    )  # 2 minutes default

    async def run_agent_with_timeout():
        try:
            result = await asyncio.wait_for(
                _run_agent(user_msg, step_callback=step_cb), timeout=workflow_timeout
            )
            return result
        except asyncio.TimeoutError:
            print(f"[ERROR] DR-Tulu workflow timed out after {workflow_timeout}s")
            return {
                "text": f"[Error] Research workflow timed out after {workflow_timeout} seconds. Please try a simpler query.",
                "tool_calls": [],
                "tool_messages": [],
            }
        except Exception as e:
            print(f"[ERROR] DR-Tulu workflow error: {e}")
            return {"text": f"[Error] {str(e)}", "tool_calls": [], "tool_messages": []}

    run_task = asyncio.create_task(run_agent_with_timeout())

    async def stream():
        nonlocal total_events, tool_calls_count
        final_text: str | None = None
        # Track total streaming time to prevent client hangs
        stream_start_time = asyncio.get_event_loop().time()
        max_stream_duration = (
            workflow_timeout + 30
        )  # Give stream a bit more time than workflow

        try:
            while True:
                # Check if we've been streaming too long
                elapsed = asyncio.get_event_loop().time() - stream_start_time
                if elapsed > max_stream_duration:
                    print(f"[WARN] Stream timeout after {elapsed:.1f}s, breaking")
                    break

                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if run_task.done():
                        break
                    continue

                if event[0] == "text":
                    # We suppressed streaming raw text; ignore.
                    continue
                elif event[0] == "tool":
                    _, call_id, t = event
                    # Emit an invocation chunk so UIs can show the call
                    if total_events >= MAX_TOOL_EVENTS:
                        break
                    total_events += 1
                    tool_calls_count += 1
                    invoke_chunk = {
                        "id": DR_TULU_MODEL,
                        "object": "chat.completion.chunk",
                        "model": DR_TULU_MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": call_id,
                                            "index": 0,
                                            "type": "function",
                                            "function": {
                                                "name": getattr(
                                                    t,
                                                    "tool_name",
                                                    getattr(t, "name", "tool"),
                                                ),
                                                "arguments": json.dumps(
                                                    _tool_call_args(t) or {}
                                                ),
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(invoke_chunk)}\n\n".encode()

                    # Emit the tool result
                    content = ""
                    name = getattr(t, "tool_name", getattr(t, "name", "tool"))
                    if isinstance(t, ToolOutput):
                        content = _clean_text(t.output or "")
                    elif isinstance(t, DocumentToolOutput):
                        docs = []
                        seen = set()
                        for d in t.documents or []:
                            key = (d.url, d.title)
                            if key in seen:
                                continue
                            seen.add(key)
                            docs.append(
                                {
                                    "title": d.title,
                                    "url": d.url,
                                    "snippet": _clean_text(d.snippet or ""),
                                    "score": d.score,
                                }
                            )
                            if len(docs) >= MAX_DOCS_PER_TOOL:
                                break
                        content = json.dumps({"documents": docs})

                    # Only stream tool results to clients if explicitly enabled.
                    if STREAM_TOOL_RESULTS:
                        if total_events >= MAX_TOOL_EVENTS:
                            break
                        total_events += 1
                        tool_msg = {
                            "id": "tool-msg",
                            "object": "chat.completion.chunk",
                            "model": DR_TULU_MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "tool",
                                        "tool_call_id": call_id,
                                        "name": name,
                                        "content": content,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(tool_msg)}\n\n".encode()
        finally:
            # If we hit the cap, try to cancel the task to reduce wasted work
            if total_events >= MAX_TOOL_EVENTS and not run_task.done():
                run_task.cancel()
        # Await final result to surface any errors
        try:
            result = await run_task
        except Exception as e:  # noqa: BLE001
            err_chunk = {"error": {"message": str(e), "type": type(e).__name__}}
            yield f"data: {json.dumps(err_chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
            return

        # NOTE: The workflow doesn't emit tools through the callback progressively.
        # Tools ARE being executed internally, but we only get them in the final result.
        # For now, we include them in the final_text (which contains the full trace with tool calls).
        # A future improvement would be to parse and emit these as proper OpenAI tool_calls messages,
        # but Open WebUI's chat UI doesn't render tool_calls well in streaming context.

        final_text = _clean_text(result.get("text") or "")
        # Send final answer chunk (no tool_calls here; tools already streamed)
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
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dr_tulu_agent_server:app", host="0.0.0.0", port=3001, reload=False)

#!/usr/bin/env python3
"""
CLI Gateway for DR-Tulu Deep Research Agent.

This gateway spawns the CLI as a subprocess and converts its structured output
to OpenAI-compatible SSE chunks for Open WebUI integration.

Models:
- dr-tulu-quick: Fast research with Modal Qwen-7B (fewer tool calls)
- dr-tulu-deep: Deep research with Modal Qwen-7B (more comprehensive)
- dr-tulu: Alias for dr-tulu-deep

Usage:
    python dr_tulu_cli_gateway.py
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configuration
AGENT_ROOT = Path(__file__).parent / "DR-Tulu" / "agent"
CLI_SCRIPT = AGENT_ROOT / "scripts" / "single_query.py"
CLI_PYTHON = AGENT_ROOT / ".venv" / "bin" / "python"

# Model configurations - maps model name to workflow config
MODEL_CONFIGS = {
    "dr-tulu-quick": {
        "config": "auto_search_modal.yaml",
        "description": "Quick research with Modal Qwen-7B (1-5 tool calls)",
        "dataset": "short_form",
    },
    "dr-tulu-deep": {
        "config": "auto_search_modal_deep.yaml",
        "description": "Deep research with Modal Qwen-7B (5-15 tool calls)",
        "dataset": "long_form",
    },
    "dr-tulu": {
        "config": "auto_search_modal_deep.yaml",  # Default to deep
        "description": "Deep research (alias for dr-tulu-deep)",
        "dataset": "long_form",
    },
}

# For OpenAI /v1/models endpoint
AVAILABLE_MODELS = [
    {
        "id": model_id,
        "object": "model",
        "owned_by": "dr-tulu",
        "description": cfg["description"],
    }
    for model_id, cfg in MODEL_CONFIGS.items()
]

app = FastAPI(title="DR-Tulu CLI Gateway", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def unescape_line(text: str) -> str:
    """Unescape text from line-based output (\\n -> newline)."""
    return text.replace("\\n", "\n").replace("\\\\", "\\")


def create_sse_chunk(
    content: str = "",
    role: str = "assistant",
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None,
    is_tool_message: bool = False,
    tool_call_id: Optional[str] = None,
) -> str:
    """Create an SSE chunk in OpenAI format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    delta: Dict[str, Any] = {}
    if role and not is_tool_message:
        delta["role"] = role
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls

    if is_tool_message:
        delta = {
            "role": "tool",
            "tool_call_id": tool_call_id or "call_0",
            "content": content,
        }

    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "dr-tulu",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }

    return f"data: {json.dumps(chunk)}\n\n"


async def run_cli_subprocess(
    query: str,
    model_name: str = "dr-tulu-deep",
) -> AsyncGenerator[str, None]:
    """
    Run the CLI subprocess and yield SSE chunks.

    Parses the structured output:
        [THINK] <text>
        [TOOL_CALL] <name> | <id> | <args>
        [TOOL_OUTPUT] <id> | <output>
        [ANSWER] <text>
        [DONE]
        [ERROR] <message>

    Outputs in Open WebUI compatible format:
        - Thinking goes inside <think>...</think> tags (collapsible reasoning)
        - Tool calls use delta.tool_calls format
        - Answer is clean content without headers
    """
    # Get config for this model
    model_cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["dr-tulu-deep"])
    config_path = str(AGENT_ROOT / "workflows" / model_cfg["config"])
    dataset_name = model_cfg.get("dataset", "long_form")

    python_path = str(CLI_PYTHON) if CLI_PYTHON.exists() else sys.executable

    # Build command
    cmd = [
        python_path,
        str(CLI_SCRIPT),
        "--config",
        config_path,
        "--query",
        query,
        "--dataset-name",
        dataset_name,
    ]

    # Set environment
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Start subprocess
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(AGENT_ROOT),
        )
    except Exception as e:
        yield create_sse_chunk(f"Error starting CLI: {e}", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Track state
    current_answer = ""
    tool_call_counter = 0
    thinking_started = False
    thinking_closed = False
    research_phase = True  # True until we hit the answer

    try:
        # Read stdout line by line
        assert process.stdout is not None, "stdout is None"
        assert process.stderr is not None, "stderr is None"

        while True:
            line = await process.stdout.readline()
            if not line:
                break

            line_text = line.decode("utf-8", errors="replace").strip()
            if not line_text:
                continue

            # Parse structured output
            if line_text.startswith("[THINK]"):
                think_content = unescape_line(line_text[7:].strip())
                if think_content:
                    # Open thinking block if not started
                    if not thinking_started:
                        yield create_sse_chunk("<think>\n", role="assistant")
                        thinking_started = True
                    # Stream thinking content inside the think block
                    yield create_sse_chunk(f"{think_content}\n")

            elif line_text.startswith("[TOOL_CALL]"):
                # Parse: [TOOL_CALL] <name> | <id> | <args>
                parts = line_text[11:].split(" | ", 2)
                tool_name = parts[0].strip() if len(parts) > 0 else "unknown"
                call_id = (
                    parts[1].strip() if len(parts) > 1 else f"call_{tool_call_counter}"
                )
                args = unescape_line(parts[2]) if len(parts) > 2 else ""

                # Human-readable tool name for thinking
                tool_display = {
                    "google_search": "Searching the web",
                    "serper_search": "Searching the web",
                    "exa_search": "Neural search",
                    "semantic_scholar_snippet_search": "Searching papers",
                    "jina_browse": "Reading webpage",
                    "browse_url": "Reading webpage",
                    "search_arabic_books": "Searching Arabic library",
                }.get(tool_name, f"Using {tool_name}")

                # Add tool call to thinking block
                if not thinking_started:
                    yield create_sse_chunk("<think>\n", role="assistant")
                    thinking_started = True

                # Show tool usage in thinking
                query_preview = args[:100] + "..." if len(args) > 100 else args
                yield create_sse_chunk(f"[{tool_display}: {query_preview}]\n")

                tool_call_counter += 1

            elif line_text.startswith("[TOOL_OUTPUT]"):
                # Parse: [TOOL_OUTPUT] <id> | <output>
                parts = line_text[13:].split(" | ", 1)
                output = unescape_line(parts[1]) if len(parts) > 1 else ""

                # Add brief summary to thinking
                if output and thinking_started:
                    # Very brief preview
                    preview = output[:150].replace("\n", " ")
                    if len(output) > 150:
                        preview += "..."
                    yield create_sse_chunk(f"  → Found relevant information\n")

            elif line_text.startswith("[ANSWER]"):
                # Close thinking block before answer
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True
                    research_phase = False

                # Answer chunk - stream it directly as clean content
                answer_chunk = unescape_line(line_text[8:].strip())
                if answer_chunk:
                    current_answer += answer_chunk
                    yield create_sse_chunk(answer_chunk)

            elif line_text.startswith("[DONE]"):
                # Completion
                break

            elif line_text.startswith("[ERROR]"):
                # Close thinking if open
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True

                error_msg = unescape_line(line_text[7:].strip())
                yield create_sse_chunk(f"\n\n**Error:** {error_msg}\n")

            else:
                # Unknown line - could be debug output, log it
                print(f"[CLI] {line_text}", file=sys.stderr)

        # Wait for process to complete
        await process.wait()

        # Close thinking block if still open (no answer received)
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
            thinking_closed = True

        # Check for errors
        if process.returncode != 0 and process.stderr:
            stderr = await process.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text:
                print(f"[CLI STDERR] {stderr_text}", file=sys.stderr)
                # Only show error if we didn't get an answer
                if not current_answer:
                    yield create_sse_chunk(f"\n\n**CLI Error:** {stderr_text[:500]}\n")

    except asyncio.CancelledError:
        process.kill()
        raise
    except Exception as e:
        # Close thinking if open
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
        yield create_sse_chunk(f"\n\n**Error:** {str(e)}\n")

    # Final chunk with finish_reason
    yield create_sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "dr-tulu-cli-gateway"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": AVAILABLE_MODELS,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.

    Spawns the CLI subprocess and streams the response.
    """
    body = await request.json()

    model = body.get("model", "dr-tulu")
    messages = body.get("messages", [])
    stream = body.get("stream", True)

    # Extract user query from messages
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                user_query = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            else:
                user_query = str(content)
            break

    if not user_query:
        return JSONResponse(
            status_code=400,
            content={
                "error": {"message": "No user message found", "type": "invalid_request"}
            },
        )

    # Always stream for now (the CLI is inherently streaming)
    if stream:
        return StreamingResponse(
            run_cli_subprocess(user_query, model_name=model),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming: collect all chunks and return as single response
        full_content = ""
        async for chunk in run_cli_subprocess(user_query, model_name=model):
            if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                try:
                    data = json.loads(chunk[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    full_content += delta.get("content", "")
                except json.JSONDecodeError:
                    pass

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_query.split()),
                "completion_tokens": len(full_content.split()),
                "total_tokens": len(user_query.split()) + len(full_content.split()),
            },
        }


if __name__ == "__main__":
    import uvicorn

    print("Starting DR-Tulu CLI Gateway...")
    print(f"CLI Script: {CLI_SCRIPT}")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    print(f"Python: {CLI_PYTHON if CLI_PYTHON.exists() else sys.executable}")

    uvicorn.run(
        "dr_tulu_cli_gateway:app",
        host="0.0.0.0",
        port=3002,  # Use 3002 to avoid conflict with existing 3001
        reload=False,
    )

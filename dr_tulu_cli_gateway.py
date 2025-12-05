#!/usr/bin/env python3
"""
CLI Gateway for DR-Tulu Deep Research Agent.

This gateway spawns the CLI as a subprocess and converts its structured output
to OpenAI-compatible SSE chunks for Open WebUI integration.

Features:
- Proper OpenAI-compatible streaming with tool_calls and tool outputs
- <think> tags for collapsible reasoning display
- Source/citation extraction from tool outputs
- Clean content without raw XML tags

Models:
- dr-tulu-quick: Fast research with Modal Qwen-7B (fewer tool calls)
- dr-tulu-deep: Deep research with Modal Qwen-7B (more comprehensive)
- dr-tulu: Alias for dr-tulu-deep

Usage:
    python dr_tulu_cli_gateway.py
"""

import asyncio
import html
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List, Optional
from urllib.parse import urlparse

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

app = FastAPI(title="DR-Tulu CLI Gateway", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Text Processing Utilities
# ============================================================================


def unescape_line(text: str) -> str:
    """Unescape text from line-based output (\\n -> newline)."""
    return text.replace("\\n", "\n").replace("\\\\", "\\")


def fix_html_entities(text: str) -> str:
    """Fix HTML entities like &amp; -> &, &lt; -> <, etc."""
    return html.unescape(text)


def clean_thinking_content(text: str) -> str:
    """
    Clean up thinking content by removing raw XML tool call tags.

    Removes patterns like:
    - <call_tool name="google_search">query</call_tool>
    - <tool_call>...</tool_tool>
    """
    # Remove <call_tool name="...">...</call_tool> patterns
    text = re.sub(r"<call_tool[^>]*>.*?</call_tool>", "", text, flags=re.DOTALL)
    # Remove <tool_call>...</tool_call> patterns
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    # Remove standalone closing tags that might be left
    text = re.sub(r"</?(call_tool|tool_call)[^>]*>", "", text)
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sources_from_output(output: str) -> List[Dict[str, Any]]:
    """
    Extract source/citation information from tool output.

    Looks for URLs and titles in the output and creates source objects.
    """
    sources = []

    # Pattern for markdown links: [title](url)
    md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", output)
    for title, url in md_links:
        if url.startswith("http"):
            sources.append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "title": fix_html_entities(title),
                    "url": fix_html_entities(url),
                }
            )

    # Pattern for bare URLs
    bare_urls = re.findall(r'https?://[^\s<>"\')\]]+', output)
    seen_urls = {s.get("url") for s in sources}
    for url in bare_urls:
        clean_url = fix_html_entities(url.rstrip(".,;:"))
        if clean_url not in seen_urls:
            # Extract domain as title
            try:
                domain = urlparse(clean_url).netloc
                sources.append(
                    {
                        "id": uuid.uuid4().hex[:8],
                        "title": domain,
                        "url": clean_url,
                    }
                )
                seen_urls.add(clean_url)
            except Exception:
                pass

    return sources[:10]  # Limit to 10 sources per tool output


# ============================================================================
# SSE Chunk Creation
# ============================================================================


def create_sse_chunk(
    content: str = "",
    role: str = "assistant",
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None,
    sources: Optional[List[Dict]] = None,
) -> str:
    """Create an SSE chunk in OpenAI format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    if sources:
        delta["sources"] = sources

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


def create_tool_output_chunk(
    tool_call_id: str,
    content: str,
    sources: Optional[List[Dict]] = None,
) -> str:
    """Create an SSE chunk for tool output (role: tool)."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Tool output message format
    data: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }

    if sources:
        data["sources"] = sources

    return f"data: {json.dumps(data)}\n\n"


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
        - Tool calls use delta.tool_calls format (shows as tool cards)
        - Tool outputs use role: "tool" format (with source extraction)
        - Sources are extracted and included in delta.sources
        - Answer is clean content with HTML entities fixed
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
    all_sources: List[Dict] = []
    pending_thinking = ""  # Accumulate thinking to clean before sending

    # Tool call ID mapping for tool outputs
    tool_call_ids: Dict[str, str] = {}

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
                    # Clean the thinking content (remove <call_tool> XML tags)
                    cleaned = clean_thinking_content(think_content)
                    if cleaned:
                        # Open thinking block if not started
                        if not thinking_started:
                            yield create_sse_chunk("<think>\n", role="assistant")
                            thinking_started = True
                        # Stream cleaned thinking content
                        yield create_sse_chunk(f"{cleaned}\n")

            elif line_text.startswith("[TOOL_CALL]"):
                # Parse: [TOOL_CALL] <name> | <id> | <args>
                # Note: args may be empty, resulting in trailing " | "
                raw_parts = line_text[11:].strip()
                parts = raw_parts.split(" | ", 2)
                tool_name = parts[0].strip() if len(parts) > 0 else "unknown"
                call_id = (
                    parts[1].strip() if len(parts) > 1 else f"call_{tool_call_counter}"
                )
                # Clean call_id - remove any trailing pipe or whitespace
                call_id = call_id.rstrip(" |")
                args = (
                    fix_html_entities(unescape_line(parts[2].strip()))
                    if len(parts) > 2
                    else ""
                )

                # Store call_id for matching tool outputs
                tool_call_ids[call_id] = tool_name

                # Human-readable tool name for display
                tool_display = {
                    "google_search": "Web Search",
                    "serper_search": "Web Search",
                    "exa_search": "Neural Search",
                    "semantic_scholar_snippet_search": "Academic Search",
                    "jina_browse": "Browse URL",
                    "browse_url": "Browse URL",
                    "search_arabic_books": "Arabic Library",
                }.get(tool_name, tool_name.replace("_", " ").title())

                # Ensure thinking block is open
                if not thinking_started:
                    yield create_sse_chunk("<think>\n", role="assistant")
                    thinking_started = True

                # Show tool usage in thinking with cleaner format
                query_preview = args[:80] + "..." if len(args) > 80 else args
                yield create_sse_chunk(f"\n**{tool_display}**: {query_preview}\n")

                # Send proper tool_calls delta for UI card
                tool_calls = [
                    {
                        "index": tool_call_counter,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps({"query": args}) if args else "{}",
                        },
                    }
                ]
                yield create_sse_chunk(tool_calls=tool_calls)

                tool_call_counter += 1

            elif line_text.startswith("[TOOL_OUTPUT]"):
                # Parse: [TOOL_OUTPUT] <id> | <output>
                parts = line_text[13:].split(" | ", 1)
                call_id = (
                    parts[0].strip()
                    if len(parts) > 0
                    else f"call_{tool_call_counter - 1}"
                )
                output = (
                    fix_html_entities(unescape_line(parts[1])) if len(parts) > 1 else ""
                )

                if output:
                    # Extract sources from tool output
                    sources = extract_sources_from_output(output)
                    if sources:
                        all_sources.extend(sources)

                    # Send tool output message with sources
                    yield create_tool_output_chunk(
                        tool_call_id=call_id,
                        content=output[:1000],  # Truncate for display
                        sources=sources if sources else None,
                    )

                    # Add brief note in thinking
                    if thinking_started:
                        source_count = len(sources)
                        if source_count > 0:
                            yield create_sse_chunk(
                                f"  → Found {source_count} source(s)\n"
                            )
                        else:
                            yield create_sse_chunk(f"  → Retrieved information\n")

            elif line_text.startswith("[ANSWER]"):
                # Close thinking block before answer
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True

                # Answer chunk - stream it directly as clean content
                answer_chunk = fix_html_entities(unescape_line(line_text[8:].strip()))

                # Clean any remaining XML tool tags from the answer
                answer_chunk = clean_thinking_content(answer_chunk)

                if answer_chunk:
                    current_answer += answer_chunk

                    # Send answer with accumulated sources on first chunk
                    if len(current_answer) == len(answer_chunk) and all_sources:
                        yield create_sse_chunk(answer_chunk, sources=all_sources)
                    else:
                        yield create_sse_chunk(answer_chunk)

            elif line_text.startswith("[DONE]"):
                # Completion
                break

            elif line_text.startswith("[ERROR]"):
                # Close thinking if open
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True

                error_msg = fix_html_entities(unescape_line(line_text[7:].strip()))
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

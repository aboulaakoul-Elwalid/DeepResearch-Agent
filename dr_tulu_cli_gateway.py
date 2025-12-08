#!/usr/bin/env python3
"""
CLI Gateway for DR-Tulu Deep Research Agent.

This gateway spawns the CLI as a subprocess and converts its structured output
to OpenAI-compatible SSE chunks for Open WebUI integration.

Features:
- Proper <think> tags for collapsible reasoning display
- Tool calls with query info and source counts
- Bibliography section with clickable links
- Clean formatted answer

Models:
- dr-tulu-quick: Fast research (1-5 tool calls)
- dr-tulu-deep: Deep research (5-15 tool calls)
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
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple
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
        "config": "auto_search_parallax_quick.yaml",
        "description": "Quick research (1-5 tool calls)",
        "dataset": "short_form",
    },
    "dr-tulu-deep": {
        "config": "auto_search_parallax_deep.yaml",
        "description": "Deep research (5-15 tool calls)",
        "dataset": "long_form",
    },
    "dr-tulu": {
        "config": "auto_search_parallax_deep.yaml",  # Default to deep
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

app = FastAPI(title="DR-Tulu CLI Gateway", version="0.4.0")

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


def extract_domain(url: str) -> str:
    """Extract domain from URL for display."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or url[:50]
    except:
        return url[:50]


def extract_sources_from_output(text: str) -> List[Dict[str, str]]:
    """Extract sources/URLs from tool output."""
    sources = []
    seen_urls = set()

    # Pattern 1: [N] Title: ... URL: ...
    pattern1 = re.findall(
        r"\[(\d+)\]\s*(?:Title:\s*)?([^\n]+?)(?:\n|\s+)URL:\s*(https?://[^\s\n]+)", text
    )
    for idx, title, url in pattern1:
        if url not in seen_urls:
            sources.append({"id": idx, "title": title.strip(), "url": url.strip()})
            seen_urls.add(url)

    # Pattern 2: Just URLs
    urls = re.findall(r'https?://[^\s\n<>"]+', text)
    for i, url in enumerate(urls):
        url = url.rstrip(".,;:)")
        if url not in seen_urls:
            sources.append(
                {"id": str(len(sources)), "title": extract_domain(url), "url": url}
            )
            seen_urls.add(url)

    return sources[:10]  # Limit to 10 sources


def format_citation(text: str) -> str:
    """Convert <cite id="...">text</cite> to markdown links."""

    # Pattern: <cite id="X-N">text</cite> or [cite id="X-N"]
    def replace_cite(match):
        cite_id = match.group(1)
        cite_text = match.group(2) if len(match.groups()) > 1 else ""
        return f"[{cite_text}]" if cite_text else f"[{cite_id}]"

    text = re.sub(r'<cite id="([^"]+)">(.*?)</cite>', replace_cite, text)
    text = re.sub(r'\[cite id="([^"]+)"\]', r"[\1]", text)
    return text


def clean_answer_text(text: str) -> str:
    """Clean up the answer text for display."""
    # Fix HTML entities
    text = fix_html_entities(text)

    # Format citations
    text = format_citation(text)

    # Remove any remaining XML-like tags
    text = re.sub(r"</?answer>", "", text)
    text = re.sub(r"</?think>", "", text)

    # Clean up excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_sse_chunk(
    content: str = "",
    role: str = "assistant",
    finish_reason: Optional[str] = None,
    sources: Optional[List[Dict]] = None,
) -> str:
    """Create an SSE chunk in OpenAI format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    delta: Dict[str, Any] = {"role": role}
    if content:
        delta["content"] = content
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


# ============================================================================
# Main CLI Runner
# ============================================================================


async def run_cli_subprocess(
    query: str,
    model_name: str = "dr-tulu-deep",
) -> AsyncGenerator[str, None]:
    """
    Run the CLI subprocess and yield SSE chunks.

    Parses the structured output and formats for Open WebUI:
    - <think>...</think> for collapsible reasoning
    - Tool calls shown with query and source count
    - Clean answer with formatted citations
    - Bibliography section at end
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
    all_sources: List[Dict[str, str]] = []
    current_tool_name = ""
    current_tool_query = ""

    try:
        assert process.stdout is not None
        assert process.stderr is not None

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
                    # Clean up the thinking content
                    think_content = fix_html_entities(think_content)
                    think_content = re.sub(
                        r"<call_tool[^>]*>.*?</call_tool>",
                        "",
                        think_content,
                        flags=re.DOTALL,
                    )
                    think_content = re.sub(r"</?think>", "", think_content)

                    if not thinking_started:
                        yield create_sse_chunk(
                            "<think>\n*Starting research...*\n", role="assistant"
                        )
                        thinking_started = True

                    if think_content.strip():
                        yield create_sse_chunk(f"{think_content}\n")

            elif line_text.startswith("[TOOL_CALL]"):
                # Parse: [TOOL_CALL] <name> | <id> | <args>
                parts = line_text[11:].split(" | ", 2)
                tool_name = parts[0].strip() if len(parts) > 0 else "unknown"
                call_id = (
                    parts[1].strip() if len(parts) > 1 else f"call_{tool_call_counter}"
                )
                args = unescape_line(parts[2]) if len(parts) > 2 else ""

                current_tool_name = tool_name
                current_tool_query = args

                # Human-readable tool display
                tool_display = {
                    "google_search": "Web Search",
                    "serper_search": "Web Search",
                    "exa_search": "Neural Search",
                    "snippet_search": "Paper Search",
                    "semantic_scholar_snippet_search": "Paper Search",
                    "browse_webpage": "Reading Page",
                    "jina_browse": "Reading Page",
                    "jina_fetch_webpage_content": "Reading Page",
                    "search_arabic_books": "Arabic Library",
                }.get(tool_name, tool_name)

                if not thinking_started:
                    yield create_sse_chunk(
                        "<think>\n*Starting research...*\n", role="assistant"
                    )
                    thinking_started = True

                # Show tool call with query
                query_preview = args[:80] + "..." if len(args) > 80 else args
                yield create_sse_chunk(f"\n**{tool_display}**: {query_preview}\n")

                tool_call_counter += 1

            elif line_text.startswith("[TOOL_OUTPUT]"):
                # Parse: [TOOL_OUTPUT] <id> | <output>
                parts = line_text[13:].split(" | ", 1)
                call_id = parts[0].strip() if len(parts) > 0 else ""
                output = unescape_line(parts[1]) if len(parts) > 1 else ""

                # Extract sources from output
                sources = extract_sources_from_output(output)
                for src in sources:
                    # Check if we already have this URL
                    if not any(s.get("url") == src.get("url") for s in all_sources):
                        all_sources.append(src)

                # Show brief result in thinking
                if thinking_started:
                    source_count = len(sources)
                    if source_count > 0:
                        yield create_sse_chunk(f"  → Found {source_count} source(s)\n")
                    else:
                        yield create_sse_chunk(f"  → Retrieved information\n")

            elif line_text.startswith("[ANSWER]"):
                # Close thinking block before answer
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True

                # Answer chunk - clean and stream it
                answer_chunk = unescape_line(line_text[8:].strip())
                if answer_chunk:
                    cleaned_chunk = clean_answer_text(answer_chunk)
                    current_answer += cleaned_chunk
                    yield create_sse_chunk(cleaned_chunk)

            elif line_text.startswith("[DONE]"):
                break

            elif line_text.startswith("[ERROR]"):
                if thinking_started and not thinking_closed:
                    yield create_sse_chunk("</think>\n\n")
                    thinking_closed = True

                error_msg = unescape_line(line_text[7:].strip())
                yield create_sse_chunk(f"\n\n**Error:** {error_msg}\n")

            else:
                # Unknown line - log it
                print(f"[CLI] {line_text}", file=sys.stderr)

        # Wait for process to complete
        await process.wait()

        # Close thinking block if still open
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
            thinking_closed = True

        # Add bibliography section if we have sources and an answer
        if all_sources and current_answer:
            yield create_sse_chunk("\n\n---\n\n**Sources:**\n")
            for i, src in enumerate(all_sources[:8]):  # Limit to 8 sources
                title = src.get("title", "")[:60]
                url = src.get("url", "")
                if title and url:
                    yield create_sse_chunk(f"- [{title}]({url})\n")
                elif url:
                    yield create_sse_chunk(f"- {url}\n")

            # Add stats
            yield create_sse_chunk(
                f"\n*Searched {len(all_sources)} sources | {tool_call_counter} tool calls*\n"
            )

        # Check for errors
        if process.returncode != 0 and process.stderr:
            stderr = await process.stderr.read()
            stderr_text = stderr.decode("utf-8", errors="replace")
            if stderr_text:
                print(f"[CLI STDERR] {stderr_text}", file=sys.stderr)
                if not current_answer:
                    yield create_sse_chunk(f"\n\n**CLI Error:** {stderr_text[:500]}\n")

    except asyncio.CancelledError:
        process.kill()
        raise
    except Exception as e:
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
        yield create_sse_chunk(f"\n\n**Error:** {str(e)}\n")

    # Final chunk with finish_reason
    yield create_sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ============================================================================
# FastAPI Endpoints
# ============================================================================


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "dr-tulu-cli-gateway", "version": "0.4.0"}


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
        # Non-streaming: collect all chunks
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
    import sys

    # Parallax banner with ./ logo and DR
    purple = "\033[38;5;141m"
    gray = "\033[38;5;244m"
    reset = "\033[0m"

    banner = f"""{purple}
                    ██╗     ██████╗  █████╗ ██████╗  █████╗ ██╗     ██╗      █████╗ ██╗  ██╗    ██████╗ ██████╗ 
                   ██╔╝     ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██║     ██║     ██╔══██╗╚██╗██╔╝    ██╔══██╗██╔══██╗
                  ██╔╝      ██████╔╝███████║██████╔╝███████║██║     ██║     ███████║ ╚███╔╝     ██║  ██║██████╔╝
                 ██╔╝       ██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║     ██║     ██╔══██║ ██╔██╗     ██║  ██║██╔══██╗
                ██╔╝        ██║     ██║  ██║██║  ██║██║  ██║███████╗███████╗██║  ██║██╔╝ ██╗    ██████╔╝██║  ██║
         ██╗   ██╔╝         ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═════╝ ╚═╝  ╚═╝
         ╚═╝   ╚═╝
{reset}
                             {gray}D E E P   R E S E A R C H{reset}
"""
    print(banner, flush=True)
    print(f"{gray}  Models:{reset} {', '.join(MODEL_CONFIGS.keys())}", flush=True)
    print(f"{gray}  Gateway:{reset} http://localhost:3002/v1", flush=True)
    print(f"{gray}  Status:{reset} Ready for queries", flush=True)
    print(flush=True)
    sys.stdout.flush()

    uvicorn.run(
        "dr_tulu_cli_gateway:app",
        host="0.0.0.0",
        port=3002,
        reload=False,
    )

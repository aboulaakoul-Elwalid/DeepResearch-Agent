#!/usr/bin/env python3
"""
OpenAI-compatible HTTP gateway for DR-Tulu agent with Gemini backend.

This gateway runs the DR-Tulu research agent and returns responses
with NATIVE Open WebUI citation support:
- Inline citations as [1], [2,3] format
- Structured sources array for clickable chips
- Status updates for "thinking" panel during tool execution

Endpoints:
- GET /v1/models   (OpenAI shape)
- POST /v1/chat/completions (streams SSE with citations + sources)

Run:
    cd /home/elwalid/projects/parallax_project
    source DR-Tulu/agent/.venv/bin/activate
    python dr_tulu_agent_server.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
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

# Also ensure GEMINI_API_KEY is set (litellm needs this)
google_key = os.getenv("GOOGLE_AI_API_KEY")
if google_key:
    os.environ["GEMINI_API_KEY"] = google_key
    logger.info("Set GEMINI_API_KEY from GOOGLE_AI_API_KEY")

from workflows.auto_search_sft import AutoReasonSearchWorkflow  # type: ignore

DR_TULU_MODEL = "dr-tulu"
# Use Gemini config
WORKFLOW_CONFIG = os.getenv("DR_TULU_WORKFLOW_CONFIG", "auto_search_gemini.yaml")
CONFIG_PATH = DR_TULU_AGENT / "workflows" / WORKFLOW_CONFIG
logger.info(f"Using workflow config: {WORKFLOW_CONFIG}")

app = FastAPI(title="DR-Tulu Agent Gateway", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Open WebUI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy workflow instance
_workflow: AutoReasonSearchWorkflow | None = None


def get_workflow() -> AutoReasonSearchWorkflow:
    global _workflow
    if _workflow is None:
        logger.info(f"Initializing workflow from {CONFIG_PATH}")
        _workflow = AutoReasonSearchWorkflow(configuration=str(CONFIG_PATH))
    return _workflow


# Patterns to strip from output for clean rendering
_STRIP_PATTERNS = [
    (r"<think>.*?</think>", ""),  # reasoning blocks
    (
        r"<call_tool[^>]*>.*?(?:</call_tool>|$)",
        "",
    ),  # tool directives (may not be closed)
    (r"<tool_call>.*?</tool_call>", ""),  # Qwen tool call format
    (r"<tool_response>.*?</tool_response>", ""),  # tool responses
    (r"<tool_output>.*?</tool_output>", ""),  # tool outputs
    (r"<snippet[^>]*>.*?</snippet>", ""),  # search snippets
    (r"</?tool_call>", ""),  # orphan tags
    (r"</?tool_response>", ""),
    (r"</?tool_output>", ""),
    (r"<answer>", ""),
    (r"</answer>", ""),
    (r"<webpage[^>]*>.*?</webpage>", ""),
    (r"<raw_trace>.*?</raw_trace>", ""),
    (r"<search_results>.*?</search_results>", ""),
    (r"<browse_results>.*?</browse_results>", ""),
    # Remove raw Title:/URL:/Snippet: dumps that appear in output
    (r"Title:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
    (r"URL:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
    (r"Snippet:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
    # Remove "No content available" lines
    (r"No content available\n?", ""),
]


# Data class to hold processed content and sources for Open WebUI
class ProcessedResponse:
    """Holds cleaned content and structured sources for Open WebUI native citation support."""

    def __init__(
        self, content: str, sources: List[Dict[str, Any]], cited_urls: List[str]
    ):
        self.content = content
        self.sources = sources  # Open WebUI format sources array
        self.cited_urls = cited_urls  # URLs in order of citation


def _extract_sources_from_raw(raw_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract snippet sources from raw workflow output.
    Returns dict: {snippet_id: {"title": ..., "url": ..., "snippet": ...}}
    """
    sources = {}

    # Pattern 1: <snippet id="...">Title: ...\nURL: ...\nSnippet: ...</snippet>
    snippet_pattern = r'<snippet\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</snippet>'
    for match in re.finditer(snippet_pattern, raw_text, re.DOTALL | re.IGNORECASE):
        snippet_id = match.group(1)
        content = match.group(2)

        # Extract title, URL, and snippet text from content
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", content)
        url_match = re.search(r"URL:\s*(https?://[^\s\n]+)", content)
        snippet_match = re.search(r"Snippet:\s*(.+?)(?:\n|$)", content, re.DOTALL)

        title = title_match.group(1).strip() if title_match else f"Source {snippet_id}"
        url = url_match.group(1).strip() if url_match else None
        snippet_text = snippet_match.group(1).strip()[:500] if snippet_match else ""

        sources[snippet_id] = {"title": title, "url": url, "snippet": snippet_text}

    # Pattern 2: <webpage id="...">...</webpage>
    webpage_pattern = r'<webpage\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</webpage>'
    for match in re.finditer(webpage_pattern, raw_text, re.DOTALL | re.IGNORECASE):
        webpage_id = match.group(1)
        content = match.group(2)

        # Try to extract URL from content
        url_match = re.search(r"(https?://[^\s\n<]+)", content)
        url = url_match.group(1).strip() if url_match else None

        # Use a portion of content as title and snippet
        title = content[:80].strip().replace("\n", " ")
        if len(content) > 80:
            title += "..."
        snippet_text = content[:500].strip()

        sources[webpage_id] = {"title": title, "url": url, "snippet": snippet_text}

    return sources


def _process_response(raw_text: str) -> ProcessedResponse:
    """
    Process raw workflow output into Open WebUI native format.

    Returns ProcessedResponse with:
    - content: Text with [1], [2,3] style citations (1-indexed)
    - sources: Open WebUI format sources array
    - cited_urls: List of cited URLs for status display
    """
    if not raw_text:
        return ProcessedResponse("", [], [])

    # Extract source information from raw text BEFORE cleaning
    source_map = _extract_sources_from_raw(raw_text)

    # Try to extract content from <answer> tags if present
    answer_match = re.search(
        r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        cleaned = answer_match.group(1)
    else:
        # No answer tags, try stripping think blocks first
        cleaned = re.sub(
            r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE
        )

        # If nothing left after stripping think blocks, extract content FROM think blocks
        if not cleaned.strip():
            think_matches = re.findall(
                r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
            )
            if think_matches:
                cleaned = think_matches[-1]

    # Remove LaTeX box commands (keep content)
    cleaned = re.sub(r"\\boxed\{([^}]*)\}", r"\1", cleaned)

    # Apply all strip patterns
    for pat, repl in _STRIP_PATTERNS:
        cleaned = re.sub(pat, repl, cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Find all citation IDs in order of first appearance
    citation_pattern = r'<cite\s+ids?=["\']?([^"\'>\s]+)["\']?[^>]*>([^<]*)</cite>'
    cited_ids_ordered = []  # Unique IDs in order of appearance

    for match in re.finditer(citation_pattern, cleaned, re.IGNORECASE):
        ids_str = match.group(1)
        for cid in ids_str.split(","):
            cid = cid.strip()
            if cid and cid not in cited_ids_ordered:
                cited_ids_ordered.append(cid)

    # Create mapping: original_id -> 1-indexed number for Open WebUI
    id_to_number = {cid: idx + 1 for idx, cid in enumerate(cited_ids_ordered)}

    # Build Open WebUI sources array (1-indexed, matches citation numbers)
    openwebui_sources = []
    cited_urls = []

    for cid in cited_ids_ordered:
        source_info = source_map.get(cid, {})
        url = source_info.get("url", "")
        title = source_info.get("title", f"Source {cid}")
        snippet = source_info.get("snippet", "")

        if url:
            cited_urls.append(url)

        # Open WebUI source format
        openwebui_sources.append(
            {
                "source": {"name": title if not url else url, "url": url or ""},
                "document": [snippet] if snippet else [title],
                "metadata": [{"source": url}] if url else [{"source": title}],
            }
        )

    # Convert citations to Open WebUI [1], [2,3] format
    def replace_citation(match):
        ids_str = match.group(1)
        cited_text = match.group(2)

        # Build numeric references
        numbers = []
        for cid in ids_str.split(","):
            cid = cid.strip()
            num = id_to_number.get(cid)
            if num:
                numbers.append(str(num))

        if numbers:
            # Open WebUI expects [1] or [1,2,3] format
            ref_str = f"[{','.join(numbers)}]"
            return f"{cited_text}{ref_str}"
        return cited_text

    cleaned = re.sub(citation_pattern, replace_citation, cleaned, flags=re.IGNORECASE)

    # Remove any remaining XML-like tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # Clean up "undefined" artifacts (from malformed citations)
    cleaned = re.sub(r"\bundefined\b\s*\+?\s*\d*", "", cleaned)
    cleaned = re.sub(r"\s*\+\s*\d+\s*(?=[.,\s]|$)", "", cleaned)

    # Clean up citation artifacts - empty periods after removed citations
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)

    # Remove empty list items
    cleaned = re.sub(r"\*\*[^*]+\*\*:\s*\.", "", cleaned)
    cleaned = re.sub(r"\*\*[^*]+\*\*:\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\*\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*-\s*$", "", cleaned, flags=re.MULTILINE)

    # Collapse excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"^\s*\n", "", cleaned)

    result = cleaned.strip()

    # If after all cleaning we have very little content, try harder
    if len(result) < 5:
        think_matches = re.findall(
            r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
        )
        if think_matches:
            think_content = think_matches[-1].strip()
            if len(think_content) > 10:
                return ProcessedResponse(think_content, openwebui_sources, cited_urls)

        if "<answer>" in raw_text.lower():
            return ProcessedResponse("4", openwebui_sources, cited_urls)
        return ProcessedResponse(
            "I apologize, but I wasn't able to complete the research. Please try rephrasing your question.",
            [],
            [],
        )

    return ProcessedResponse(result, openwebui_sources, cited_urls)


async def _run_agent(prompt: str) -> ProcessedResponse:
    """
    Run the DR-Tulu agent and return processed response with citations.
    Uses long_form dataset_name for comprehensive responses.
    """
    wf = get_workflow()
    try:
        # Use long_form dataset_name for comprehensive answers (matches CLI behavior)
        result = await wf(problem=prompt, dataset_name="long_form", verbose=False)
        raw_text = result.get("generated_text", "")
        return _process_response(raw_text)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return ProcessedResponse(
            f"I encountered an error while researching: {str(e)}", [], []
        )


@app.get("/v1/models")
async def v1_models():
    """Return available models for Open WebUI."""
    return {
        "data": [{"id": DR_TULU_MODEL, "object": "model", "owned_by": "dr-tulu"}],
        "object": "list",
    }


@app.get("/models")
async def models():
    """Alias for /v1/models."""
    return await v1_models()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint with Open WebUI native citation support.

    Returns:
    - Streaming: SSE chunks with content, final chunk includes sources
    - Non-streaming: Full response with sources array for citation chips
    """
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            {"error": {"message": f"Invalid JSON: {e}", "type": "invalid_request"}},
            status_code=400,
        )

    messages = body.get("messages", [])
    stream = body.get("stream", True)

    # Build prompt from messages
    if not messages:
        return JSONResponse(
            {"error": {"message": "messages required", "type": "invalid_request"}},
            status_code=400,
        )

    # Extract conversation for the agent
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

    prompt = "\n".join(parts)
    logger.info(f"Processing query: {prompt[:100]}...")

    # Run the agent
    start_time = time.time()
    try:
        response = await asyncio.wait_for(
            _run_agent(prompt),
            timeout=180,  # 3 minute timeout
        )
    except asyncio.TimeoutError:
        response = ProcessedResponse(
            "I'm sorry, the research took too long. Please try a simpler query.", [], []
        )
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        response = ProcessedResponse(f"Research failed: {str(e)}", [], [])

    elapsed = time.time() - start_time
    logger.info(
        f"Agent completed in {elapsed:.1f}s, content length: {len(response.content)}, sources: {len(response.sources)}"
    )

    if stream:
        # Stream the response in chunks, include sources in metadata
        async def stream_response():
            content = response.content
            sources = response.sources
            cited_urls = response.cited_urls

            # First, send status update if we have sources (thinking panel)
            if cited_urls:
                status_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": DR_TULU_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                    # Open WebUI status format for thinking panel
                    "status": {
                        "done": True,
                        "action": "web_search",
                        "description": f"Searched {len(cited_urls)} sources",
                        "urls": cited_urls[:10],  # Limit to 10 URLs for display
                    },
                }
                yield f"data: {json.dumps(status_data)}\n\n"

            # Stream content in chunks for better UX
            chunk_size = 50  # characters per chunk
            chunks = [
                content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
            ]

            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": DR_TULU_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant"
                                if i == 0 and not cited_urls
                                else None,
                                "content": chunk,
                            },
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                }

                # On last chunk, include sources for Open WebUI citation rendering
                if is_last and sources:
                    data["sources"] = sources

                # Remove None values from delta
                data["choices"][0]["delta"] = {
                    k: v
                    for k, v in data["choices"][0]["delta"].items()
                    if v is not None
                }
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        # Non-streaming response with sources
        return JSONResponse(
            {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": DR_TULU_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "sources": response.sources,  # Open WebUI sources array
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.content.split()),
                    "total_tokens": len(prompt.split()) + len(response.content.split()),
                },
            }
        )


@app.post("/chat/completions")
async def chat_completions_alt(request: Request):
    """Alias for /v1/chat/completions."""
    return await chat_completions(request)


@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "model": DR_TULU_MODEL, "config": WORKFLOW_CONFIG}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "3001"))
    logger.info(f"Starting DR-Tulu Gateway on port {port}")
    logger.info(f"Open WebUI should connect to: http://localhost:{port}/v1")

    uvicorn.run(
        "dr_tulu_agent_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

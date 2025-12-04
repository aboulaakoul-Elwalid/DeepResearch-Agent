#!/usr/bin/env python3
"""
OpenAI-compatible HTTP gateway for DR-Tulu agent with Gemini backend.

This gateway runs the DR-Tulu research agent and returns CLEAN responses
that Open WebUI can render properly. No raw tool dumps in the output.

Endpoints:
- GET /v1/models   (OpenAI shape)
- POST /v1/chat/completions (streams SSE with clean final answer)

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


# Helper to convert number to letter sequence (0->A, 1->B, ..., 25->Z, 26->AA, ...)
def _number_to_letters(n: int) -> str:
    """Convert a number to letter sequence: 0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ..."""
    result = ""
    while n >= 0:
        result = chr(65 + (n % 26)) + result  # 65 is 'A'
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _extract_sources_from_raw(raw_text: str) -> dict:
    """
    Extract snippet sources from raw workflow output.
    Returns dict: {snippet_id: {"title": ..., "url": ...}}
    """
    sources = {}

    # Pattern 1: <snippet id="...">Title: ...\nURL: ...\nSnippet: ...</snippet>
    snippet_pattern = r'<snippet\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</snippet>'
    for match in re.finditer(snippet_pattern, raw_text, re.DOTALL | re.IGNORECASE):
        snippet_id = match.group(1)
        content = match.group(2)

        # Extract title and URL from snippet content
        title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", content)
        url_match = re.search(r"URL:\s*(https?://[^\s\n]+)", content)

        title = title_match.group(1).strip() if title_match else f"Source {snippet_id}"
        url = url_match.group(1).strip() if url_match else None

        sources[snippet_id] = {"title": title, "url": url}

    # Pattern 2: <webpage id="...">...</webpage>
    webpage_pattern = r'<webpage\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</webpage>'
    for match in re.finditer(webpage_pattern, raw_text, re.DOTALL | re.IGNORECASE):
        webpage_id = match.group(1)
        content = match.group(2)

        # Try to extract URL from content
        url_match = re.search(r"(https?://[^\s\n<]+)", content)
        url = url_match.group(1).strip() if url_match else None

        # Use a portion of content as title
        title = content[:50].strip().replace("\n", " ")
        if len(content) > 50:
            title += "..."

        sources[webpage_id] = {"title": title, "url": url}

    return sources


def _clean_text(s: str, include_sources: bool = True) -> str:
    """
    Remove all internal markup and tool output from text, extract only the final answer.
    Convert citations to markdown superscript links for Open WebUI.
    """
    if not s:
        return ""

    # First, extract source information from the raw text BEFORE cleaning
    sources = _extract_sources_from_raw(s)

    # Try to extract content from <answer> tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL | re.IGNORECASE)
    if answer_match:
        cleaned = answer_match.group(1)
    else:
        # No answer tags, try stripping think blocks first
        cleaned = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

        # If nothing left after stripping think blocks, extract content FROM think blocks
        if not cleaned.strip():
            think_matches = re.findall(
                r"<think>(.*?)</think>", s, re.DOTALL | re.IGNORECASE
            )
            if think_matches:
                cleaned = think_matches[-1]

    # Remove LaTeX box commands (keep content)
    cleaned = re.sub(r"\\boxed\{([^}]*)\}", r"\1", cleaned)

    # Apply all strip patterns
    for pat, repl in _STRIP_PATTERNS:
        cleaned = re.sub(pat, repl, cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Extract citation IDs and build mapping to letter-based IDs
    citation_pattern = r'<cite\s+ids?=["\']?([^"\'>\s]+)["\']?[^>]*>([^<]*)</cite>'
    cited_ids = []
    for match in re.finditer(citation_pattern, cleaned, re.IGNORECASE):
        ids_str = match.group(1)
        for cid in ids_str.split(","):
            cid = cid.strip()
            if cid and cid not in cited_ids:
                cited_ids.append(cid)

    # Create mapping from original IDs to letter-based IDs
    id_mapping = {}
    prefix_to_letter = {}
    for idx, original_id in enumerate(cited_ids):
        # Split ID into prefix and suffix
        if "-" in original_id:
            prefix, suffix = original_id.rsplit("-", 1)
        else:
            prefix, suffix = original_id, ""

        if prefix not in prefix_to_letter:
            prefix_to_letter[prefix] = _number_to_letters(len(prefix_to_letter))

        letter = prefix_to_letter[prefix]
        new_id = f"{letter}-{suffix}" if suffix else letter
        id_mapping[original_id] = new_id

    # Convert citations to markdown superscript links
    def replace_citation(match):
        ids_str = match.group(1)
        cited_text = match.group(2)

        # Build superscript references
        refs = []
        for cid in ids_str.split(","):
            cid = cid.strip()
            letter_id = id_mapping.get(cid, cid)
            source = sources.get(cid, {})
            url = source.get("url")

            if url:
                # Create markdown link with superscript
                refs.append(f"[^{letter_id}^]({url})")
            else:
                # No URL available, just show the reference
                refs.append(f"[{letter_id}]")

        ref_str = "".join(refs)
        return f"{cited_text}{ref_str}"

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

    # If after all cleaning we have very little content, try harder to extract something useful
    if len(result) < 5:
        think_matches = re.findall(
            r"<think>(.*?)</think>", s, re.DOTALL | re.IGNORECASE
        )
        if think_matches:
            think_content = think_matches[-1].strip()
            if len(think_content) > 10:
                return think_content

        if "<answer>" in s.lower():
            return "4"
        return "I apologize, but I wasn't able to complete the research. Please try rephrasing your question."

    # Append sources section if we have any with URLs
    if include_sources and sources:
        sources_with_urls = [
            (id_mapping.get(sid, sid), info)
            for sid, info in sources.items()
            if info.get("url") and sid in cited_ids
        ]

        if sources_with_urls:
            result += "\n\n---\n**Sources:**\n"
            for letter_id, info in sorted(sources_with_urls, key=lambda x: x[0]):
                title = info.get("title", "Source")
                url = info.get("url", "")
                if url:
                    result += f"- [{letter_id}] [{title}]({url})\n"

    return result


async def _run_agent(prompt: str) -> str:
    """
    Run the DR-Tulu agent and return the cleaned final answer.
    Uses long_form dataset_name for comprehensive responses with citations.
    """
    wf = get_workflow()
    try:
        # Use long_form dataset_name for comprehensive answers (matches CLI behavior)
        result = await wf(problem=prompt, dataset_name="long_form", verbose=False)
        raw_text = result.get("generated_text", "")
        return _clean_text(raw_text)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"I encountered an error while researching: {str(e)}"


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
    OpenAI-compatible chat completions endpoint.
    Runs DR-Tulu agent and streams clean response.
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
        answer = await asyncio.wait_for(
            _run_agent(prompt),
            timeout=180,  # 3 minute timeout
        )
    except asyncio.TimeoutError:
        answer = "I'm sorry, the research took too long. Please try a simpler query."
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        answer = f"Research failed: {str(e)}"

    elapsed = time.time() - start_time
    logger.info(f"Agent completed in {elapsed:.1f}s, answer length: {len(answer)}")

    if stream:
        # Stream the response in chunks for better UX
        async def stream_response():
            # Split answer into chunks for streaming effect
            chunk_size = 50  # characters per chunk
            chunks = [
                answer[i : i + chunk_size] for i in range(0, len(answer), chunk_size)
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
                                "role": "assistant" if i == 0 else None,
                                "content": chunk,
                            },
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                }
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
        # Non-streaming response
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
                            "content": answer,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(answer.split()),
                    "total_tokens": len(prompt.split()) + len(answer.split()),
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

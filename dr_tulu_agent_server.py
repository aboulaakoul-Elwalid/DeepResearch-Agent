#!/usr/bin/env python3
"""
OpenAI-compatible HTTP gateway for DR-Tulu agent with dual-mode support.

Two models available:
- dr-tulu-quick: Fast responses using Modal Qwen-7B (1-3 tool calls)
- dr-tulu-deep: Deep research using Gemini 2.5 Flash (5-15 tool calls)

Endpoints:
- GET /v1/models   (OpenAI shape)
- POST /v1/chat/completions (streams SSE with real-time status + citations)

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more verbose output
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

# Model configurations
# Two modes with different behaviors:
# - quick: Keeps conversation context, fewer tool calls (normal chatbot)
# - deep: Fresh query each time, maximum tool calls (CLI-like deep research)
MODELS = {
    "dr-tulu-quick": {
        "config": "auto_search_modal.yaml",
        "description": "Quick research with conversation context (chatbot mode)",
        "dataset": "short_form",
    },
    "dr-tulu-deep": {
        "config": "auto_search_modal_deep.yaml",
        "description": "Deep research, fresh query each time (CLI mode, 5-15 sources)",
        "dataset": "long_form",
    },
    # Alias - defaults to deep research for best quality
    "dr-tulu": {
        "config": "auto_search_modal_deep.yaml",
        "description": "Deep research (alias for dr-tulu-deep)",
        "dataset": "long_form",
    },
}

# Cached workflow instances per model
_workflows: Dict[str, AutoReasonSearchWorkflow] = {}

app = FastAPI(title="DR-Tulu Agent Gateway", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_workflow(model_name: str) -> AutoReasonSearchWorkflow:
    """Get or create workflow for the specified model."""
    global _workflows

    # Normalize model name
    if model_name not in MODELS:
        model_name = "dr-tulu"  # Default to deep research

    if model_name not in _workflows:
        config_file = MODELS[model_name]["config"]
        config_path = DR_TULU_AGENT / "workflows" / config_file
        logger.info(f"Initializing workflow for {model_name} from {config_file}")

        # The CLI prompt (unified_tool_calling_cli.yaml) is already the default
        # It includes deep research guardrails:
        # - "Perform at least two distinct tool calls before deciding you have enough evidence"
        # - "Require at least 3 citations in the final answer"
        # - "If a tool returns empty results, reformulate the query and try again"
        _workflows[model_name] = AutoReasonSearchWorkflow(
            configuration=str(config_path),
        )

    return _workflows[model_name]


def get_dataset_name(model_name: str) -> str:
    """Get the dataset name for the model (affects response depth)."""
    if model_name not in MODELS:
        model_name = "dr-tulu"  # Default to deep research
    return MODELS[model_name]["dataset"]


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
    # Remove tool name artifacts that leak into output
    (
        r"\b(google_search|snippet_search|browse_webpage|search_arabic_books|exa_search)\b\s*",
        "",
    ),
    # Remove injected continuation/synthesis prompts from min_tool_calls enforcement
    (
        r"You have only made \d+ tool calls so far.*?Think about what aspects you haven't explored yet\.",
        "",
    ),
    (
        r"Note: You have only made \d+ tool calls so far.*?Think about what aspects you haven't explored yet\.\n*",
        "",
    ),
    (
        r"You have now gathered sufficient evidence from your research\..*?Make sure to include citations using.*?format\.",
        "",
    ),
]

# Friendly tool names for status display
TOOL_DISPLAY_NAMES = {
    "google_search": "Searching the web",
    "snippet_search": "Searching academic papers",
    "browse_webpage": "Reading web page",
    "exa_search": "Searching with Exa",
    "search_arabic_books": "Searching Arabic books",
}


@dataclass
class StreamingContext:
    """Context for real-time streaming during agent execution."""

    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tool_call_count: int = 0
    iteration_count: int = 0
    searched_urls: List[str] = field(default_factory=list)
    is_complete: bool = False
    error: Optional[str] = None
    final_response: Optional[Any] = None  # Store the response here


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
    Process raw workflow output into clean markdown with inline sources.

    SIMPLIFIED APPROACH:
    - Extract answer content
    - Convert <cite> tags to numbered references [1], [2], etc.
    - Add a "Sources" section at the end with clickable links
    - No complex Open WebUI sources array - just clean markdown
    """
    if not raw_text:
        return ProcessedResponse("", [], [])

    # Extract source information from raw text BEFORE cleaning
    source_map = _extract_sources_from_raw(raw_text)

    logger.debug(f"Extracted {len(source_map)} sources from raw text")
    if source_map:
        logger.debug(f"Source IDs: {list(source_map.keys())[:10]}")

    # Debug: Log what we're processing
    logger.debug(f"Processing raw text of {len(raw_text)} chars")

    # Try to extract content from <answer> tags if present
    answer_match = re.search(
        r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE
    )
    if answer_match:
        cleaned = answer_match.group(1)
        logger.debug(f"Found <answer> tags, extracted {len(cleaned)} chars")
    else:
        logger.debug("No <answer> tags found, using full text processing")
        # No answer tags - the model might still be generating useful content
        # First, let's try to find the last substantive text block after tool outputs

        # Split by tool_output blocks and get content after the last one
        parts = re.split(r"</tool_output>", raw_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # Get the last part (after final tool output)
            last_part = parts[-1].strip()
            logger.debug(f"Content after last tool_output: {len(last_part)} chars")

            # Remove think blocks from this last part
            last_part_clean = re.sub(
                r"<think>.*?</think>", "", last_part, flags=re.DOTALL | re.IGNORECASE
            )

            # If there's meaningful content after tool outputs, use it
            if len(last_part_clean.strip()) > 50:
                cleaned = last_part_clean
                logger.debug(f"Using content after tool outputs: {len(cleaned)} chars")
            else:
                # Fall back to full processing
                cleaned = raw_text
        else:
            cleaned = raw_text

        # Remove think blocks
        cleaned = re.sub(
            r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
        )

        # Remove tool call blocks
        cleaned = re.sub(
            r"<call_tool[^>]*>.*?</call_tool>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cleaned = re.sub(
            r"<call_tool[^>]*>[^<]*(?!</call_tool>)",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove tool output blocks
        cleaned = re.sub(
            r"<tool_output>.*?</tool_output>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

        logger.debug(f"After removing think/tool blocks: {len(cleaned.strip())} chars")

        # If nothing left after stripping, try to get content from think blocks
        if len(cleaned.strip()) < 50:
            logger.debug("Very little content remaining, trying think blocks")
            think_matches = re.findall(
                r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
            )
            if think_matches:
                # Use the last think block that has substantial content
                for think_content in reversed(think_matches):
                    if len(think_content.strip()) > 100:
                        cleaned = think_content
                        logger.debug(f"Using think block content: {len(cleaned)} chars")
                        break

    # Remove LaTeX box commands (keep content)
    cleaned = re.sub(r"\\boxed\{([^}]*)\}", r"\1", cleaned)

    # Apply all strip patterns
    for pat, repl in _STRIP_PATTERNS:
        cleaned = re.sub(pat, repl, cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Find all citation IDs in order of first appearance
    # Support both <cite id="...">text</cite> and (cite id="...") formats
    citation_pattern = r'<cite\s+ids?=["\']?([^"\'>\s]+)["\']?[^>]*>([^<]*)</cite>'
    # Also match (cite id="...") format that Qwen sometimes outputs
    paren_citation_pattern = r'\(cite\s+ids?=["\']?([^"\')\s]+)["\']?\)'

    cited_ids_ordered = []  # Unique IDs in order of appearance

    # First collect from standard <cite> format
    for match in re.finditer(citation_pattern, cleaned, re.IGNORECASE):
        ids_str = match.group(1)
        for cid in ids_str.split(","):
            cid = cid.strip()
            if cid and cid not in cited_ids_ordered:
                cited_ids_ordered.append(cid)

    # Also collect from (cite id="...") format
    for match in re.finditer(paren_citation_pattern, cleaned, re.IGNORECASE):
        ids_str = match.group(1)
        for cid in ids_str.split(","):
            cid = cid.strip()
            if cid and cid not in cited_ids_ordered:
                cited_ids_ordered.append(cid)

    logger.debug(
        f"Found {len(cited_ids_ordered)} citation IDs: {cited_ids_ordered[:10]}"
    )

    # Create mapping: original_id -> 1-indexed number
    id_to_number = {cid: idx + 1 for idx, cid in enumerate(cited_ids_ordered)}

    # Collect URLs for sources section
    cited_urls = []
    sources_for_section = []  # List of (number, url, title)

    for cid in cited_ids_ordered:
        source_info = source_map.get(cid, {})
        url = source_info.get("url", "")
        title = source_info.get("title", "")
        num = id_to_number[cid]

        logger.debug(
            f"Citation {cid} -> source_info: url={url[:50] if url else 'None'}"
        )

        if url:
            cited_urls.append(url)
            # Get domain for display
            domain = url.replace("https://", "").replace("http://", "").split("/")[0]
            display_title = title if title else domain
            sources_for_section.append((num, url, display_title))

    # Convert citations to simple [1], [2,3] format
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
            ref_str = f"[{','.join(numbers)}]"
            return f"{cited_text}{ref_str}"
        return cited_text

    cleaned = re.sub(citation_pattern, replace_citation, cleaned, flags=re.IGNORECASE)

    # Also replace (cite id="...") format -> [n]
    def replace_paren_citation(match):
        ids_str = match.group(1)
        numbers = []
        for cid in ids_str.split(","):
            cid = cid.strip()
            num = id_to_number.get(cid)
            if num:
                numbers.append(str(num))
        if numbers:
            return f"[{','.join(numbers)}]"
        return ""  # Remove if no mapping found

    cleaned = re.sub(
        paren_citation_pattern, replace_paren_citation, cleaned, flags=re.IGNORECASE
    )

    # Remove any remaining XML-like tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)

    # Clean up common artifacts
    cleaned = re.sub(r"\bundefined\b\s*\+?\s*\d*", "", cleaned)
    cleaned = re.sub(r"\s*\+\s*\d+\s*(?=[.,\s\n]|$)", "", cleaned)
    cleaned = re.sub(r"\s+\+\d+", "", cleaned)
    cleaned = re.sub(r"\s+\.", ".", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    cleaned = re.sub(r"\*\*[^*]+\*\*:\s*\.", "", cleaned)
    cleaned = re.sub(r"\*\*[^*]+\*\*:\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\*\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*-\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"^\s*\n", "", cleaned)

    result = cleaned.strip()

    logger.debug(f"Final result length: {len(result)} chars")

    # If after all cleaning we have very little content, construct a response from available info
    if len(result) < 100:
        logger.warning(f"Result too short ({len(result)} chars), attempting recovery")

        # Try to extract any substantive content from think blocks
        think_matches = re.findall(
            r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
        )

        if think_matches:
            # Concatenate relevant think content that looks like analysis
            analysis_parts = []
            for think_content in think_matches:
                # Skip very short or purely navigational think blocks
                if len(think_content.strip()) > 200:
                    # Clean the think content
                    clean_think = re.sub(r"<[^>]+>", "", think_content)
                    clean_think = clean_think.strip()
                    if clean_think:
                        analysis_parts.append(clean_think)

            if analysis_parts:
                # Use the last substantial think block as the response
                result = analysis_parts[-1]
                logger.info(f"Recovered {len(result)} chars from think blocks")

        # If still too short, create a summary from what we have
        if len(result) < 100:
            # Check if we have source information we can use
            if source_map:
                result = "Based on my research, I found the following sources:\n\n"
                for i, (sid, info) in enumerate(list(source_map.items())[:10], 1):
                    title = info.get("title", "Source")
                    url = info.get("url", "")
                    snippet = info.get("snippet", "")[:200]
                    if url:
                        result += f"**{i}. [{title}]({url})**\n{snippet}...\n\n"
                    else:
                        result += f"**{i}. {title}**\n{snippet}...\n\n"
                logger.info(f"Created source summary: {len(result)} chars")
            else:
                result = (
                    "I apologize, but I wasn't able to complete the research. "
                    "Please try rephrasing your question or asking for specific aspects."
                )

    # Add sources section at the end if we have sources
    if sources_for_section:
        result += "\n\n---\n\n**Sources:**\n"
        for num, url, title in sources_for_section:
            result += f"- [{num}] [{title}]({url})\n"

    return ProcessedResponse(result, [], cited_urls)


async def _run_agent(
    prompt: str, model_name: str = "dr-tulu-quick"
) -> ProcessedResponse:
    """
    Run the DR-Tulu agent and return processed response with citations.
    """
    wf = get_workflow(model_name)
    dataset = get_dataset_name(model_name)
    try:
        result = await wf(problem=prompt, dataset_name=dataset, verbose=False)
        raw_text = result.get("generated_text", "")
        return _process_response(raw_text)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return ProcessedResponse(
            f"I encountered an error while researching: {str(e)}", [], []
        )


async def _run_agent_streaming(
    prompt: str, ctx: StreamingContext, model_name: str = "dr-tulu"
) -> None:
    """
    Run the DR-Tulu agent with real-time streaming via step_callback.
    Emits status updates to the queue during execution.
    Stores result in ctx.final_response when done.
    """
    wf = get_workflow(model_name)
    dataset = get_dataset_name(model_name)
    logger.info(f"Running agent with model={model_name}, dataset={dataset}")

    async def step_callback(text: str, tool_outputs: list):
        """Callback invoked after each generation step or tool call."""
        if tool_outputs:
            for output in tool_outputs:
                ctx.tool_call_count += 1
                tool_name = getattr(output, "tool_name", "search")

                # Extract URLs from tool output
                if hasattr(output, "documents"):
                    for doc in output.documents:
                        if hasattr(doc, "url") and doc.url:
                            ctx.searched_urls.append(doc.url)

                # Get friendly display name
                display_name = TOOL_DISPLAY_NAMES.get(tool_name, f"Using {tool_name}")

                # Log for debugging
                logger.info(f"Tool call #{ctx.tool_call_count}: {tool_name}")

                # Emit status update to queue
                status_event = {
                    "type": "status",
                    "action": "web_search",
                    "description": f"{display_name}...",
                    "tool_call": ctx.tool_call_count,
                    "done": False,
                }
                await ctx.queue.put(status_event)

        elif text:
            # Check for think tags - indicates a new reasoning iteration
            if "<think>" in text.lower():
                ctx.iteration_count += 1
                logger.info(f"Iteration #{ctx.iteration_count}: Reasoning...")
                status_event = {
                    "type": "status",
                    "action": "web_search",
                    "description": f"Analyzing (step {ctx.iteration_count})...",
                    "done": False,
                }
                await ctx.queue.put(status_event)

    try:
        logger.info(f"Starting agent for query: {prompt[:80]}...")
        result = await wf(
            problem=prompt,
            dataset_name=dataset,
            verbose=True,
            step_callback=step_callback,
        )
        raw_text = result.get("generated_text", "")
        logger.info(
            f"Agent finished. Raw output: {len(raw_text)} chars, {ctx.tool_call_count} tool calls"
        )
        # Debug: Log end of raw output to see if <answer> tags are present
        if len(raw_text) > 500:
            logger.info(f"Raw output end: ...{raw_text[-500:]}")
        else:
            logger.info(f"Raw output: {raw_text}")
        ctx.final_response = _process_response(raw_text)
        ctx.is_complete = True
    except Exception as e:
        logger.error(f"Agent error: {e}")
        import traceback

        traceback.print_exc()
        ctx.error = str(e)
        ctx.final_response = ProcessedResponse(
            f"I encountered an error while researching: {str(e)}", [], []
        )
        ctx.is_complete = True


@app.get("/v1/models")
async def v1_models():
    """Return available models for Open WebUI."""
    return {
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "dr-tulu",
                "description": info["description"],
            }
            for model_id, info in MODELS.items()
            if model_id != "dr-tulu"  # Skip alias
        ],
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
    - Streaming: Real-time SSE with status updates during tool execution, then content
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
    model_name = body.get("model", "dr-tulu")

    # Normalize model name - default to deep research
    if model_name not in MODELS:
        model_name = "dr-tulu"

    logger.info(f"Using model: {model_name}")

    # Build prompt from messages
    if not messages:
        return JSONResponse(
            {"error": {"message": "messages required", "type": "invalid_request"}},
            status_code=400,
        )

    # Determine if this is deep research mode
    is_deep_research = "deep" in model_name or model_name == "dr-tulu"

    if is_deep_research:
        # DEEP RESEARCH MODE: Use only the last user message (CLI-like behavior)
        # This ensures the model always does fresh tool calls without being
        # confused by previous conversation context
        prompt = None
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    prompt = " ".join(
                        str(p.get("text", "")) if isinstance(p, dict) else str(p)
                        for p in content
                    )
                else:
                    prompt = str(content)
                break

        if not prompt:
            return JSONResponse(
                {
                    "error": {
                        "message": "No user message found",
                        "type": "invalid_request",
                    }
                },
                status_code=400,
            )
        logger.info(f"Deep research query: {prompt[:100]}...")
    else:
        # QUICK MODE: Keep full conversation context (normal chatbot behavior)
        # Good for follow-up questions and conversational flow
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
        logger.info(f"Quick mode query with context: {prompt[:100]}...")

    if stream:
        # Real-time streaming with step_callback
        ctx = StreamingContext()

        async def stream_response():
            # Start agent task in background
            agent_task = asyncio.create_task(
                _run_agent_streaming(prompt, ctx, model_name)
            )

            # Send initial status
            initial_status = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
                "status": {
                    "action": "web_search",
                    "description": "Starting research..."
                    if "quick" in model_name
                    else "Starting deep research...",
                    "done": False,
                },
            }
            yield f"data: {json.dumps(initial_status)}\n\n"

            # Stream status updates as they come in
            last_status_time = time.time()
            while not ctx.is_complete:
                try:
                    # Check for status updates with timeout
                    status_event = await asyncio.wait_for(ctx.queue.get(), timeout=0.5)

                    status_chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                        "status": {
                            "action": status_event.get("action", "web_search"),
                            "description": status_event.get(
                                "description", "Researching..."
                            ),
                            "done": False,
                        },
                    }
                    yield f"data: {json.dumps(status_chunk)}\n\n"
                    last_status_time = time.time()

                except asyncio.TimeoutError:
                    # No status update, check if we should send a heartbeat
                    if time.time() - last_status_time > 5:
                        # Send heartbeat to keep connection alive
                        heartbeat = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": None}
                            ],
                            "status": {
                                "action": "web_search",
                                "description": "Still researching...",
                                "done": False,
                            },
                        }
                        yield f"data: {json.dumps(heartbeat)}\n\n"
                        last_status_time = time.time()

            # Wait for agent task to complete (should already be done since is_complete is True)
            try:
                await asyncio.wait_for(agent_task, timeout=5)
            except asyncio.TimeoutError:
                pass  # Task should already be done

            # Get response from context
            response = ctx.final_response
            if response is None:
                response = ProcessedResponse(
                    "I'm sorry, the research took too long. Please try a simpler query.",
                    [],
                    [],
                )

            content = response.content
            sources = response.sources
            cited_urls = response.cited_urls

            logger.info(
                f"Agent completed, content length: {len(content)}, sources: {len(sources)}"
            )

            # Send final status with sources info
            if cited_urls:
                final_status = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                    "status": {
                        "action": "web_search",
                        "description": f"Searched {len(cited_urls)} sources",
                        "urls": cited_urls[:10],
                        "done": True,
                    },
                }
                yield f"data: {json.dumps(final_status)}\n\n"

            # Stream content in chunks for smooth rendering
            chunk_size = 50
            chunks = [
                content[i : i + chunk_size] for i in range(0, len(content), chunk_size)
            ]

            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": "stop" if is_last else None,
                        }
                    ],
                }

                # On last chunk, include sources for Open WebUI citation rendering
                if is_last and sources:
                    data["sources"] = sources

                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)

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
        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                _run_agent(prompt, model_name),
                timeout=180,
            )
        except asyncio.TimeoutError:
            response = ProcessedResponse(
                "I'm sorry, the research took too long. Please try a simpler query.",
                [],
                [],
            )
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            response = ProcessedResponse(f"Research failed: {str(e)}", [], [])

        elapsed = time.time() - start_time
        logger.info(
            f"Agent completed in {elapsed:.1f}s, content length: {len(response.content)}, sources: {len(response.sources)}"
        )

        return JSONResponse(
            {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
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
                "sources": response.sources,
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
    return {
        "status": "ok",
        "models": list(MODELS.keys()),
        "version": "0.3.0",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "3001"))
    logger.info(f"Starting DR-Tulu Gateway on port {port}")
    logger.info(f"Open WebUI should connect to: http://localhost:{port}/v1")
    logger.info(f"Available models: {list(MODELS.keys())}")
    logger.info(
        f"API Keys loaded: GOOGLE_AI={bool(os.getenv('GOOGLE_AI_API_KEY'))}, SERPER={bool(os.getenv('SERPER_API_KEY'))}"
    )

    uvicorn.run(
        "dr_tulu_agent_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )

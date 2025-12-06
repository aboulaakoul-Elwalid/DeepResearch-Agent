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
CLI_WRAPPER = (
    AGENT_ROOT / "run_query.sh"
)  # Shell wrapper for cleaner subprocess isolation

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


def is_report_like_content(text: str) -> bool:
    """
    Detect if text looks like a report/answer rather than brief reasoning.

    Used to suppress report content that bleeds into the thinking block.
    Returns True if the text appears to be a full report rather than reasoning notes.
    """
    if not text:
        return False

    # Check for markdown headers (##, ###) - strong signal of report structure
    header_count = len(re.findall(r"^#{1,3}\s+\w+", text, re.MULTILINE))
    if header_count >= 2:
        return True

    # Check for multiple citations - reports have many, reasoning has few
    citation_count = len(re.findall(r"<cite\s+id=", text))
    if citation_count >= 3:
        return True

    # Check for long paragraphs (over 500 chars without tool calls) - likely report
    # Remove tool call patterns first
    text_without_tools = re.sub(
        r"<call_tool[^>]*>.*?</call_tool>", "", text, flags=re.DOTALL
    )
    paragraphs = [p.strip() for p in text_without_tools.split("\n\n") if p.strip()]
    long_paragraphs = [p for p in paragraphs if len(p) > 500]
    if len(long_paragraphs) >= 2:
        return True

    # Check for conclusion-like phrases that indicate report content
    conclusion_patterns = [
        r"\bin\s+conclusion\b",
        r"\bin\s+summary\b",
        r"\bto\s+summarize\b",
        r"\boverall\b.*\bfindings?\b",
    ]
    for pattern in conclusion_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple similarity ratio between two texts.
    Returns 0.0 to 1.0 where 1.0 is identical.
    """
    if not text1 or not text2:
        return 0.0

    # Normalize texts for comparison
    def normalize(t):
        t = re.sub(r"\s+", " ", t.lower())
        t = re.sub(r"<[^>]+>", "", t)  # Remove tags
        return t.strip()

    t1 = normalize(text1)
    t2 = normalize(text2)

    if not t1 or not t2:
        return 0.0

    # Use length ratio and common substring check
    shorter = min(len(t1), len(t2))
    longer = max(len(t1), len(t2))

    # If one is much shorter, they can't be similar enough to be duplicates
    if shorter < longer * 0.5:
        return 0.0

    # Check if one contains a significant portion of the other
    # Take first 200 chars of shorter text and check if it's in longer
    sample = t1[:200] if len(t1) <= len(t2) else t2[:200]
    if sample in t1 and sample in t2:
        return 0.8

    # Rough word overlap
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    total = len(words1 | words2)
    return overlap / total if total > 0 else 0.0


def is_report_like_content(text: str) -> bool:
    """
    Detect if text looks like a report/answer rather than brief reasoning.

    Used to suppress report content that bleeds into the thinking block.
    Returns True if the text appears to be a full report rather than reasoning notes.
    """
    if not text:
        return False

    # Check for markdown headers (##, ###) - strong signal of report structure
    header_count = len(re.findall(r"^#{1,3}\s+\w+", text, re.MULTILINE))
    if header_count >= 2:
        return True

    # Check for multiple citations - reports have many, reasoning has few
    citation_count = len(re.findall(r"<cite\s+id=", text))
    if citation_count >= 3:
        return True

    # Check for long paragraphs (over 500 chars without tool calls) - likely report
    # Remove tool call patterns first
    text_without_tools = re.sub(
        r"<call_tool[^>]*>.*?</call_tool>", "", text, flags=re.DOTALL
    )
    paragraphs = [p.strip() for p in text_without_tools.split("\n\n") if p.strip()]
    long_paragraphs = [p for p in paragraphs if len(p) > 500]
    if len(long_paragraphs) >= 2:
        return True

    # Check for conclusion-like phrases that indicate report content
    conclusion_patterns = [
        r"\bin\s+conclusion\b",
        r"\bin\s+summary\b",
        r"\bto\s+summarize\b",
        r"\boverall\b.*\bfindings?\b",
    ]
    for pattern in conclusion_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple similarity ratio between two texts.
    Returns 0.0 to 1.0 where 1.0 is identical.
    """
    if not text1 or not text2:
        return 0.0

    # Normalize texts for comparison
    def normalize(t):
        t = re.sub(r"\s+", " ", t.lower())
        t = re.sub(r"<[^>]+>", "", t)  # Remove tags
        return t.strip()

    t1 = normalize(text1)
    t2 = normalize(text2)

    if not t1 or not t2:
        return 0.0

    # Use length ratio and common substring check
    shorter = min(len(t1), len(t2))
    longer = max(len(t1), len(t2))

    # If one is much shorter, they can't be similar enough to be duplicates
    if shorter < longer * 0.5:
        return 0.0

    # Check if one contains a significant portion of the other
    # Take first 200 chars of shorter text and check if it's in longer
    sample = t1[:200] if len(t1) <= len(t2) else t2[:200]
    if sample in t1 and sample in t2:
        return 0.8

    # Rough word overlap
    words1 = set(t1.split())
    words2 = set(t2.split())
    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    total = len(words1 | words2)
    return overlap / total if total > 0 else 0.0


def extract_sources_from_output(
    output: str,
    call_id: str,
    source_registry: Optional[Dict[str, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract source/citation information from tool output.

    Parses the indexed format from search results:
        [0] Title: BERT explained\nURL: https://example.com/bert\nSnippet: ...
        [1] Title: Another result\nURL: https://example.com/other\nSnippet: ...

    Registers sources with IDs like "{call_id}-0", "{call_id}-1" so they match
    the citation IDs generated by the agent (<cite id="abc123-0">).

    Args:
        output: The tool output text
        call_id: The tool call ID (e.g., "9b6f69ac")
        source_registry: Optional dict to register sources for citation lookup

    Returns:
        List of source dicts with id, title, url
    """
    sources = []
    seen_urls = set()

    # Primary format: [N] Title: ...\nURL: https://...\n
    # This is the format used by search tools (google, serper, exa, etc.)
    indexed_pattern = re.compile(
        r"\[(\d+)\]\s*Title:\s*([^\n]+)\n\s*URL:\s*(https?://[^\s\n]+)",
        re.MULTILINE,
    )

    for match in indexed_pattern.finditer(output):
        index = match.group(1)
        title = fix_html_entities(match.group(2).strip())
        url = fix_html_entities(match.group(3).strip().rstrip(".,;:"))

        # Create source ID matching agent citation format: {call_id}-{index}
        source_id = f"{call_id}-{index}"

        if url not in seen_urls:
            source = {
                "id": source_id,
                "title": title,
                "url": url,
            }
            sources.append(source)
            seen_urls.add(url)

            # Register in source_registry for citation lookup
            if source_registry is not None:
                source_registry[source_id] = source

    # Fallback: markdown links [title](url) - for browse results, etc.
    if not sources:
        md_links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", output)
        for idx, (title, url) in enumerate(md_links):
            if url.startswith("http") and url not in seen_urls:
                source_id = f"{call_id}-{idx}"
                source = {
                    "id": source_id,
                    "title": fix_html_entities(title),
                    "url": fix_html_entities(url),
                }
                sources.append(source)
                seen_urls.add(url)
                if source_registry is not None:
                    source_registry[source_id] = source

    # Fallback: bare URLs
    if not sources:
        bare_urls = re.findall(r'https?://[^\s<>"\')\]]+', output)
        for idx, url in enumerate(bare_urls):
            clean_url = fix_html_entities(url.rstrip(".,;:"))
            if clean_url not in seen_urls:
                try:
                    domain = urlparse(clean_url).netloc
                    source_id = f"{call_id}-{idx}"
                    source = {
                        "id": source_id,
                        "title": domain,
                        "url": clean_url,
                    }
                    sources.append(source)
                    seen_urls.add(clean_url)
                    if source_registry is not None:
                        source_registry[source_id] = source
                except Exception:
                    pass

    return sources[:10]  # Limit to 10 sources per tool output


def convert_citations_to_markdown(text: str, source_registry: Dict[str, Dict]) -> str:
    """
    Convert citations to markdown links.

    Handles multiple citation formats the model might generate:
    1. XML: <cite id="X">claim</cite> → [claim](url)
    2. Parenthetical: (cite id="X") → [[N]](url)
    3. Inline reference: cite id="X" → [[N]](url)

    Uses the source_registry to look up URLs for citation IDs.
    Falls back to footnote-style if URL not found.
    """

    def get_short_id(cite_id: str) -> str:
        """Extract short index from citation ID (e.g., 'ca47bae5-1' → '1')."""
        if "-" in cite_id:
            return cite_id.split("-")[-1]
        return cite_id[-4:]

    def is_id_like(text: str) -> bool:
        """Check if text looks like a citation ID rather than actual content."""
        if not text:
            return True
        text = text.strip()
        # Looks like an ID: hex-number pattern or just the ID itself
        if re.match(r"^[a-f0-9]{6,}-\d+$", text):
            return True
        # Just a number
        if re.match(r"^\d+$", text):
            return True
        # Very short text (less than 3 chars) that's not a word
        if len(text) < 3:
            return True
        return False

    def lookup_and_format(cite_id: str, claim_text: Optional[str] = None) -> str:
        """Look up citation and format as markdown link."""
        source = source_registry.get(cite_id)
        short_id = get_short_id(cite_id)

        # If claim_text looks like an ID, treat it as no claim text
        if claim_text and is_id_like(claim_text):
            claim_text = None

        if source and source.get("url"):
            url = source["url"]
            if claim_text:
                # Full link with claim text
                return f"[{claim_text}]({url})"
            else:
                # Short superscript-style link like [1]
                return f"[[{short_id}]]({url})"
        else:
            # Fallback: bracketed reference without URL
            if claim_text:
                return f"{claim_text} [{short_id}]"
            else:
                return f"[{short_id}]"

    # Pattern 1: XML cite tags <cite id="X">claim</cite>
    def replace_xml_cite(match):
        cite_id = match.group(1)
        claim_text = match.group(2).strip()
        return lookup_and_format(cite_id, claim_text)

    result = re.sub(
        r'<cite\s+id="([^"]+)">(.*?)</cite>',
        replace_xml_cite,
        text,
        flags=re.DOTALL,
    )
    # Also handle single quotes
    result = re.sub(
        r"<cite\s+id='([^']+)'>(.*?)</cite>",
        replace_xml_cite,
        result,
        flags=re.DOTALL,
    )

    # Pattern 2: Parenthetical citations (cite id="X") or (cite id="X", cite id="Y")
    def replace_paren_cite(match):
        cite_ids = re.findall(r'cite id="([^"]+)"', match.group(0))
        if not cite_ids:
            return match.group(0)
        links = [lookup_and_format(cid) for cid in cite_ids]
        return " ".join(links)

    result = re.sub(
        r'\(cite id="[^"]+"\s*(?:,\s*cite id="[^"]+"\s*)*\)',
        replace_paren_cite,
        result,
    )

    # Pattern 3: Standalone cite id="X" (without parentheses)
    def replace_inline_cite(match):
        cite_id = match.group(1)
        return lookup_and_format(cite_id)

    result = re.sub(
        r'\bcite id="([^"]+)"',
        replace_inline_cite,
        result,
    )

    # Pattern 4: cited in <snippet id="X"> format (some prompts use this)
    def replace_snippet_cite(match):
        snippet_id = match.group(1)
        return lookup_and_format(snippet_id)

    result = re.sub(
        r'<snippet id="([^"]+)">',
        replace_snippet_cite,
        result,
    )
    # Remove closing snippet tags
    result = re.sub(r"</snippet>", "", result)

    # Clean up sequences of multiple citation links - consolidate them
    # e.g., [[1]](url) [[2]](url) [[3]](url) → [[1]](url) [[2]](url) [[3]](url)
    # No change needed, but remove excessive commas/ands between them
    result = re.sub(r"\]\s*,\s*\[", "] [", result)
    result = re.sub(r"\]\s+and\s+\[", "] [", result)

    return result


def strip_remaining_cite_tags(text: str) -> str:
    """Remove any remaining citation patterns that weren't converted."""
    # Remove XML cite tags
    text = re.sub(r"<cite[^>]*>", "", text)
    text = re.sub(r"</cite>", "", text)
    # Remove parenthetical citations (cite id="...")
    text = re.sub(r'\(cite id="[^"]+"\s*(?:,\s*cite id="[^"]+"\s*)*\)', "", text)
    # Remove standalone cite id="..." references
    text = re.sub(r'\bcite id="[^"]+"', "", text)
    # Remove snippet tags
    text = re.sub(r"<snippet[^>]*>", "", text)
    text = re.sub(r"</snippet>", "", text)
    # Clean up extra whitespace from removals
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def truncate_for_display(text: str, max_lines: int = 3, max_chars: int = 300) -> str:
    """
    Truncate text for UI display - show first few lines.

    Used for tool output preview in thinking section.
    """
    lines = text.split("\n")
    if len(lines) > max_lines:
        preview = "\n".join(lines[:max_lines])
        return preview[:max_chars] + f"... [{len(lines) - max_lines} more lines]"
    elif len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


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
    """Create an SSE chunk for tool output (role: tool).

    Uses proper OpenAI chunk envelope format so Open WebUI can render
    tool outputs as collapsible cards linked to tool_call_id.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Tool output goes in delta, wrapped in proper OpenAI envelope
    delta: Dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }

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
                "finish_reason": None,
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
        - Tool calls use delta.tool_calls format (shows as tool cards)
        - Tool outputs use role: "tool" format (with source extraction)
        - Sources are extracted and included in delta.sources
        - Answer is clean content with HTML entities fixed
    """
    # Get config for this model
    model_cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["dr-tulu-deep"])
    config_path = str(AGENT_ROOT / "workflows" / model_cfg["config"])
    dataset_name = model_cfg.get("dataset", "long_form")

    # Use shell wrapper if available (provides cleaner subprocess isolation)
    # The wrapper script sources its own venv and handles environment properly
    if CLI_WRAPPER.exists():
        cmd = [
            "/bin/bash",
            str(CLI_WRAPPER),
            "--config",
            config_path,
            "--query",
            query,
            "--dataset-name",
            dataset_name,
        ]
        use_shell_wrapper = True
    else:
        # Fallback to direct Python invocation
        python_path = str(CLI_PYTHON) if CLI_PYTHON.exists() else sys.executable
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
        use_shell_wrapper = False

    # Set environment - minimal for shell wrapper, more explicit for direct Python
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Only set SSL certs explicitly if not using shell wrapper
    # (wrapper activates its own venv which has proper SSL config)
    if not use_shell_wrapper:
        env["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
        env["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

    # Disable proxy if set (can cause issues with Modal)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    # Start subprocess using Popen (more reliable than asyncio subprocess with uvicorn)
    import subprocess

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(AGENT_ROOT),
            bufsize=1,  # Line buffered
            text=False,  # Binary mode for consistent handling
            start_new_session=True,  # Create new process group to avoid signal/network inheritance issues
        )
    except Exception as e:
        yield create_sse_chunk(f"Error starting CLI: {e}", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Helper to read line in thread (non-blocking for asyncio)
    def read_line():
        if process.stdout:
            return process.stdout.readline()
        return b""

    # Track state
    current_answer = ""
    tool_call_counter = 0
    thinking_started = False
    thinking_closed = False
    all_sources: List[Dict] = []
    pending_thinking = ""  # Accumulate thinking to clean before sending

    # Tool call ID mapping for tool outputs
    tool_call_ids: Dict[str, str] = {}

    # Source registry: maps citation IDs to source info (url, title)
    # Built from tool outputs, used to convert <cite> tags to markdown links
    source_registry: Dict[str, Dict] = {}

    # Deduplication state: accumulate thinking content to detect report bleeding
    accumulated_thinking = ""  # All thinking content seen so far
    report_detected_in_thinking = False  # Flag when report content detected in thinking

    # Deduplication state: accumulate thinking content to detect report bleeding
    accumulated_thinking = ""  # All thinking content seen so far
    report_detected_in_thinking = False  # Flag when report content detected in thinking

    try:
        # Read stdout line by line using thread for non-blocking I/O
        assert process.stdout is not None, "stdout is None"
        assert process.stderr is not None, "stderr is None"

        while True:
            # Read line in thread to avoid blocking the event loop
            line = await asyncio.to_thread(read_line)
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
                        # Accumulate for deduplication check
                        accumulated_thinking += cleaned + "\n"

                        # Check if this thinking content looks like a report
                        # If so, suppress it to avoid duplication with the answer
                        if not report_detected_in_thinking:
                            if is_report_like_content(accumulated_thinking):
                                report_detected_in_thinking = True
                                # Don't stream this report-like content
                                # Add a note instead
                                if thinking_started:
                                    yield create_sse_chunk(
                                        "\n*[Synthesizing findings...]*\n"
                                    )
                                continue

                        # If we already detected report content, skip further thinking
                        if report_detected_in_thinking:
                            continue

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
                    # Extract sources from tool output and register them
                    # Pass call_id so sources are registered as {call_id}-{index}
                    sources = extract_sources_from_output(
                        output, call_id, source_registry
                    )
                    if sources:
                        all_sources.extend(sources)

                    # Create truncated preview for UI display (first 3 lines)
                    display_content = truncate_for_display(
                        output, max_lines=3, max_chars=500
                    )

                    # Send tool output message with sources
                    yield create_tool_output_chunk(
                        tool_call_id=call_id,
                        content=display_content,
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

                # Convert <cite id="X">text</cite> to markdown links
                answer_chunk = convert_citations_to_markdown(
                    answer_chunk, source_registry
                )

                # Strip any remaining unconverted cite tags
                answer_chunk = strip_remaining_cite_tags(answer_chunk)

                if answer_chunk:
                    current_answer += answer_chunk

                    # Send answer with accumulated sources on first chunk
                    if len(current_answer) == len(answer_chunk) and all_sources:
                        yield create_sse_chunk(answer_chunk, sources=all_sources)
                    else:
                        yield create_sse_chunk(answer_chunk)

            elif line_text.startswith("[DONE]"):
                # Completion - but don't emit finish yet, wait for process
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

        # Wait for process to complete before emitting finish
        await asyncio.to_thread(process.wait)

        # Drain any remaining stderr
        if process.stderr:
            try:
                stderr_bytes = await asyncio.to_thread(process.stderr.read)
                stderr_text = (
                    stderr_bytes.decode("utf-8", errors="replace")
                    if stderr_bytes
                    else ""
                )
                if stderr_text:
                    print(f"[CLI STDERR] {stderr_text}", file=sys.stderr)
                    # Only show error if we didn't get an answer
                    if not current_answer:
                        yield create_sse_chunk(
                            f"\n\n**CLI Error:** {stderr_text[:500]}\n"
                        )
            except Exception as e:
                print(f"[CLI] Stderr drain error: {e}", file=sys.stderr)

        # Close thinking block if still open (no answer received)
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
            thinking_closed = True

    except asyncio.CancelledError:
        # Client disconnected - kill subprocess and emit cancelled finish
        print("[CLI Gateway] Client disconnected, killing subprocess", file=sys.stderr)
        try:
            process.kill()
            process.wait()  # Synchronous wait for Popen
        except Exception:
            pass
        # Emit finish with cancelled reason
        yield create_sse_chunk("", finish_reason="cancelled")
        yield "data: [DONE]\n\n"
        return
    except Exception as e:
        # Close thinking if open
        if thinking_started and not thinking_closed:
            yield create_sse_chunk("</think>\n\n")
        yield create_sse_chunk(f"\n\n**Error:** {str(e)}\n")
        # Still emit proper finish
        yield create_sse_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    # Final chunk with finish_reason - only after process is fully done
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

"""
title: DR-Tulu Deep Research Pipeline
author: DR-Tulu Team
date: 2024-12-04
version: 1.0.0
license: MIT
description: A pipeline for deep research using DR-Tulu agent with real-time status updates and native Open WebUI citations.
requirements: pydantic, aiohttp
environment_variables: GOOGLE_AI_API_KEY, SERPER_API_KEY
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from pydantic import BaseModel, Field


class Pipeline:
    """
    DR-Tulu Deep Research Pipeline for Open WebUI.

    Features:
    - Real-time status updates during tool execution (thinking panel)
    - Native citation support with [1], [2] format
    - Source chips with favicons
    - Streaming response with proper Open WebUI integration
    """

    class Valves(BaseModel):
        """Configuration valves for the pipeline."""

        # API Keys
        GOOGLE_AI_API_KEY: str = Field(
            default="", description="Google AI API Key for Gemini"
        )
        SERPER_API_KEY: str = Field(
            default="", description="Serper API Key for web search"
        )

        # Model settings
        MODEL_NAME: str = Field(
            default="gemini/gemini-2.5-flash", description="Model to use for research"
        )
        MAX_TOOL_CALLS: int = Field(
            default=10, description="Maximum number of tool calls per request"
        )
        MAX_TOKENS: int = Field(default=8000, description="Maximum tokens for response")
        TEMPERATURE: float = Field(
            default=1.0, description="Temperature for generation"
        )

        # Feature flags
        USE_BROWSE_AGENT: bool = Field(
            default=True, description="Enable browse agent for fetching web pages"
        )
        USE_ARABIC_LIBRARY: bool = Field(
            default=False, description="Enable Arabic books search"
        )

        # Paths
        DR_TULU_PATH: str = Field(
            default="",
            description="Path to DR-Tulu agent directory (auto-detected if empty)",
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "dr-tulu-deep-research"
        self.name = "DR-Tulu Deep Research"

        self.valves = self.Valves(
            GOOGLE_AI_API_KEY=os.getenv("GOOGLE_AI_API_KEY", ""),
            SERPER_API_KEY=os.getenv("SERPER_API_KEY", ""),
            DR_TULU_PATH=os.getenv("DR_TULU_PATH", ""),
        )

        self._workflow = None
        self._initialized = False

    async def on_startup(self):
        """Called when the server starts."""
        print(f"on_startup: {__name__}")
        await self._initialize_workflow()

    async def on_shutdown(self):
        """Called when the server stops."""
        print(f"on_shutdown: {__name__}")

    async def on_valves_updated(self):
        """Called when valves are updated."""
        print(f"on_valves_updated: {__name__}")
        self._initialized = False
        await self._initialize_workflow()

    async def _initialize_workflow(self):
        """Initialize the DR-Tulu workflow."""
        if self._initialized:
            return

        try:
            # Find DR-Tulu path
            dr_tulu_path = self.valves.DR_TULU_PATH
            if not dr_tulu_path:
                # Try common locations
                possible_paths = [
                    Path(__file__).parent / "DR-Tulu" / "agent",
                    Path.home() / "projects" / "parallax_project" / "DR-Tulu" / "agent",
                    Path("/home/elwalid/projects/parallax_project/DR-Tulu/agent"),
                ]
                for p in possible_paths:
                    if p.exists():
                        dr_tulu_path = str(p)
                        break

            if not dr_tulu_path or not Path(dr_tulu_path).exists():
                print(f"Warning: DR-Tulu path not found: {dr_tulu_path}")
                return

            # Add to path
            if dr_tulu_path not in sys.path:
                sys.path.insert(0, dr_tulu_path)

            # Set environment variables
            if self.valves.GOOGLE_AI_API_KEY:
                os.environ["GOOGLE_AI_API_KEY"] = self.valves.GOOGLE_AI_API_KEY
                os.environ["GEMINI_API_KEY"] = self.valves.GOOGLE_AI_API_KEY

            if self.valves.SERPER_API_KEY:
                os.environ["SERPER_API_KEY"] = self.valves.SERPER_API_KEY

            # Import and create workflow
            from workflows.auto_search_sft import AutoReasonSearchWorkflow

            config_path = Path(dr_tulu_path) / "workflows" / "auto_search_gemini.yaml"
            self._workflow = AutoReasonSearchWorkflow(configuration=str(config_path))
            self._initialized = True
            print(f"DR-Tulu workflow initialized from {config_path}")

        except Exception as e:
            print(f"Error initializing DR-Tulu workflow: {e}")
            import traceback

            traceback.print_exc()

    def _extract_sources(self, raw_text: str) -> Dict[str, Dict[str, Any]]:
        """Extract sources from raw workflow output."""
        sources = {}

        # Pattern: <snippet id="...">Title: ...\nURL: ...\nSnippet: ...</snippet>
        snippet_pattern = r'<snippet\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</snippet>'
        for match in re.finditer(snippet_pattern, raw_text, re.DOTALL | re.IGNORECASE):
            snippet_id = match.group(1)
            content = match.group(2)

            title_match = re.search(r"Title:\s*(.+?)(?:\n|$)", content)
            url_match = re.search(r"URL:\s*(https?://[^\s\n]+)", content)
            snippet_match = re.search(r"Snippet:\s*(.+?)(?:\n|$)", content, re.DOTALL)

            title = (
                title_match.group(1).strip() if title_match else f"Source {snippet_id}"
            )
            url = url_match.group(1).strip() if url_match else None
            snippet_text = snippet_match.group(1).strip()[:500] if snippet_match else ""

            sources[snippet_id] = {"title": title, "url": url, "snippet": snippet_text}

        # Pattern: <webpage id="...">...</webpage>
        webpage_pattern = r'<webpage\s+id=["\']?([^"\'>\s]+)["\']?[^>]*>(.*?)</webpage>'
        for match in re.finditer(webpage_pattern, raw_text, re.DOTALL | re.IGNORECASE):
            webpage_id = match.group(1)
            content = match.group(2)

            url_match = re.search(r"(https?://[^\s\n<]+)", content)
            url = url_match.group(1).strip() if url_match else None

            title = content[:80].strip().replace("\n", " ")
            if len(content) > 80:
                title += "..."

            sources[webpage_id] = {"title": title, "url": url, "snippet": content[:500]}

        return sources

    def _process_response(self, raw_text: str, source_map: Dict[str, Dict]) -> tuple:
        """
        Process raw workflow output into Open WebUI format.

        Returns:
            tuple: (content, sources_array, cited_urls)
        """
        # Strip patterns for cleaning
        strip_patterns = [
            (r"<think>.*?</think>", ""),
            (r"<call_tool[^>]*>.*?(?:</call_tool>|$)", ""),
            (r"<tool_call>.*?</tool_call>", ""),
            (r"<tool_response>.*?</tool_response>", ""),
            (r"<tool_output>.*?</tool_output>", ""),
            (r"<snippet[^>]*>.*?</snippet>", ""),
            (r"</?tool_call>", ""),
            (r"</?tool_response>", ""),
            (r"</?tool_output>", ""),
            (r"<answer>", ""),
            (r"</answer>", ""),
            (r"<webpage[^>]*>.*?</webpage>", ""),
            (r"<raw_trace>.*?</raw_trace>", ""),
            (r"<search_results>.*?</search_results>", ""),
            (r"<browse_results>.*?</browse_results>", ""),
            (r"Title:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
            (r"URL:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
            (r"Snippet:.*?(?=Title:|URL:|Snippet:|<|$)", ""),
            (r"No content available\n?", ""),
        ]

        # Extract answer content
        answer_match = re.search(
            r"<answer>(.*?)</answer>", raw_text, re.DOTALL | re.IGNORECASE
        )
        if answer_match:
            cleaned = answer_match.group(1)
        else:
            cleaned = re.sub(
                r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE
            )
            if not cleaned.strip():
                think_matches = re.findall(
                    r"<think>(.*?)</think>", raw_text, re.DOTALL | re.IGNORECASE
                )
                if think_matches:
                    cleaned = think_matches[-1]

        # Apply strip patterns
        for pat, repl in strip_patterns:
            cleaned = re.sub(pat, repl, cleaned, flags=re.DOTALL | re.IGNORECASE)

        # Find citations in order
        citation_pattern = r'<cite\s+ids?=["\']?([^"\'>\s]+)["\']?[^>]*>([^<]*)</cite>'
        cited_ids_ordered = []

        for match in re.finditer(citation_pattern, cleaned, re.IGNORECASE):
            ids_str = match.group(1)
            for cid in ids_str.split(","):
                cid = cid.strip()
                if cid and cid not in cited_ids_ordered:
                    cited_ids_ordered.append(cid)

        # Create ID to number mapping (1-indexed)
        id_to_number = {cid: idx + 1 for idx, cid in enumerate(cited_ids_ordered)}

        # Build Open WebUI sources array
        openwebui_sources = []
        cited_urls = []

        for cid in cited_ids_ordered:
            source_info = source_map.get(cid, {})
            url = source_info.get("url", "")
            title = source_info.get("title", f"Source {cid}")
            snippet = source_info.get("snippet", "")

            if url:
                cited_urls.append(url)

            openwebui_sources.append(
                {
                    "source": {"name": url if url else title, "url": url or ""},
                    "document": [snippet] if snippet else [title],
                    "metadata": [{"source": url}] if url else [{"source": title}],
                }
            )

        # Convert citations to [1], [2,3] format
        def replace_citation(match):
            ids_str = match.group(1)
            cited_text = match.group(2)

            numbers = []
            for cid in ids_str.split(","):
                cid = cid.strip()
                num = id_to_number.get(cid)
                if num:
                    numbers.append(str(num))

            if numbers:
                return f"{cited_text}[{','.join(numbers)}]"
            return cited_text

        cleaned = re.sub(
            citation_pattern, replace_citation, cleaned, flags=re.IGNORECASE
        )

        # Remove remaining XML tags
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        # Clean up artifacts
        cleaned = re.sub(r"\bundefined\b\s*\+?\s*\d*", "", cleaned)
        cleaned = re.sub(r"\s*\+\s*\d+\s*(?=[.,\s]|$)", "", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"\.\s*\.", ".", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"  +", " ", cleaned)

        return cleaned.strip(), openwebui_sources, cited_urls

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline entry point.

        Args:
            body: Request body with messages, model, etc.
            __user__: User information
            __event_emitter__: Function to emit events to Open WebUI
            __event_call__: Function to call events and wait for response

        Returns:
            Generated response with citations
        """
        # Ensure workflow is initialized
        if not self._initialized:
            await self._initialize_workflow()

        if not self._workflow:
            error_msg = (
                "DR-Tulu workflow not initialized. Please check the configuration."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_msg,
                            "done": True,
                            "error": True,
                        },
                    }
                )
            return error_msg

        messages = body.get("messages", [])
        if not messages:
            return "No messages provided."

        # Extract user query
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_message = item.get("text", "")
                            break
                else:
                    user_message = str(content)
                break

        if not user_message:
            return "No user message found."

        # Emit initial status
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "action": "web_search",
                        "description": "Starting deep research...",
                        "done": False,
                    },
                }
            )

        # Track tool calls for status updates
        tool_call_count = 0
        searched_urls = []
        current_tool = None

        async def step_callback(text: str, tool_outputs: list):
            """Callback for each step of the workflow."""
            nonlocal tool_call_count, searched_urls, current_tool

            if tool_outputs:
                for output in tool_outputs:
                    tool_call_count += 1
                    tool_name = getattr(output, "tool_name", "search")

                    # Extract URLs from tool output if available
                    if hasattr(output, "documents"):
                        for doc in output.documents:
                            if hasattr(doc, "url") and doc.url:
                                searched_urls.append(doc.url)

                    # Emit status update
                    if __event_emitter__:
                        # Friendly tool names
                        tool_display = {
                            "google_search": "Searching the web",
                            "snippet_search": "Searching academic papers",
                            "browse_webpage": "Reading web page",
                            "exa_search": "Searching with Exa",
                            "search_arabic_books": "Searching Arabic books",
                        }.get(tool_name, f"Using {tool_name}")

                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "action": "web_search",
                                    "description": tool_display,
                                    "done": False,
                                },
                            }
                        )

            elif text:
                # Model is generating text (thinking)
                if __event_emitter__ and "<think>" in text.lower():
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "action": "web_search",
                                "description": "Analyzing information...",
                                "done": False,
                            },
                        }
                    )

        try:
            # Run the workflow with step callback
            result = await self._workflow(
                problem=user_message,
                dataset_name="long_form",
                verbose=False,
                step_callback=step_callback,
            )

            raw_text = result.get("generated_text", "")

            # Extract sources from raw output
            source_map = self._extract_sources(raw_text)

            # Process response
            content, sources, cited_urls = self._process_response(raw_text, source_map)

            # Emit final status with sources
            if __event_emitter__:
                # Status: search complete
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "action": "web_search",
                            "description": f"Searched {len(cited_urls)} sources",
                            "urls": cited_urls[:10],
                            "done": True,
                        },
                    }
                )

                # Emit sources for citations
                if sources:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "action": "sources_retrieved",
                                "count": len(sources),
                                "done": True,
                            },
                        }
                    )

                    # Emit each source for citation chips
                    for source in sources:
                        await __event_emitter__(
                            {
                                "type": "source",
                                "data": source,
                            }
                        )

            return content

        except asyncio.TimeoutError:
            error_msg = "Research took too long. Please try a simpler query."
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_msg,
                            "done": True,
                            "error": True,
                        },
                    }
                )
            return error_msg

        except Exception as e:
            error_msg = f"Research error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_msg,
                            "done": True,
                            "error": True,
                        },
                    }
                )
            import traceback

            traceback.print_exc()
            return error_msg

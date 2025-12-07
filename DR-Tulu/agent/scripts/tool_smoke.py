#!/usr/bin/env python3
"""
Quick MCP tool smoke test.

Runs a few representative tool calls via FastMCPTransport (in-process) and prints
compact results. Intended to sanity-check API keys/connectivity before UI work.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport


def _env_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"


def load_env() -> None:
    env_path = _env_path()
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded env from {env_path}")
    else:
        print(f"⚠ No .env found at {env_path}; relying on process env.")


async def call_tool(client: Client, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = await client.call_tool(name, params)
        content = getattr(result, "content", None) or []
        if content and hasattr(content[0], "text"):
            return {"name": name, "ok": True, "data": content[0].text[:400]}
        if content and isinstance(content[0], dict):
            return {"name": name, "ok": True, "data": str(content[0])[:400]}
        return {"name": name, "ok": True, "data": str(content)[:400]}
    except Exception as exc:  # pragma: no cover - defensive surface
        return {"name": name, "ok": False, "error": str(exc)}


def pick_tool(preferred: List[str], available: List[str]) -> Optional[str]:
    for name in preferred:
        if name in available:
            return name
    return None


async def main() -> None:
    load_env()
    from dr_agent.mcp_backend import main as mcp_module

    transport = FastMCPTransport(mcp_module.mcp)
    client = Client(transport, timeout=30)

    async with client:
        tool_list = await client.list_tools()
        available = [getattr(t, "name", None) or t.get("name") for t in (tool_list or []) if t]

        tests: List[tuple[str, Dict[str, Any]]] = []
        if (name := pick_tool(["serper_google_search", "serper_google_scholar_search"], available)):
            tests.append((name, {"query": "current weather in Seattle", "num_results": 3}))
        if (name := pick_tool(["semantic_scholar_snippet_search"], available)):
            tests.append((name, {"query": "transformers retrieval 2024", "limit": 3}))
        if (name := pick_tool(["exa_search"], available)):
            tests.append((name, {"query": "latest AI papers on arxiv 2024", "num_results": 3, "include_domains": "arxiv.org"}))
        if (name := pick_tool(["arabic_books_search"], available)):
            tests.append((name, {"query": "الذكاء الاصطناعي", "n_results": 2}))

        print(f"Tools available: {available}")
        for name, params in tests:
            result = await call_tool(client, name, params)
            if result.get("ok"):
                print(f"✓ {name}: {result.get('data')}")
            else:
                print(f"✗ {name}: {result.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())

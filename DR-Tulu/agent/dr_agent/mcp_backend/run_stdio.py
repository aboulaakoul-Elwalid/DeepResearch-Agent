"""
Minimal entrypoint to launch the MCP server over stdio.

This avoids HTTP port binding (disallowed in some sandboxes) and is used by
FastMCP clients spawned via PythonStdioTransport.
"""

import os

from dr_agent.mcp_backend.main import mcp
from dr_agent.mcp_backend.cache import set_cache_enabled


def main():
    # Respect optional cache toggle via env for flexibility
    if os.getenv("MCP_DISABLE_CACHE") == "1":
        set_cache_enabled(False)
    else:
        set_cache_enabled(True)

    # Run FastMCP server over stdio transport
    # Disable banner because it corrupts stdio transport streams
    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()

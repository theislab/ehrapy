"""MCP server layer for ehrapy.

Install the optional extra with ``pip install ehrapy[mcp]``.
Start the server with ``ehrapy-mcp`` or ``python -m ehrapy.mcp``.
"""

from __future__ import annotations

from ehrapy.mcp.catalog import catalog_summary, list_functions, list_namespaces


def main() -> None:
    """Run the ehrapy FastMCP server."""
    from ehrapy.mcp.server import main as run_server

    run_server()


__all__ = ["catalog_summary", "list_functions", "list_namespaces", "main"]

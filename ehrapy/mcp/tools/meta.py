"""Meta, runtime, and workflow guide tools for ehrapy MCP."""

from __future__ import annotations

import sys
from typing import Any

from fastmcp import Context  # noqa: TC002
from fastmcp.tools.tool import ToolResult

import ehrapy as ep
from ehrapy.mcp.policy import get_allowed_roots, is_read_only_mode
from ehrapy.mcp.prompts import WORKFLOW_PROMPT
from ehrapy.mcp.registry import registry
from ehrapy.mcp.session import get_session


def get_workflow_guide() -> ToolResult:
    """Return recommended multi-step clinical analysis workflows covering quality control, survival analysis, and causal inference.

    Use this to determine the standard sequence of ehrapy operations before processing a cohort.
    """
    struct = {"status": "ok", "guide": "Standard clinical workflow guide"}
    return ToolResult(structured_content=struct, content=WORKFLOW_PROMPT)


def get_runtime_context(ctx: Context = None) -> ToolResult:
    """Return MCP runtime environment details including ehrapy version, cache directories, and session handle.

    Use this to verify environment state, discover the cache directory for Cache-Bridge ingestion, and identify active dataset identifiers.
    """
    session = get_session(ctx)
    allowed = get_allowed_roots()
    datasets = registry.list_datasets()
    cache_dir = registry.cache_dir()
    demo_dir = registry.demo_data_dir()
    cache_writable = registry.is_cache_writable()

    struct: dict[str, Any] = {
        "status": "ok",
        "ehrapy_version": ep.__version__,
        "python_version": sys.version.split()[0],
        "cache_dir": str(cache_dir),
        "demo_data_dir": str(demo_dir),
        "plots_dir": str(registry.plots_dir()),
        "active_edata_id": session.get_latest_edata_id(),
        "read_only_mode": is_read_only_mode(),
        "allowed_roots": [str(r) for r in allowed] if allowed else None,
        "cached_datasets_count": len(datasets),
        "cache_writable": cache_writable,
    }

    md_lines = [
        "### ehrapy MCP Runtime Context",
        f"- **ehrapy version:** `{ep.__version__}`",
        f"- **Python version:** `{sys.version.split()[0]}`",
        f"- **Active dataset (edata_id):** `{session.get_latest_edata_id() or 'None'}`",
        f"- **Cached datasets:** {len(datasets)}",
        f"- **Cache directory:** `{cache_dir}`",
        f"- **Demo data directory:** `{demo_dir}`",
        f"- **Plots directory:** `{registry.plots_dir()}`",
        f"- **Read-only mode:** `{is_read_only_mode()}`",
        f"- **Allowed roots:** `{', '.join(str(r) for r in allowed) if allowed else 'Unrestricted'}`",
        f"- **Cache writable:** `{cache_writable}`",
        "",
        "**Cache-Bridge Ingestion:**",
        f"To ingest files from a sandboxed client, copy them to `{cache_dir}` and pass the resulting path to `ingest_dataset(file_path=...)`.",
    ]

    return ToolResult(structured_content=struct, content="\n".join(md_lines))

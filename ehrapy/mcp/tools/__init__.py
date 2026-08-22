"""Tool list and registration for ehrapy MCP."""

from __future__ import annotations

from ehrapy.mcp.tools.dispatch_tools import (
    fork_edata_handle,
    get_edata_snapshot,
    get_function_help,
    list_ehrapy_functions,
    load_demo_dataset,
    run_analysis,
    run_get,
    run_io,
    run_plot,
    run_preprocessing,
)
from ehrapy.mcp.tools.ingestion import export_edata, ingest_dataset
from ehrapy.mcp.tools.meta import get_runtime_context, get_workflow_guide

ALL_TOOLS_LIST = [
    # Reference & Discovery
    get_workflow_guide,
    get_runtime_context,
    list_ehrapy_functions,
    get_function_help,
    # Cohort Management
    ingest_dataset,
    load_demo_dataset,
    fork_edata_handle,
    export_edata,
    get_edata_snapshot,
    # Execution
    run_preprocessing,
    run_analysis,
    run_get,
    run_plot,
    run_io,
]

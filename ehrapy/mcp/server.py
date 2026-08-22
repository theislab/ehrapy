"""FastMCP server definition, tool registration, middleware, and prompts for ehrapy."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import mcp.types as mt
from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext

from ehrapy.mcp.errors import unknown_argument_error
from ehrapy.mcp.prompts import SERVER_INSTRUCTIONS
from ehrapy.mcp.registry import registry
from ehrapy.mcp.tools import ALL_TOOLS_LIST

if TYPE_CHECKING:
    from fastmcp.tools.tool import ToolResult

    pass

_TOOL_ANNOTATIONS: dict[str, dict[str, Any]] = {
    "load_demo_dataset": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    "ingest_dataset": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    "fork_edata_handle": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    "export_edata": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    "get_edata_snapshot": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "get_workflow_guide": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "get_runtime_context": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "list_ehrapy_functions": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "get_function_help": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "run_preprocessing": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    "run_analysis": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    "run_get": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "run_plot": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    "run_io": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
}


# Client-side orchestration keys that are not part of any tool schema and are silently dropped.
_ORCHESTRATION_KEYS = frozenset({"wait_for_previous"})


def _append_note(result: ToolResult, note: str, folded: list[str]) -> ToolResult:
    """Append a steering note to a tool result on both channels."""
    from fastmcp.tools.tool import ToolResult as _ToolResult

    try:
        blocks = list(result.content or [])
        blocks.append(mt.TextContent(type="text", text=f"\n\n_{note}_"))
        struct = result.structured_content
        if isinstance(struct, dict):
            struct = {**struct, "folded_arguments": folded}
        return _ToolResult(content=blocks, structured_content=struct, is_error=bool(getattr(result, "is_error", False)))
    except Exception:  # noqa: BLE001
        # Never let the advisory note break an otherwise successful call.
        return result


class ArgumentValidationMiddleware(Middleware):
    """Reconcile client-supplied arguments with the tool schema before execution.

    Agents routinely pass an ehrapy function's own kwargs at the top level
    (``run_preprocessing(function='qc_metrics', groupby='service_unit')``) instead of
    nesting them under ``params``. For tools that take a ``params`` dict, those keys are
    folded in and the fold is reported back so the agent learns the correct shape
    (origin fix #3). For tools with no ``params`` -- or keys that survive folding -- the
    call is rejected with ``UNKNOWN_ARGUMENT`` rather than silently dropped, so a
    hallucinated argument never passes unnoticed.
    """

    def __init__(self, server: FastMCP) -> None:
        self.server = server

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        """Fold or reject unknown arguments, and strip client orchestration keys."""
        message = context.message
        name = message.name
        arguments = dict(message.arguments or {})

        tool = await self.server.get_tool(name)
        if tool is None:
            return await call_next(context)

        # Orchestration keys are stripped unconditionally -- including for zero-argument
        # tools, whose empty schema would otherwise reject them during validation.
        stripped = _ORCHESTRATION_KEYS & set(arguments)
        for key in stripped:
            del arguments[key]

        has_schema = isinstance(getattr(tool, "parameters", None), dict)
        if not has_schema:
            if stripped:
                context = context.copy(message=mt.CallToolRequestParams(name=name, arguments=arguments))
            return await call_next(context)

        valid_keys = set(tool.parameters.get("properties", {}).keys())

        unknown = set(arguments) - valid_keys
        folded: list[str] = []
        if unknown and "params" in valid_keys:
            existing = arguments.get("params")
            existing = dict(existing) if isinstance(existing, dict) else {}
            # An explicit params entry wins over a folded top-level kwarg of the same name.
            merged = {k: arguments.pop(k) for k in sorted(unknown)}
            folded = sorted(merged)
            arguments["params"] = {**merged, **existing}
            unknown = set()

        if unknown:
            return unknown_argument_error(name, unknown, valid_keys)

        if folded or stripped:
            context = context.copy(message=mt.CallToolRequestParams(name=name, arguments=arguments))

        result = await call_next(context)

        if folded:
            note = (
                f"Folded top-level argument(s) {', '.join(f'`{k}`' for k in folded)} into `params`. "
                f"Pass function keyword arguments inside `params` to avoid relying on this."
            )
            result = _append_note(result, note, folded)
        return result


def create_server() -> FastMCP:
    """Create and configure the FastMCP server instance."""
    # Startup cache purge
    try:
        registry.purge()
    except Exception:  # noqa: BLE001
        pass

    server = FastMCP("ehrapy", instructions=SERVER_INSTRUCTIONS)
    server.add_middleware(ArgumentValidationMiddleware(server))

    # Register tools with annotations
    for tool_fn in ALL_TOOLS_LIST:
        name = tool_fn.__name__
        annotations = _TOOL_ANNOTATIONS.get(name)
        if annotations:
            server.tool(annotations=annotations)(tool_fn)
        else:
            server.tool()(tool_fn)

    # Register MCP Prompts
    @server.prompt("ehrapy-explore")
    def prompt_ehrapy_explore(dataset: str = "mimic_2") -> str:
        """Prompt template for exploratory data analysis of a clinical cohort."""
        return (
            f"Please perform an exploratory analysis on dataset '{dataset}':\n"
            f"1. Load the dataset using load_demo_dataset(dataset='{dataset}') or get_edata_snapshot() if already loaded.\n"
            f"2. Calculate quality control metrics using run_preprocessing(function='qc_metrics').\n"
            f"3. Visualize missingness using run_plot(function='missing_values_matrix').\n"
            f"4. Summarize key cohort characteristics and data quality issues."
        )

    @server.prompt("ehrapy-clustering")
    def prompt_ehrapy_clustering(dataset: str = "mimic_2", resolution: float = 1.0) -> str:
        """Prompt template for unsupervised subtyping and clustering of clinical cohorts."""
        return (
            f"Please perform patient sub-phenotyping and clustering on dataset '{dataset}':\n"
            f"1. Encode categorical features with run_preprocessing(function='encode', params={{'autodetect': True}}).\n"
            f"2. Impute missing values with run_preprocessing(function='knn_impute').\n"
            f"3. Compute PCA with run_preprocessing(function='pca') and neighbor graph with run_preprocessing(function='neighbors').\n"
            f"4. Perform Leiden clustering with run_analysis(function='leiden', params={{'resolution': {resolution}}}).\n"
            f"5. Visualize clusters with run_plot(function='umap', params={{'color': 'leiden'}}).\n"
            f"6. Extract top differentiating features with run_analysis(function='rank_features_groups', params={{'groupby': 'leiden'}}) and run_get(function='rank_features_groups_df')."
        )

    @server.prompt("ehrapy-survival")
    def prompt_ehrapy_survival(
        dataset: str = "mimic_2",
        duration_col: str = "mort_day_censored",
        event_col: str = "censor_flg",
    ) -> str:
        """Prompt template for clinical survival analysis."""
        return (
            f"Please perform survival analysis on dataset '{dataset}':\n"
            f"1. Fit Kaplan-Meier survival curves using run_analysis(function='kaplan_meier', params={{'duration_col': '{duration_col}', 'event_col': '{event_col}'}}).\n"
            f"2. Plot survival curves using run_plot(function='kaplan_meier').\n"
            f"3. Fit a Cox proportional hazards model using run_analysis(function='cox_ph', params={{'duration_col': '{duration_col}', 'event_col': '{event_col}'}}).\n"
            f"4. Render the forest plot of hazard ratios using run_plot(function='cox_ph_forestplot')."
        )

    return server


mcp = create_server()


def main() -> None:
    """Entry point for running the MCP server."""
    parser = argparse.ArgumentParser(description="ehrapy MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "http"], help="MCP transport protocol")
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

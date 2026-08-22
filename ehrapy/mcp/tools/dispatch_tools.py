"""Dispatch, inspection, demo, and help tools for ehrapy MCP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from fastmcp import Context  # noqa: TC002
from fastmcp.tools.tool import ToolResult

from ehrapy.mcp.catalog import (
    function_help,
    function_help_markdown,
    list_functions,
    list_namespaces,
)
from ehrapy.mcp.dispatch import run_dispatch
from ehrapy.mcp.edata_store import fork_edata, load_edata
from ehrapy.mcp.errors import mcp_error, unknown_handle_error
from ehrapy.mcp.session import get_session
from ehrapy.mcp.steering import get_suggested_next


async def load_demo_dataset(dataset: str, ctx: Context = None) -> ToolResult:
    """Load a built-in demonstration cohort into the MCP session.

    Use 'mimic_2' for ICU mortality and survival workflows, or 'physionet2012' for in-hospital mortality benchmarking.
    Returns a dataset handle (edata_id) for subsequent tool calls.
    """
    return await run_dispatch("demo", dataset, ctx=ctx)


def fork_edata_handle(edata_id: str | None = None, name: str | None = None, ctx: Context = None) -> ToolResult:
    """Create an independent copy of an existing EHRData dataset handle.

    Use this to branch analysis pipelines or preserve intermediate states before destructive transformations.
    """
    session = get_session(ctx)
    used_latest = False
    handle = edata_id
    if handle is None:
        handle = session.get_latest_edata_id()
        used_latest = True

    if handle is None:
        return mcp_error(
            "fork_edata_handle",
            "No dataset handle provided and no active dataset in session.",
            error_code="NO_ACTIVE_DATASET",
            agent_action="Load a dataset with load_demo_dataset(dataset='mimic_2') first.",
        )

    try:
        record = fork_edata(handle, name=name)
        session.set_latest_edata_id(record.edata_id)
        struct: dict[str, Any] = {
            "status": "ok",
            "edata_id": record.edata_id,
            "parent_id": record.parent_id,
            "name": record.name,
            "n_obs": record.n_obs,
            "n_vars": record.n_vars,
        }
        if used_latest:
            struct["used_latest"] = True

        md = (
            f"### Forked EHRData handle\n"
            f"- **New handle (edata_id):** `{record.edata_id}`\n"
            f"- **Parent handle:** `{record.parent_id}`\n"
            f"- **Name:** `{record.name}`\n"
            f"- **Observations:** {record.n_obs}\n"
            f"- **Variables:** {record.n_vars}"
        )
        return ToolResult(structured_content=struct, content=md)
    except (KeyError, FileNotFoundError):
        return unknown_handle_error("fork_edata_handle", "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return mcp_error("fork_edata_handle", f"Failed to fork handle '{handle}': {exc}", error_code="FORK_ERROR")


def _non_none_str_list(items: Any) -> list[str]:
    return [str(x) for x in items if x is not None]


def _truncated_list_str(items: list[str], limit: int = 25) -> str:
    suffix = "..." if len(items) > limit else ""
    return f"{', '.join(items[:limit])}{suffix}"


def _snapshot_struct(edata: Any, handle: str, cols: dict[str, list[str]], *, used_latest: bool) -> dict[str, Any]:
    struct = {
        "status": "ok",
        "edata_id": handle,
        "n_obs": edata.n_obs,
        "n_vars": edata.n_vars,
        "obs_columns": cols["obs"],
        "var_columns": cols["var"],
        "layers": cols["layers"],
        "uns_keys": cols["uns"],
    }
    if used_latest:
        struct["used_latest"] = True
    return struct


def _snapshot_markdown(edata: Any, handle: str, cols: dict[str, list[str]]) -> str:
    md_lines = [
        f"### Snapshot for EHRData `{handle}`",
        f"- **Dimensions:** {edata.n_obs} observations × {edata.n_vars} variables",
        f"- **Observation columns ({len(cols['obs'])}):** {_truncated_list_str(cols['obs'])}",
        f"- **Variable columns ({len(cols['var'])}):** {_truncated_list_str(cols['var'])}",
        f"- **Layers ({len(cols['layers'])}):** {', '.join(cols['layers']) if cols['layers'] else 'None'}",
        f"- **Annotations (uns keys):** {', '.join(cols['uns']) if cols['uns'] else 'None'}",
    ]
    return "\n".join(md_lines)


def _edata_snapshot_payload(edata: Any, handle: str, *, used_latest: bool) -> tuple[dict[str, Any], str]:
    """Build the (structured_content, markdown) pair describing an EHRData handle's shape."""
    cols = {
        "obs": _non_none_str_list(edata.obs.columns),
        "var": _non_none_str_list(edata.var.columns),
        "layers": _non_none_str_list(edata.layers.keys()),
        "uns": _non_none_str_list(edata.uns.keys()),
    }
    struct = _snapshot_struct(edata, handle, cols, used_latest=used_latest)
    return struct, _snapshot_markdown(edata, handle, cols)


def get_edata_snapshot(edata_id: str | None = None, ctx: Context = None) -> ToolResult:
    """Return structural metadata for an EHRData handle including observation count, variable count, layers, obs columns, and uns keys.

    Read-only summary of dataset dimensions.
    """
    session = get_session(ctx)
    handle = edata_id or session.get_latest_edata_id()
    if handle is None:
        return mcp_error(
            "get_edata_snapshot",
            "No dataset handle provided and no active dataset in session.",
            error_code="NO_ACTIVE_DATASET",
            agent_action="Load a dataset with load_demo_dataset(dataset='mimic_2') or ingest_dataset(file_path=...).",
        )

    try:
        edata = load_edata(handle)
        struct, md = _edata_snapshot_payload(edata, handle, used_latest=edata_id is None)
        return ToolResult(structured_content=struct, content=md)
    except (KeyError, FileNotFoundError):
        return unknown_handle_error("get_edata_snapshot", "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return mcp_error("get_edata_snapshot", f"Snapshot failed: {exc}", error_code="SNAPSHOT_ERROR")


def list_ehrapy_functions(namespace: str) -> ToolResult:
    """List available functions within a specified ehrapy dispatch namespace.

    Use this to discover supported operations in preprocessing, analysis, get, plot, io, or demo namespaces.
    """
    try:
        funcs = list_functions(namespace)
        struct = {"status": "ok", "namespace": namespace, "count": len(funcs), "functions": funcs}
        md = f"### Available functions in `{namespace}` ({len(funcs)})\n\n" + ", ".join(f"`{f}`" for f in funcs)
        return ToolResult(structured_content=struct, content=md)
    except KeyError:
        valid = list_namespaces()
        return mcp_error(
            "list_ehrapy_functions",
            f"Unknown namespace '{namespace}'. Valid namespaces: {', '.join(valid)}.",
            error_code="UNKNOWN_NAMESPACE",
            agent_action=f"Specify one of the valid namespaces: {', '.join(valid)}.",
        )


def get_function_help(namespace: str, function: str) -> ToolResult:
    """Retrieve signature, parameter descriptions, accepted values, and an example call for any ehrapy function.

    Use this when unsure which arguments a function accepts before calling run_preprocessing, run_analysis, or run_get.
    """
    try:
        info = function_help(namespace, function)
        md = function_help_markdown(namespace, function)
        struct = {
            "status": "ok",
            "namespace": info["namespace"],
            "function": info["function"],
            "kind": info["kind"],
            "parameters": info["parameters"],
            "example_call": info["example_call"],
        }
        return ToolResult(structured_content=struct, content=md)
    except KeyError as exc:
        return mcp_error(
            "get_function_help",
            str(exc),
            error_code="UNKNOWN_FUNCTION",
            agent_action=f"Call list_ehrapy_functions(namespace='{namespace}') to see available functions.",
        )
    except Exception as exc:  # noqa: BLE001
        return mcp_error("get_function_help", f"Failed to retrieve help: {exc}", error_code="HELP_ERROR")


async def run_preprocessing(
    function: str,
    edata_id: str | None = None,
    params: dict | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context = None,
) -> ToolResult:
    """Run an ehrapy preprocessing function (ep.pp.*) on an EHRData object.

    Use this for quality control, encoding, imputation, normalization, filtering, PCA, and neighbor graph computation.
    Updates the dataset in place and returns a column profile of the result.
    Set response_format='detailed' only when you explicitly need sample rows.
    """
    return await run_dispatch(
        "preprocessing",
        function,
        edata_id=edata_id,
        params=params,
        response_format=response_format,
        ctx=ctx,
    )


async def run_analysis(
    function: str,
    edata_id: str | None = None,
    params: dict | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context = None,
) -> ToolResult:
    """Run an ehrapy analysis function (ep.tl.*) on an EHRData object.

    Use this for survival analysis (Kaplan-Meier, Cox PH), causal inference (IPTW, AIPW, G-computation),
    embeddings (UMAP, t-SNE), clustering (Leiden), and differential feature ranking.
    Returns concise statistical summaries or column profiles.
    """
    return await run_dispatch(
        "analysis",
        function,
        edata_id=edata_id,
        params=params,
        response_format=response_format,
        ctx=ctx,
    )


async def run_get(
    function: str,
    edata_id: str | None = None,
    params: dict | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context = None,
) -> ToolResult:
    """Read observation metadata, variable annotations, or ranked feature tables from an EHRData object.

    This is a read-only operation that does not modify the dataset.
    Always specify keys to narrow returned columns and avoid truncated output.
    """
    return await run_dispatch(
        "get",
        function,
        edata_id=edata_id,
        params=params,
        response_format=response_format,
        ctx=ctx,
    )


async def run_plot(
    function: str,
    edata_id: str | None = None,
    params: dict | None = None,
    ctx: Context = None,
) -> ToolResult:
    """Render a visualization for an EHRData object and save it as a PNG artifact.

    Use this for QC plots, survival curves, embedding projections, and causal balance diagnostics.
    Returns an image payload and file path.
    """
    return await run_dispatch(
        "plot",
        function,
        edata_id=edata_id,
        params=params,
        ctx=ctx,
    )


async def run_io(
    function: str,
    edata_id: str | None = None,
    params: dict | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context = None,
) -> ToolResult:
    """Execute an EHRData I/O function to read from or write to disk.

    Use this for advanced file format conversions and data loading.
    """
    return await run_dispatch(
        "io",
        function,
        edata_id=edata_id,
        params=params,
        response_format=response_format,
        ctx=ctx,
    )

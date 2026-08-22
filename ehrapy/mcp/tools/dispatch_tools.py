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
        obs_cols = [str(c) for c in edata.obs.columns if c is not None]
        var_cols = [str(c) for c in edata.var.columns if c is not None]
        layers = [str(k) for k in edata.layers.keys() if k is not None]
        uns_keys = [str(k) for k in edata.uns.keys() if k is not None]

        struct = {
            "status": "ok",
            "edata_id": handle,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "obs_columns": obs_cols,
            "var_columns": var_cols,
            "layers": layers,
            "uns_keys": uns_keys,
        }
        if edata_id is None:
            struct["used_latest"] = True

        md_lines = [
            f"### Snapshot for EHRData `{handle}`",
            f"- **Dimensions:** {edata.n_obs} observations × {edata.n_vars} variables",
            f"- **Observation columns ({len(obs_cols)}):** {', '.join(obs_cols[:25])}{'...' if len(obs_cols) > 25 else ''}",
            f"- **Variable columns ({len(var_cols)}):** {', '.join(var_cols[:25])}{'...' if len(var_cols) > 25 else ''}",
            f"- **Layers ({len(layers)}):** {', '.join(layers) if layers else 'None'}",
            f"- **Annotations (uns keys):** {', '.join(uns_keys) if uns_keys else 'None'}",
        ]
        return ToolResult(structured_content=struct, content="\n".join(md_lines))
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

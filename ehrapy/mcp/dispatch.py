"""Dispatch engine for ehrapy/ehrdata namespaces with dual-channel output."""

from __future__ import annotations

import asyncio
import collections
import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastmcp import Context  # noqa: TC002
from fastmcp.tools.tool import ToolResult
from fastmcp.utilities.types import Image

from ehrapy.mcp.catalog import get_callable, get_namespace_kind
from ehrapy.mcp.edata_store import load_edata, persist_edata, save_edata
from ehrapy.mcp.errors import (
    classify_exception_error,
    mcp_error,
    path_access_error,
    unknown_handle_error,
)
from ehrapy.mcp.policy import SecurityPolicyError, check_path_allowed
from ehrapy.mcp.registry import registry
from ehrapy.mcp.serialization import serialize_result
from ehrapy.mcp.session import get_session
from ehrapy.mcp.steering import get_suggested_next

if TYPE_CHECKING:
    from collections.abc import Callable

    from ehrapy.mcp.session import _EHRapySession

SLOW_FUNCTIONS = frozenset(
    {
        "knn_impute",
        "miss_forest_impute",
        "neighbors",
        "umap",
        "tsne",
        "leiden",
        "iptw",
        "aipw",
        "g_computation",
        "t_learner",
        "read_csv",
    }
)


_FITTER_CACHE_CAPACITY = 8
_FITTER_CACHE: collections.OrderedDict[tuple[str, str], Any] = collections.OrderedDict()

# Plot functions whose first parameter is a fitted model/result rather than the EHRData
# object, mapped to the run_analysis function that produces it. Ported from origin fix #5.
# The third element is an attribute the bound object must expose; ep.tl.kaplan_meier also
# writes a summary DataFrame to uns["kaplan_meier"], and binding that instead of the fitter
# reproduces the exact AttributeError this mapping exists to prevent.
_PLOT_FITTER_SOURCES: dict[str, tuple[str, str, str | None]] = {
    # plot function -> (analysis function that fits it, parameter it binds to, required attr)
    "kaplan_meier": ("kaplan_meier", "kmfs", "survival_function_"),
    "love_plot": ("covariate_balance", "balance", None),
    "propensity_overlap": ("positivity_check", "positivity", None),
}


def _usable_fitter(value: Any, required_attr: str | None) -> bool:
    """Return True if ``value`` looks like the fitted object the plot expects."""
    if value is None:
        return False
    if required_attr is None:
        return True
    if isinstance(value, (list, tuple)):
        return bool(value) and all(hasattr(v, required_attr) for v in value)
    return hasattr(value, required_attr)


def _handle_mtime_ns(edata_id: str) -> int | None:
    """Return the current cache mtime for a handle, or None if it is unknown."""
    record = registry.get_dataset(edata_id)
    return record.mtime_ns if record is not None else None


def cache_fitter(edata_id: str, function: str, value: Any) -> None:
    """Record a fitted model so a downstream plot call can bind it (bounded LRU).

    Stamped with the handle's cache mtime: any later write-through to the same
    ``edata_id`` invalidates the fitter, so a plot can never silently render a model
    fitted against a cohort that has since been transformed.
    """
    key = (edata_id, function)
    if key in _FITTER_CACHE:
        _FITTER_CACHE.move_to_end(key)
    _FITTER_CACHE[key] = (_handle_mtime_ns(edata_id), value)
    while len(_FITTER_CACHE) > _FITTER_CACHE_CAPACITY:
        _FITTER_CACHE.popitem(last=False)


def get_fitter(edata_id: str, function: str) -> Any | None:
    """Return a cached fitter for a handle, or None if absent or stale."""
    key = (edata_id, function)
    if key not in _FITTER_CACHE:
        return None
    cached_mtime, value = _FITTER_CACHE[key]
    if cached_mtime != _handle_mtime_ns(edata_id):
        # The dataset changed under the fitter; treat it as absent so the caller
        # tells the agent to re-run the analysis.
        del _FITTER_CACHE[key]
        return None
    _FITTER_CACHE.move_to_end(key)
    return value


def clear_fitter_cache() -> None:
    """Clear the fitter cache (useful for testing)."""
    _FITTER_CACHE.clear()


def _resolve_plot_fitter(
    edata: Any,
    edata_id: str | None,
    function: str,
    bind_name: str,
    source_fn: str,
    required_attr: str | None,
) -> Any:
    """Resolve the fitted object a plot function should bind, raising if none is available."""
    fitted = get_fitter(edata_id or "", source_fn)
    if not _usable_fitter(fitted, required_attr) and hasattr(edata, "uns"):
        # Fall back to a copy persisted in uns, but only if it is the fitted
        # object rather than a summary table stored under the same key.
        candidate = edata.uns.get(source_fn)
        fitted = candidate if _usable_fitter(candidate, required_attr) else None
    if not _usable_fitter(fitted, required_attr):
        raise ValueError(
            f"No fitted result available for plot '{function}'. "
            f"Run run_analysis(function='{source_fn}', ...) on this handle first "
            f"(re-run it if the dataset has been transformed since)."
        )
    if bind_name == "kmfs" and not isinstance(fitted, (list, tuple)):
        fitted = [fitted]
    return fitted


def _build_call_params(
    fn: Callable[..., Any],
    edata: Any,
    params: dict[str, Any],
    *,
    edata_id: str | None = None,
    function: str | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Build the kwargs for a dispatched call, binding edata or a fitted model as needed.

    Most ehrapy functions take the EHRData object first. A few plot functions instead take a
    fitted model produced by an earlier run_analysis call (ported from origin fix #5); for
    those, bind the cached fitter -- or the copy persisted in ``edata.uns`` -- rather than
    handing them an EHRData they cannot use.
    """
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    if not param_names:
        return dict(params)

    first_param = param_names[0]
    if first_param in params:
        return dict(params)

    if kind == "plot" and function in _PLOT_FITTER_SOURCES:
        source_fn, bind_name, required_attr = _PLOT_FITTER_SOURCES[function]
        if bind_name in sig.parameters:
            fitted = _resolve_plot_fitter(edata, edata_id, function, bind_name, source_fn, required_attr)
            return {bind_name: fitted, **params}

    return {first_param: edata, **params}


def _inject_edata(fn: Callable[..., Any], edata: Any, params: dict[str, Any], **kwargs: Any) -> Any:
    """Call ``fn`` with edata (or a fitted model) bound to its first parameter."""
    return fn(**_build_call_params(fn, edata, params, **kwargs))


# --- Shared helpers for the dispatch branches below --------------------------------------
#
# Every branch assembles the same "suggested_next" affordance and a subset share path-policy
# checks or provenance bookkeeping; factored out here so each branch reads as its own logic
# rather than repeating this scaffolding.


def _suggestion_lines(suggestions: list[dict[str, str]]) -> list[str]:
    """Format suggested-next-call entries as markdown bullet lines."""
    return [f"- `{s['call']}` — {s['reason']}" for s in suggestions]


def _apply_suggestions(struct: dict[str, Any], suggestions: list[dict[str, str]]) -> None:
    """Attach a ``suggested_next`` field to ``struct`` when suggestions exist."""
    if suggestions:
        struct["suggested_next"] = [s["call"] for s in suggestions]


def _content_with_suggestions(content_payload: Any, suggestions: list[dict[str, str]]) -> str:
    """Render a result payload as markdown text, appending suggested-next calls."""
    md_content = content_payload if isinstance(content_payload, str) else "\n".join(str(c) for c in content_payload)
    if suggestions:
        md_content += "\n".join(["\n\n**Suggested next:**", *_suggestion_lines(suggestions)])
    return md_content


def _policy_error(tool_name: str, exc: SecurityPolicyError) -> ToolResult:
    """Translate a path/read-only policy violation into an MCP error result."""
    return mcp_error(tool_name, str(exc), error_code=exc.error_code, agent_action=exc.agent_action)


def _resolve_path_param(params: dict[str, Any], *, for_write: bool, operation: str = "write") -> str | None:
    """Resolve and rewrite a filename/path/file_path param against the path policy."""
    raw = params.get("filename") or params.get("path") or params.get("file_path")
    if raw:
        checked = check_path_allowed(raw, for_write=for_write, operation=operation)
        for k in ("filename", "path", "file_path"):
            if k in params:
                params[k] = str(checked)
    return raw


async def _report_progress_if_slow(
    ctx: Context | None, function: str, *, progress: int, info: str | None = None
) -> None:
    """Report MCP progress for long-running functions; a no-op for fast ones or without a ctx."""
    if ctx is None or function not in SLOW_FUNCTIONS:
        return
    try:
        if info is not None:
            await ctx.info(info)
        await ctx.report_progress(progress=progress, total=100)
    except Exception:  # noqa: BLE001
        pass


def _record_op_log(edata: Any, namespace: str, function: str, params: dict[str, Any]) -> None:
    """Append a provenance entry to edata.uns for this dispatched call (T19)."""
    import json

    ops = edata.uns.setdefault("ehrapy_mcp_ops", [])
    ops.append(
        json.dumps(
            {
                "namespace": namespace,
                "function": function,
                "params": {
                    str(k): (v if isinstance(v, (int, float, str, bool, list)) else str(v)) for k, v in params.items()
                },
                "timestamp": time.time(),
            }
        )
    )


def _maybe_cache_fitter(handle: str, function: str, res: Any) -> None:
    """Stamp a freshly produced fitter so a later plot call can bind it (see cache_fitter)."""
    if res is not None and function in {src for src, _, _ in _PLOT_FITTER_SOURCES.values()}:
        cache_fitter(handle, function, res)


# --- Dispatch branches, one per namespace kind --------------------------------------------


async def _dispatch_demo(
    fn: Callable[..., Any],
    namespace: str,
    function: str,
    params: dict[str, Any],
    session: _EHRapySession,
    tool_name: str,
    ctx: Context | None,
) -> ToolResult:
    """Load a demo dataset and cache it under a new handle."""
    try:
        edata = fn(**params)
        record = save_edata(edata, name=function, fmt="demo")
        session.set_latest_edata_id(record.edata_id)

        suggestions = get_suggested_next(namespace, function)
        struct: dict[str, Any] = {
            "status": "ok",
            "edata_id": record.edata_id,
            "name": record.name,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "namespace": namespace,
            "function": function,
        }
        _apply_suggestions(struct, suggestions)

        md_lines = [
            f"### Loaded demo dataset `{function}`",
            f"- **Handle (edata_id):** `{record.edata_id}`",
            f"- **Observations:** {edata.n_obs}",
            f"- **Variables:** {edata.n_vars}",
        ]
        if suggestions:
            md_lines.extend(["", "**Suggested next:**", *_suggestion_lines(suggestions)])

        await _report_progress_if_slow(ctx, function, progress=100)

        return ToolResult(structured_content=struct, content="\n".join(md_lines))
    except Exception as exc:  # noqa: BLE001
        return mcp_error(
            tool_name,
            f"Failed to load demo dataset '{function}': {exc}",
            error_code="DEMO_LOAD_ERROR",
        )


def _dispatch_io_read(
    fn: Callable[..., Any],
    namespace: str,
    function: str,
    params: dict[str, Any],
    session: _EHRapySession,
    tool_name: str,
) -> ToolResult:
    """Read a dataset from disk via a standalone IO function and cache it."""
    try:
        file_arg = _resolve_path_param(params, for_write=False)

        edata = fn(**params)
        name_stem = Path(file_arg).name if file_arg else function
        record = save_edata(edata, name=name_stem, source_path=str(file_arg) if file_arg else None)
        session.set_latest_edata_id(record.edata_id)

        suggestions = get_suggested_next(namespace, function)
        struct = {
            "status": "ok",
            "edata_id": record.edata_id,
            "name": record.name,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "namespace": namespace,
            "function": function,
        }
        _apply_suggestions(struct, suggestions)

        md_lines = [
            f"### Loaded dataset from `{file_arg or function}`",
            f"- **Handle (edata_id):** `{record.edata_id}`",
            f"- **Observations:** {edata.n_obs}",
            f"- **Variables:** {edata.n_vars}",
        ]
        if suggestions:
            md_lines.extend(["", "**Suggested next:**", *_suggestion_lines(suggestions)])

        return ToolResult(structured_content=struct, content="\n".join(md_lines))
    except SecurityPolicyError as exc:
        return _policy_error(tool_name, exc)
    except FileNotFoundError:
        return path_access_error(tool_name, str(params.get("filename") or params.get("path") or ""))
    except Exception as exc:  # noqa: BLE001
        return classify_exception_error(
            tool_name, exc, namespace=namespace, function=function, fallback_code="IO_READ_ERROR"
        )


def _dispatch_get(
    fn: Callable[..., Any],
    edata: Any,
    handle: str,
    used_latest: bool,
    namespace: str,
    function: str,
    params: dict[str, Any],
    response_format: Literal["concise", "detailed"],
    tool_name: str,
) -> ToolResult:
    """Run a read-only ``get`` function against a cached dataset."""
    try:
        res = _inject_edata(fn, edata, params)
        meta, content_payload = serialize_result(
            res,
            response_format=response_format,
            params=params,
            function=function,
            edata=edata,
        )
        suggestions = get_suggested_next(namespace, function)

        status_val = meta.get("status", "ok")
        struct = {
            "status": status_val,
            "edata_id": handle,
            "namespace": namespace,
            "function": function,
            **{k: v for k, v in meta.items() if k != "status"},
        }
        if used_latest:
            struct["used_latest"] = True
        _apply_suggestions(struct, suggestions)

        md_content = _content_with_suggestions(content_payload, suggestions)

        return ToolResult(structured_content=struct, content=md_content)
    except Exception as exc:  # noqa: BLE001
        return classify_exception_error(
            tool_name, exc, namespace=namespace, function=function, fallback_code="GET_ERROR"
        )


def _inject_plot_defaults(sig: inspect.Signature, params: dict[str, Any]) -> None:
    """Default a plot call to a non-interactive, figure-returning invocation."""
    # Inject show=False if function accepts show
    if "show" in sig.parameters and "show" not in params:
        params["show"] = False
    # Prefer an explicit figure handle over scraping plt.gcf() (origin fix #4).
    if "return_fig" in sig.parameters and "return_fig" not in params:
        params["return_fig"] = True


def _dispatch_plot(
    fn: Callable[..., Any],
    edata: Any,
    handle: str,
    used_latest: bool,
    namespace: str,
    function: str,
    params: dict[str, Any],
    tool_name: str,
) -> ToolResult:
    """Run a plot function against a cached dataset and render the resulting figure."""
    try:
        _inject_plot_defaults(inspect.signature(fn), params)

        plt.close("all")
        res = _inject_edata(fn, edata, params, edata_id=handle, function=function, kind="plot")
        if res is None and plt.get_fignums():
            res = plt.gcf()

        meta, content_payload = serialize_result(
            res,
            plots_dir=registry.plots_dir(),
            params=params,
            function=function,
            edata=edata,
        )
        plt.close("all")

        suggestions = get_suggested_next(namespace, function)
        rendered = meta.get("type") == "figure"
        struct = {
            "status": "ok" if rendered else "ok_no_figure",
            "edata_id": handle,
            "namespace": namespace,
            "function": function,
            **meta,
        }
        if not rendered:
            # A plot call that produced no image previously reported a bare "ok",
            # leaving the agent to assume a figure existed.
            struct["agent_action"] = (
                f"'{function}' returned {meta.get('type', 'no renderable figure')} rather than a figure. "
                "Check that the required inputs were fitted first, or call "
                f"get_function_help(namespace='{namespace}', function='{function}')."
            )
        if used_latest:
            struct["used_latest"] = True
        _apply_suggestions(struct, suggestions)

        content_blocks: list[Any] = list(content_payload) if isinstance(content_payload, list) else [content_payload]
        if suggestions:
            content_blocks.append("\n".join(["\n\n**Suggested next:**", *_suggestion_lines(suggestions)]))

        return ToolResult(structured_content=struct, content=content_blocks)
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return classify_exception_error(
            tool_name, exc, namespace=namespace, function=function, fallback_code="PLOT_ERROR"
        )


def _dispatch_to_pandas(
    fn: Callable[..., Any],
    edata: Any,
    handle: str,
    used_latest: bool,
    namespace: str,
    function: str,
    params: dict[str, Any],
    response_format: Literal["concise", "detailed"],
) -> ToolResult:
    """Convert a cached dataset to a pandas DataFrame and serialize it."""
    df = fn(edata, **params)
    meta, content_payload = serialize_result(
        df,
        response_format=response_format,
        params=params,
        function=function,
        edata=edata,
    )
    struct = {
        "status": "ok",
        "edata_id": handle,
        "namespace": namespace,
        "function": function,
        **meta,
    }
    if used_latest:
        struct["used_latest"] = True
    return ToolResult(
        structured_content=struct,
        content=(content_payload if isinstance(content_payload, str) else str(content_payload)),
    )


def _dispatch_io_write(
    fn: Callable[..., Any],
    edata: Any,
    handle: str,
    used_latest: bool,
    namespace: str,
    function: str,
    params: dict[str, Any],
    response_format: Literal["concise", "detailed"],
    tool_name: str,
) -> ToolResult:
    """Run an IO export/write function (or ``to_pandas``) against a cached dataset."""
    try:
        if function == "to_pandas":
            return _dispatch_to_pandas(fn, edata, handle, used_latest, namespace, function, params, response_format)

        # write functions
        target_path = _resolve_path_param(params, for_write=True, operation=function)
        fn(edata, **params)
        struct = {
            "status": "ok",
            "edata_id": handle,
            "namespace": namespace,
            "function": function,
            "target_path": str(target_path) if target_path else None,
        }
        if used_latest:
            struct["used_latest"] = True
        md = f"Dataset `{handle}` successfully exported to `{target_path}`."
        return ToolResult(structured_content=struct, content=md)
    except SecurityPolicyError as exc:
        return _policy_error(tool_name, exc)
    except Exception as exc:  # noqa: BLE001
        return classify_exception_error(
            tool_name, exc, namespace=namespace, function=function, fallback_code="IO_ERROR"
        )


async def _dispatch_edata(
    fn: Callable[..., Any],
    edata: Any,
    handle: str,
    used_latest: bool,
    kind: str,
    namespace: str,
    function: str,
    params: dict[str, Any],
    response_format: Literal["concise", "detailed"],
    session: _EHRapySession,
    tool_name: str,
    ctx: Context | None,
) -> ToolResult:
    """Run a mutating preprocessing/analysis function and persist the result."""
    try:
        res = _inject_edata(fn, edata, params, edata_id=handle, function=function, kind=kind)
        if hasattr(res, "n_obs") and hasattr(res, "n_vars"):
            # Function returned a modified copy
            edata = res

        _record_op_log(edata, namespace, function, params)

        # Persist dataset write-through
        persist_edata(handle, edata)
        session.set_latest_edata_id(handle)

        # Stamp the fitter with the post-write mtime, so it stays valid until the
        # next transformation of this handle rather than invalidating immediately.
        _maybe_cache_fitter(handle, function, res)

        meta, content_payload = serialize_result(
            res,
            response_format=response_format,
            params=params,
            function=function,
            edata=edata,
        )
        suggestions = get_suggested_next(namespace, function)

        struct = {
            "status": "ok",
            "edata_id": handle,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "namespace": namespace,
            "function": function,
            **meta,
        }
        if used_latest:
            struct["used_latest"] = True
        _apply_suggestions(struct, suggestions)

        md_content = _content_with_suggestions(content_payload, suggestions)

        await _report_progress_if_slow(ctx, function, progress=100)

        return ToolResult(structured_content=struct, content=md_content)
    except TypeError as exc:
        return mcp_error(
            tool_name,
            f"Invalid parameter for {namespace}.{function}: {exc}",
            error_code="INVALID_PARAMETER",
            agent_action=f"Call get_function_help(namespace='{namespace}', function='{function}') to inspect valid parameters.",
        )
    except ValueError as exc:
        return mcp_error(
            tool_name,
            f"Value error in {namespace}.{function}: {exc}",
            error_code="INVALID_VALUE",
            agent_action=f"Inspect data or parameters for {namespace}.{function}.",
        )
    except Exception as exc:  # noqa: BLE001
        return classify_exception_error(tool_name, exc, namespace=namespace, function=function)


def _resolve_handle(edata_id: str | None, session: _EHRapySession) -> tuple[str | None, bool]:
    """Return (handle, used_latest): the explicit handle, or the session's latest."""
    if edata_id is not None:
        return edata_id, False
    return session.get_latest_edata_id(), True


def _load_dispatch_edata(handle: str, tool_name: str) -> tuple[Any, ToolResult | None]:
    """Load a cached dataset, returning (edata, None) or (None, error result)."""
    try:
        return load_edata(handle), None
    except (KeyError, FileNotFoundError):
        return None, unknown_handle_error(tool_name, "edata_id", handle)
    except Exception as exc:  # noqa: BLE001
        return None, mcp_error(
            tool_name,
            f"Failed to load dataset '{handle}': {exc}",
            error_code="DATASET_LOAD_ERROR",
        )


async def run_dispatch(
    namespace: str,
    function: str,
    *,
    edata_id: str | None = None,
    params: dict[str, Any] | None = None,
    response_format: Literal["concise", "detailed"] = "concise",
    ctx: Context | None = None,
) -> ToolResult:
    """Execute a function from an ehrapy namespace and return a dual-channel ToolResult."""
    params = dict(params or {})
    tool_name = f"run_{namespace}" if namespace != "demo" else "load_demo_dataset"
    session = get_session(ctx)

    await _report_progress_if_slow(ctx, function, progress=0, info=f"Starting {namespace}.{function}...")

    # Resolve callable
    try:
        fn = get_callable(namespace, function)
        kind = get_namespace_kind(namespace)
    except KeyError as exc:
        return mcp_error(
            tool_name,
            str(exc),
            error_code="UNKNOWN_FUNCTION",
            agent_action=f"Call list_ehrapy_functions(namespace='{namespace}') to inspect valid function names.",
        )

    # 1. Demo datasets
    if kind == "demo":
        return await _dispatch_demo(fn, namespace, function, params, session, tool_name, ctx)

    # 2. IO read functions (standalone ingestion)
    if kind == "io" and function.startswith("read_"):
        return _dispatch_io_read(fn, namespace, function, params, session, tool_name)

    # For all other operations, resolve the active dataset handle
    handle, used_latest = _resolve_handle(edata_id, session)
    if handle is None:
        return mcp_error(
            tool_name,
            "No dataset handle provided and no active dataset in session.",
            error_code="NO_ACTIVE_DATASET",
            agent_action="Load a dataset with load_demo_dataset(dataset='mimic_2') or ingest_dataset(file_path=...).",
        )

    edata, error = _load_dispatch_edata(handle, tool_name)
    if error is not None:
        return error

    # 3. Read-Only get namespace (T1)
    if kind == "get":
        return _dispatch_get(fn, edata, handle, used_latest, namespace, function, params, response_format, tool_name)

    # 4. Plot namespace (T10)
    if kind == "plot":
        return _dispatch_plot(fn, edata, handle, used_latest, namespace, function, params, tool_name)

    # 5. IO write / export / to_pandas functions
    if kind == "io":
        return _dispatch_io_write(
            fn, edata, handle, used_latest, namespace, function, params, response_format, tool_name
        )

    # 6. Mutating edata namespaces (preprocessing & analysis)
    if kind == "edata":
        return await _dispatch_edata(
            fn, edata, handle, used_latest, kind, namespace, function, params, response_format, session, tool_name, ctx
        )

    return mcp_error(tool_name, f"Unhandled namespace kind '{kind}'", error_code="UNHANDLED_KIND")

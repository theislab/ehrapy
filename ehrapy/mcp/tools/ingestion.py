"""Dataset ingestion and export tools for ehrapy MCP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import ehrdata.io as ed_io
from fastmcp import Context  # noqa: TC002
from fastmcp.tools.tool import ToolResult

from ehrapy.mcp.edata_store import load_edata, save_edata
from ehrapy.mcp.errors import mcp_error, path_access_error, unknown_handle_error
from ehrapy.mcp.policy import PathNotAllowedError, ReadOnlyModeError, check_path_allowed
from ehrapy.mcp.session import get_session
from ehrapy.mcp.steering import get_suggested_next


def ingest_dataset(
    file_path: str,
    dataset_name: str | None = None,
    sep: str = ",",
    index_col: str | None = None,
    ctx: Context = None,
) -> ToolResult:
    """Read an external tabular data file (CSV, TSV) and convert it into a managed EHRData handle.

    Use this to load clinical cohorts from the host filesystem into the session cache.
    """
    try:
        resolved_path = check_path_allowed(file_path, for_write=False)
    except PathNotAllowedError as exc:
        return mcp_error("ingest_dataset", str(exc), error_code=exc.error_code, agent_action=exc.agent_action)
    except ReadOnlyModeError as exc:
        return mcp_error("ingest_dataset", str(exc), error_code=exc.error_code, agent_action=exc.agent_action)

    if not resolved_path.is_file():
        return path_access_error("ingest_dataset", str(file_path))

    try:
        read_kwargs: dict[str, Any] = {"sep": sep}
        if index_col is not None:
            read_kwargs["index_column"] = index_col
        edata = ed_io.read_csv(resolved_path, **read_kwargs)
        name = dataset_name or resolved_path.stem
        record = save_edata(edata, name=name, source_path=str(resolved_path))

        session = get_session(ctx)
        session.set_latest_edata_id(record.edata_id)

        suggestions = get_suggested_next("io", "read_csv")
        struct: dict[str, Any] = {
            "status": "ok",
            "edata_id": record.edata_id,
            "name": record.name,
            "n_obs": edata.n_obs,
            "n_vars": edata.n_vars,
            "source_path": str(resolved_path),
        }
        if suggestions:
            struct["suggested_next"] = [s["call"] for s in suggestions]

        md_lines = [
            f"### Ingested dataset `{record.name}`",
            f"- **Handle (edata_id):** `{record.edata_id}`",
            f"- **Observations:** {edata.n_obs}",
            f"- **Variables:** {edata.n_vars}",
            f"- **Source file:** `{resolved_path}`",
        ]
        if suggestions:
            md_lines.extend(["", "**Suggested next:**"])
            for s in suggestions:
                md_lines.append(f"- `{s['call']}` — {s['reason']}")

        return ToolResult(structured_content=struct, content="\n".join(md_lines))
    except Exception as exc:  # noqa: BLE001
        return mcp_error("ingest_dataset", f"Failed to ingest file '{file_path}': {exc}", error_code="INGEST_ERROR")


def export_edata(
    target_path: str,
    edata_id: str | None = None,
    fmt: Literal["h5ad", "zarr", "csv"] = "h5ad",
    ctx: Context = None,
) -> ToolResult:
    """Export an EHRData object to a specified file path.

    Supports writing to h5ad, zarr, or CSV formats.
    """
    session = get_session(ctx)
    used_latest = False
    handle = edata_id
    if handle is None:
        handle = session.get_latest_edata_id()
        used_latest = True

    if handle is None:
        return mcp_error(
            "export_edata",
            "No dataset handle provided and no active dataset in session.",
            error_code="NO_ACTIVE_DATASET",
            agent_action="Load a dataset with load_demo_dataset(dataset='mimic_2') or ingest_dataset(file_path=...).",
        )

    try:
        resolved_path = check_path_allowed(target_path, for_write=True, operation="export_edata")
    except PathNotAllowedError as exc:
        return mcp_error("export_edata", str(exc), error_code=exc.error_code, agent_action=exc.agent_action)
    except ReadOnlyModeError as exc:
        return mcp_error("export_edata", str(exc), error_code=exc.error_code, agent_action=exc.agent_action)

    try:
        edata = load_edata(handle)
        if fmt == "h5ad":
            ed_io.write_h5ad(edata, resolved_path)
        elif fmt == "zarr":
            ed_io.write_zarr(edata, resolved_path)
        elif fmt == "csv":
            df = ed_io.to_pandas(edata)
            df.to_csv(resolved_path)
        else:
            return mcp_error("export_edata", f"Unsupported export format '{fmt}'", error_code="INVALID_FORMAT")

        struct: dict[str, Any] = {
            "status": "ok",
            "edata_id": handle,
            "target_path": str(resolved_path),
            "format": fmt,
        }
        if used_latest:
            struct["used_latest"] = True

        md = f"Dataset `{handle}` successfully exported to `{resolved_path}` ({fmt} format)."
        return ToolResult(structured_content=struct, content=md)
    except Exception as exc:  # noqa: BLE001
        return mcp_error("export_edata", f"Export failed: {exc}", error_code="EXPORT_ERROR")

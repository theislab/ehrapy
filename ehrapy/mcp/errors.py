"""Structured MCP error envelopes with dual-channel (JSON + Markdown) output."""

from __future__ import annotations

from typing import Any

from fastmcp.tools.tool import ToolResult

_SANDBOX_PATH_PREFIXES = (
    "/home/claude",
    "/mnt/",
    "/workspace/",
    "/tmp/claude",
    "/System/Volumes/Data/home/claude",
)


def format_error_markdown(reason: str, *, error_code: str | None = None, agent_action: str | None = None) -> str:
    """Render a clean 2-line Markdown block for error results."""
    title = f"# Error: {error_code}" if error_code else "# Error"
    lines = [title, "", reason]
    if agent_action:
        lines.extend(["", f"**Action:** {agent_action}"])
    return "\n".join(lines)


def mcp_error(
    tool: str,
    reason: str,
    *,
    error_code: str | None = None,
    agent_action: str | None = None,
    details: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    is_error: bool = False,
) -> ToolResult:
    """Return a dual-channel error ToolResult."""
    # Truncate reason at 300 chars to avoid leaking huge internal tracebacks
    clean_reason = reason[:300] if len(reason) > 300 else reason
    struct: dict[str, Any] = {
        "status": "error",
        "tool": tool,
        "reason": clean_reason,
        "error_code": error_code or "EXECUTION_ERROR",
    }
    if agent_action:
        struct["agent_action"] = agent_action
    if details:
        struct["details"] = details
    if metrics:
        struct["metrics"] = metrics

    md = format_error_markdown(clean_reason, error_code=error_code, agent_action=agent_action)
    return ToolResult(content=md, structured_content=struct, is_error=is_error)


def classify_path(path: str) -> dict[str, Any]:
    """Flag paths that look like agent-sandbox locations."""
    matched_prefix = next(
        (prefix for prefix in _SANDBOX_PATH_PREFIXES if path.startswith(prefix)),
        None,
    )
    return {
        "attempted_path": path,
        "looks_like_sandbox_path": matched_prefix is not None,
        "matched_prefix": matched_prefix,
    }


def path_access_error(
    tool: str,
    path: str,
    *,
    missing_code: str = "FILE_NOT_FOUND",
    missing_reason: str = "Path does not exist on the MCP host filesystem.",
    missing_action: str = "Provide a host-visible absolute path, or call get_runtime_context first.",
    sandbox_action: str | None = None,
) -> ToolResult:
    """Return a path-access error ToolResult."""
    path_context = classify_path(path)
    if path_context["looks_like_sandbox_path"]:
        return mcp_error(
            tool,
            "Path is not visible to the MCP server host filesystem.",
            error_code="HOST_PATH_NOT_VISIBLE",
            agent_action=sandbox_action
            or "Ask the user for a host-visible absolute path, or call get_runtime_context first.",
            details={"path_context": path_context},
        )

    return mcp_error(
        tool,
        missing_reason,
        error_code=missing_code,
        agent_action=missing_action,
        details={"path_context": path_context},
    )


def unknown_handle_error(tool: str, handle_name: str, handle_value: str) -> ToolResult:
    """Return an unknown-handle error ToolResult."""
    return mcp_error(
        tool,
        f"Unknown {handle_name} '{handle_value}'.",
        error_code=f"{handle_name.upper()}_UNKNOWN",
        agent_action=f"Create or retrieve a valid {handle_name} (e.g. via load_demo_dataset or ingest_dataset) before retrying.",
        details={handle_name: handle_value},
    )


def unknown_argument_error(tool: str, extra_keys: set[str], valid_keys: set[str]) -> ToolResult:
    """Return UNKNOWN_ARGUMENT error ToolResult."""
    extra_str = ", ".join(sorted(extra_keys))
    valid_str = ", ".join(sorted(valid_keys))
    return mcp_error(
        tool,
        f"Unknown argument(s): {extra_str}. Valid arguments for {tool}: {valid_str}.",
        error_code="UNKNOWN_ARGUMENT",
        agent_action="Rename the argument and retry. Function kwargs go inside `params`.",
        details={
            "unknown_arguments": sorted(extra_keys),
            "valid_arguments": sorted(valid_keys),
        },
    )


def classify_exception_error(
    tool: str,
    exc: Exception,
    *,
    namespace: str | None = None,
    function: str | None = None,
    fallback_code: str = "EXECUTION_ERROR",
    fallback_action: str | None = None,
) -> ToolResult:
    """Classify an exception into a structured error ToolResult with a specific error_code.

    Ported from origin fix #6. Every branch yields a non-generic ``error_code`` and an
    ``agent_action``, so an agent that hits a failure always has a next move.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    details: dict[str, Any] = {"exception_type": exc_type}
    if namespace:
        details["namespace"] = namespace
    if function:
        details["function"] = function

    help_action = (
        f"Call get_function_help(namespace='{namespace}', function='{function}') to inspect expected parameters."
        if namespace and function
        else "Inspect the tool parameters and retry."
    )

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "Install with" in exc_msg:
        return mcp_error(
            tool,
            exc_msg,
            error_code="DEPENDENCY_MISSING",
            agent_action="Install the required optional dependency or ehrapy extra, then retry.",
            details=details,
        )

    if isinstance(exc, TypeError):
        return mcp_error(tool, exc_msg, error_code="INVALID_INPUT", agent_action=help_action, details=details)

    if isinstance(exc, FileNotFoundError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="FILE_NOT_FOUND",
            agent_action="Verify the file path exists on the MCP host filesystem.",
            details=details,
        )

    if isinstance(exc, KeyError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="FUNCTION_UNKNOWN" if "Unknown" in exc_msg else "KEY_NOT_FOUND",
            agent_action=("Check the requested key or column name against get_edata_snapshot() for available columns."),
            details=details,
        )

    if isinstance(exc, AttributeError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="INVALID_INPUT",
            agent_action=help_action,
            details=details,
        )

    if isinstance(exc, ValueError):
        return mcp_error(
            tool,
            exc_msg,
            error_code="INVALID_VALUE",
            agent_action=fallback_action or help_action,
            details=details,
        )

    return mcp_error(
        tool,
        exc_msg,
        error_code=fallback_code,
        agent_action=fallback_action or help_action,
        details=details,
    )

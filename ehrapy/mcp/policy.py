"""Security policy and filesystem confinement for ehrapy MCP."""

from __future__ import annotations

import os
from pathlib import Path


class SecurityPolicyError(Exception):
    """Base exception for security policy violations."""

    def __init__(self, message: str, *, error_code: str, agent_action: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.agent_action = agent_action


class PathNotAllowedError(SecurityPolicyError):
    """Raised when an accessed path is outside EHRAPY_MCP_ALLOWED_ROOTS."""

    def __init__(self, path: str, allowed_roots: list[Path]) -> None:
        roots_str = ", ".join(str(r) for r in allowed_roots)
        super().__init__(
            f"Path '{path}' is outside allowed roots: {roots_str}",
            error_code="PATH_NOT_ALLOWED",
            agent_action=f"Provide a path within one of the allowed roots: {roots_str}",
        )
        self.path = path
        self.allowed_roots = allowed_roots


class ReadOnlyModeError(SecurityPolicyError):
    """Raised when a write operation is attempted in read-only mode."""

    def __init__(self, operation: str = "write") -> None:
        super().__init__(
            f"Operation '{operation}' is prohibited: server is in read-only mode (EHRAPY_MCP_READ_ONLY=1).",
            error_code="READ_ONLY_MODE",
            agent_action="Read-only mode is active. Write/export operations are blocked.",
        )


def is_read_only_mode() -> bool:
    """Return True if EHRAPY_MCP_READ_ONLY is set to 1, true, or yes."""
    val = os.environ.get("EHRAPY_MCP_READ_ONLY", "").strip().lower()
    return val in {"1", "true", "yes"}


def get_allowed_roots() -> list[Path] | None:
    """Return configured allowed roots or None if confinement is not enabled."""
    val = os.environ.get("EHRAPY_MCP_ALLOWED_ROOTS", "").strip()
    if not val:
        return None
    roots: list[Path] = []
    for part in val.split(":"):
        part = part.strip()
        if part:
            roots.append(Path(part).expanduser().resolve())
    return roots if roots else None


def check_path_allowed(path: str | Path, *, for_write: bool = False, operation: str = "write") -> Path:
    """Validate a path against read-only mode and allowed roots.

    Returns the resolved Path if allowed.
    """
    if for_write and is_read_only_mode():
        raise ReadOnlyModeError(operation)

    target = Path(path).expanduser().resolve()
    allowed_roots = get_allowed_roots()
    if allowed_roots is not None:
        allowed = False
        for root in allowed_roots:
            try:
                target.relative_to(root)
                allowed = True
                break
            except ValueError:
                continue
        if not allowed:
            raise PathNotAllowedError(str(path), allowed_roots)

    return target

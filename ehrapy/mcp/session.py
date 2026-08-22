"""Per-client MCP session state."""

from __future__ import annotations

import threading
from typing import Any

_SESSIONS_LOCK = threading.RLock()


class _EHRapySession:
    """Thread-safe per-client container for the active EHRData handle."""

    def __init__(self, edata_id: str | None = None) -> None:
        self._edata_id = edata_id
        self._lock = threading.RLock()

    @property
    def edata_id(self) -> str | None:
        """Return the active edata_id for this session."""
        with self._lock:
            return self._edata_id

    @edata_id.setter
    def edata_id(self, value: str | None) -> None:
        with self._lock:
            self._edata_id = value

    def get_latest_edata_id(self) -> str | None:
        """Return the active edata_id for this session."""
        return self.edata_id

    def set_latest_edata_id(self, edata_id: str | None) -> None:
        """Set the active edata_id for this session."""
        self.edata_id = edata_id


_SESSIONS: dict[str, _EHRapySession] = {}


def _session_key(ctx: Any) -> str:
    """Derive a stable per-client key from a FastMCP Context.

    Each attribute is probed defensively: FastMCP raises when these are touched
    outside an active request context, and a server-wide fallback to "default" is
    preferable to propagating that error into every tool call.
    """
    if ctx is None:
        return "default"
    for attr in ("client_id", "session_id", "request_id"):
        try:
            value = getattr(ctx, attr, None)
        except (RuntimeError, AttributeError):
            continue
        if value:
            return str(value)
    return "default"


def get_session(ctx: Any = None) -> _EHRapySession:
    """Return the per-client session, creating one if needed."""
    key = _session_key(ctx)
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            _SESSIONS[key] = _EHRapySession()
        return _SESSIONS[key]


def reset_sessions() -> None:
    """Reset all session states (useful for testing)."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()

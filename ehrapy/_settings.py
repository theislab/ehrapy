from __future__ import annotations

from typing import Literal

from ehrdata._logger import logger
from pydantic import field_validator
from scverse_misc import Settings

_VerbosityName = Literal["error", "warning", "success", "info", "hint", "debug"]
_VERBOSITY_NAME_TO_INT: dict[_VerbosityName, int] = {
    "error": 0,
    "warning": 1,
    "success": 2,
    "info": 3,
    "hint": 4,
    "debug": 5,
}


class _EhrapySettings(Settings, exported_object_name="settings", docstring_style="google"):  # type: ignore[call-arg]
    verbosity: _VerbosityName = "warning"
    """Logger verbosity (one of ``'error'``, ``'warning'``, ``'success'``, ``'info'``, ``'hint'``, ``'debug'``)."""

    n_jobs: int = -1
    """Default number of jobs / CPUs for parallel computing. ``-1`` uses all available cores."""

    @field_validator("verbosity")
    @classmethod
    def _propagate_verbosity_to_logger(cls, value: _VerbosityName) -> _VerbosityName:
        logger.set_verbosity(_VERBOSITY_NAME_TO_INT[value])
        return value


settings = _EhrapySettings()

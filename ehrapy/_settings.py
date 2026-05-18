from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from ehrdata._logger import logger
from pydantic import Field, field_validator
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

    plot_suffix: str = ""
    """Global suffix appended to figure filenames."""

    file_format_data: Literal["csv", "h5ad"] = "h5ad"
    """File format for saving EHRData objects."""

    file_format_figs: str = "pdf"
    """Default file format for saving figures (e.g. ``'png'``, ``'pdf'``, ``'svg'``)."""

    autosave: bool = False
    """Automatically save figures to ``figdir`` instead of displaying them."""

    autoshow: bool = True
    """Automatically show figures when ``autosave`` is ``False``."""

    writedir: Path = Path("./ehrapy_write/")
    """Default directory for :func:`scanpy.write`."""

    cachedir: Path = Path("./ehrapy_cache/")
    """Directory for cache files."""

    datasetdir: Path = Path("./ehrapy_data/")
    """Directory for downloaded example datasets."""

    figdir: Path = Path("./figures/")
    """Directory for saving figures."""

    cache_compression: Literal["lzf", "gzip"] | None = "lzf"
    """Compression for cached reads."""

    max_memory: Annotated[float, Field(gt=0)] = 15
    """Maximum memory usage in Gigabytes (advisory)."""

    n_jobs: int = -1
    """Default number of jobs / CPUs for parallel computing. ``-1`` uses all available cores."""

    categories_to_ignore: list[str] = Field(default_factory=lambda: ["N/A", "dontknow", "no_gate", "?"])
    """Category labels omitted in plotting."""

    n_pcs: Annotated[int, Field(gt=0)] = 50
    """Default number of principal components."""

    @field_validator("verbosity")
    @classmethod
    def _propagate_verbosity_to_logger(cls, value: _VerbosityName) -> _VerbosityName:
        logger.set_verbosity(_VERBOSITY_NAME_TO_INT[value])
        return value


settings = _EhrapySettings()

from __future__ import annotations

import inspect
import logging
import sys
from contextlib import contextmanager
from enum import IntEnum
from logging import getLevelName
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Literal, TextIO

from matplotlib import pyplot as plt
from scanpy.plotting import set_rcParams_scanpy

from ehrapy.logging import _RootLogger, _set_log_file, _set_log_level

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

_VERBOSITY_TO_LOGLEVEL: dict[str, str] = {
    "error": "ERROR",
    "warning": "WARNING",
    "info": "INFO",
    "hint": "HINT",
    "debug": "DEBUG",
}
# Python 3.7 ensures iteration order
for index, level in enumerate(list(_VERBOSITY_TO_LOGLEVEL.values())):  # pragma: no cover
    _VERBOSITY_TO_LOGLEVEL[index] = level  # type: ignore


class Verbosity(IntEnum):  # pragma: no cover
    error = 0
    warn = 1
    info = 2
    hint = 3
    debug = 4

    @property
    def level(self) -> int:
        # getLevelName(str) returns the int level…
        return getLevelName(_VERBOSITY_TO_LOGLEVEL[self])  # type: ignore

    @contextmanager  # type: ignore
    def override(self, verbosity: Verbosity) -> Generator:
        """Temporarily override verbosity."""
        ehrapy_settings.verbosity = verbosity
        yield self
        ehrapy_settings.verbosity = self


def _type_check(var: Any, varname: str, types: type | tuple[type, ...]):  # pragma: no cover
    if isinstance(var, types):
        return
    if isinstance(types, type):
        possible_types_str = types.__name__
    else:
        type_names = [t.__name__ for t in types]
        possible_types_str = "{} or {}".format(", ".join(type_names[:-1]), type_names[-1])
    raise TypeError(f"{varname} must be of type {possible_types_str}")


class EhrapyConfig:  # pragma: no cover
    """Configuration manager for ehrapy.

    Strongly adapted from Scanpy.
    """

    def __init__(
        self,
        *,
        verbosity: str = "warning",
        plot_suffix: str = "",
        file_format_data: str = "h5ad",
        file_format_figs: str = "pdf",
        autosave: bool = False,
        autoshow: bool = True,
        writedir: str | Path = "./ehrapy_write/",
        cachedir: str | Path = "./ehrapy_cache/",
        datasetdir: str | Path = "./ehrapy_data/",
        figdir: str | Path = "./figures/",
        cache_compression: str | None = "lzf",
        max_memory=15,
        n_jobs: int = 1,
        logfile: str | Path | None = None,
        categories_to_ignore: Iterable[str] = ("N/A", "dontknow", "no_gate", "?"),
        _frameon: bool = True,
        _vector_friendly: bool = False,
        _low_resolution_warning: bool = True,
        n_pcs=50,
    ):
        # logging
        self._root_logger = _RootLogger(logging.INFO)  # level will be replaced
        self.logfile = logfile  # type: ignore
        self.verbosity = verbosity  # type: ignore
        # rest
        self.plot_suffix = plot_suffix
        self.file_format_data = file_format_data
        self.file_format_figs = file_format_figs
        self.autosave = autosave
        self.autoshow = autoshow
        self.writedir = writedir  # type: ignore
        self.cachedir = cachedir  # type: ignore
        self.datasetdir = datasetdir  # type: ignore
        self.figdir = figdir  # type: ignore
        self.cache_compression = cache_compression
        self.max_memory = max_memory
        self.n_jobs = n_jobs
        self.categories_to_ignore = categories_to_ignore  # type: ignore
        self._frameon = _frameon
        """bool: See set_figure_params."""

        self._vector_friendly = _vector_friendly
        """Set to true if you want to include pngs in svgs and pdfs."""

        self._low_resolution_warning = _low_resolution_warning
        """Print warning when saving a figure with low resolution."""

        self._start = time()
        """Time when the settings module is first imported."""

        self._previous_time = self._start
        """Variable for timing program parts."""

        self._previous_memory_usage = -1
        """Stores the previous memory usage."""

        self.N_PCS = n_pcs
        """Default number of principal components to use."""

    @property
    def verbosity(self) -> Verbosity:
        """
        Verbosity level (default `warning`)

        Level 0: only show 'error' messages.
        Level 1: also show 'warning' messages.
        Level 2: also show 'info' messages.
        Level 3: also show 'hint' messages.
        Level 4: also show very detailed progress for 'debug'ging.
        """
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity: Verbosity | int | str):
        verbosity_str_options = [v for v in _VERBOSITY_TO_LOGLEVEL if isinstance(v, str)]
        if isinstance(verbosity, Verbosity):
            self._verbosity = verbosity
        elif isinstance(verbosity, int):
            self._verbosity = Verbosity(verbosity)
        elif isinstance(verbosity, str):
            verbosity = verbosity.lower()
            if verbosity not in verbosity_str_options:
                raise ValueError(
                    f"Cannot set verbosity to {verbosity}. " f"Accepted string values are: {verbosity_str_options}"
                )
            else:
                self._verbosity = Verbosity(verbosity_str_options.index(verbosity))
        else:
            _type_check(verbosity, "verbosity", (str, int))
        _set_log_level(self, _VERBOSITY_TO_LOGLEVEL[self._verbosity])  # type: ignore

    @property
    def plot_suffix(self) -> str:
        """Global suffix that is appended to figure filenames."""
        return self._plot_suffix

    @plot_suffix.setter
    def plot_suffix(self, plot_suffix: str):
        _type_check(plot_suffix, "plot_suffix", str)
        self._plot_suffix = plot_suffix

    @property
    def file_format_data(self) -> str:
        """File format for saving AnnData objects.

        Allowed are 'txt', 'csv' (comma separated value file) for exporting and 'h5ad' (hdf5) for lossless saving.
        """
        return self._file_format_data

    @file_format_data.setter
    def file_format_data(self, file_format: str):
        _type_check(file_format, "file_format_data", str)
        file_format_options = {"csv", "h5ad"}
        if file_format not in file_format_options:
            raise ValueError(f"Cannot set file_format_data to {file_format}. " f"Must be one of {file_format_options}")
        self._file_format_data = file_format

    @property
    def file_format_figs(self) -> str:
        """File format for saving figures.

        For example 'png', 'pdf' or 'svg'. Many other formats work as well (see `matplotlib.pyplot.savefig`).
        """
        return self._file_format_figs

    @file_format_figs.setter
    def file_format_figs(self, figure_format: str):
        _type_check(figure_format, "figure_format_data", str)
        self._file_format_figs = figure_format

    @property
    def autosave(self) -> bool:
        """Automatically save figures in :attr:`~scanpy._settings.ScanpyConfig.figdir` (default `False`).

        Do not show plots/figures interactively.
        """
        return self._autosave

    @autosave.setter
    def autosave(self, autosave: bool):
        _type_check(autosave, "autosave", bool)
        self._autosave = autosave

    @property
    def autoshow(self) -> bool:
        """Automatically show figures if `autosave == False` (default `True`).

        There is no need to call the matplotlib pl.show() in this case.
        """
        return self._autoshow

    @autoshow.setter
    def autoshow(self, autoshow: bool):
        _type_check(autoshow, "autoshow", bool)
        self._autoshow = autoshow

    @property
    def writedir(self) -> Path:
        """Directory where the function scanpy.write writes to by default."""
        return self._writedir

    @writedir.setter
    def writedir(self, writedir: str | Path):
        _type_check(writedir, "writedir", (str, Path))
        self._writedir = Path(writedir)

    @property
    def cachedir(self) -> Path:
        """Directory for cache files (default `'./cache/'`)."""
        return self._cachedir

    @cachedir.setter
    def cachedir(self, cachedir: str | Path):
        _type_check(cachedir, "cachedir", (str, Path))
        self._cachedir = Path(cachedir)

    @property
    def datasetdir(self) -> Path:
        """Directory for example :mod:`~scanpy.datasets` (default `'./data/'`)."""
        return self._datasetdir

    @datasetdir.setter
    def datasetdir(self, datasetdir: str | Path):
        _type_check(datasetdir, "datasetdir", (str, Path))
        self._datasetdir = Path(datasetdir).resolve()

    @property
    def figdir(self) -> Path:
        """Directory for saving figures (default `'./figures/'`)."""
        return self._figdir

    @figdir.setter
    def figdir(self, figdir: str | Path):
        _type_check(figdir, "figdir", (str, Path))
        self._figdir = Path(figdir)

    @property
    def cache_compression(self) -> str | None:
        """Compression for `sc.read(..., cache=True)` (default `'lzf'`).

        May be `'lzf'`, `'gzip'`, or `None`.
        """
        return self._cache_compression

    @cache_compression.setter
    def cache_compression(self, cache_compression: str | None):
        if cache_compression not in {"lzf", "gzip", None}:
            raise ValueError(f"`cache_compression` ({cache_compression}) " "must be in {'lzf', 'gzip', None}")
        self._cache_compression = cache_compression

    @property
    def max_memory(self) -> int | float:
        """Maximal memory usage in Gigabyte.

        Is currently not well respected....
        """
        return self._max_memory

    @max_memory.setter
    def max_memory(self, max_memory: int | float):
        _type_check(max_memory, "max_memory", (int, float))
        self._max_memory = max_memory

    @property
    def n_jobs(self) -> int:
        """Default number of jobs/ CPUs to use for parallel computing."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int):
        _type_check(n_jobs, "n_jobs", int)
        self._n_jobs = n_jobs

    @property
    def logpath(self) -> Path | None:
        """The file path `logfile` was set to."""
        return self._logpath  # type: ignore

    @logpath.setter
    def logpath(self, logpath: str | Path | None):
        _type_check(logpath, "logfile", (str, Path))
        # set via “file object” branch of logfile.setter
        self.logfile = Path(logpath).open("a")
        self._logpath = Path(logpath)

    @property
    def logfile(self) -> TextIO:
        """The open file to write logs to.

        Set it to a :class:`~pathlib.Path` or :class:`str` to open a new one.
        The default `None` corresponds to :obj:`sys.stdout` in jupyter notebooks
        and to :obj:`sys.stderr` otherwise.

        For backwards compatibility, setting it to `''` behaves like setting it to `None`.
        """
        return self._logfile  # type: ignore

    @logfile.setter
    def logfile(self, logfile: str | Path | TextIO | None):
        if not hasattr(logfile, "write") and logfile:
            self.logpath = logfile  # type: ignore
        else:  # file object
            if not logfile:  # None or ''
                logfile = sys.stdout if self._is_run_from_ipython() else sys.stderr
            self._logfile = logfile
            self._logpath = None
            _set_log_file(self)

    @property
    def categories_to_ignore(self) -> list[str]:
        """Categories that are omitted in plotting etc."""
        return self._categories_to_ignore

    @categories_to_ignore.setter
    def categories_to_ignore(self, categories_to_ignore: Iterable[str]):
        categories_to_ignore = list(categories_to_ignore)
        for i, cat in enumerate(categories_to_ignore):
            _type_check(cat, f"categories_to_ignore[{i}]", str)
        self._categories_to_ignore = categories_to_ignore

    # --------------------------------------------------------------------------------
    # Functions
    # --------------------------------------------------------------------------------

    # Collected from the print_* functions in matplotlib.backends
    # fmt: off
    _Format = Literal[
        'png', 'jpg', 'tif', 'tiff',
        'pdf', 'ps', 'eps', 'svg', 'svgz', 'pgf',
        'raw', 'rgba',
    ]

    # fmt: on

    def set_figure_params(
        self,
        scanpy: bool = True,
        dpi: int = 80,
        dpi_save: int = 150,
        frameon: bool = True,
        vector_friendly: bool = True,
        fontsize: int = 14,
        figsize: int | None = None,
        color_map: str | None = None,
        format: _Format = "pdf",
        facecolor: str | None = None,
        transparent: bool = False,
        ipython_format: str = "png2x",
        dark: bool = False,
    ):
        """Set resolution/size, styling and format of figures.

        Args:
            scanpy: Init default values for :obj:`matplotlib.rcParams` based on Scanpy's.
            dpi: Resolution of rendered figures – this influences the size of figures in notebooks.
            dpi_save: Resolution of saved figures. This should typically be higher to achieve publication quality.
            frameon: Add frames and axes labels to scatter plots.
            vector_friendly: Plot scatter plots using `png` backend even when exporting as `pdf` or `svg`.
            fontsize: Set the fontsize for several `rcParams` entries. Ignored if `scanpy=False`.
            figsize: Set plt.rcParams['figure.figsize'].
            color_map: Convenience method for setting the default color map. Ignored if `scanpy=False`.
            format: This sets the default format for saving figures: `file_format_figs`.
            facecolor: Sets backgrounds via `rcParams['figure.facecolor'] = facecolor` and `rcParams['axes.facecolor'] = facecolor`.
            transparent: Save figures with transparent back ground. Sets `rcParams['savefig.transparent']`.
            ipython_format: Only concerns the notebook/IPython environment; see :func:`~IPython.display.set_matplotlib_formats` for details.
            dark: Whether to enable Matplotlibs dark styled. Inverts all colors.
        """
        if self._is_run_from_ipython():
            import IPython

            if isinstance(ipython_format, str):
                ipython_format = [ipython_format]  # type: ignore
            IPython.display.set_matplotlib_formats(*ipython_format)

        from matplotlib import rcParams

        self._vector_friendly = vector_friendly
        self.file_format_figs = format
        if dpi is not None:
            rcParams["figure.dpi"] = dpi
        if dpi_save is not None:
            rcParams["savefig.dpi"] = dpi_save
        if transparent is not None:
            rcParams["savefig.transparent"] = transparent
        if facecolor is not None:
            rcParams["figure.facecolor"] = facecolor
            rcParams["axes.facecolor"] = facecolor
        if scanpy:
            set_rcParams_scanpy(fontsize=fontsize, color_map=color_map)
        if figsize is not None:
            rcParams["figure.figsize"] = figsize
        if dark:
            plt.style.use("dark_background")
        self._frameon = frameon

    @staticmethod
    def _is_run_from_ipython() -> bool:
        """Determines whether we are currently in IPython."""
        import builtins

        return getattr(builtins, "__IPYTHON__", False)

    def __str__(self) -> str:
        return "\n".join(
            f"{k} = {v!r}" for k, v in inspect.getmembers(self) if not k.startswith("_") and not k == "getdoc"
        )


ehrapy_settings = EhrapyConfig()

ehrapy_settings._root_logger = _RootLogger(ehrapy_settings.verbosity)
_set_log_file(ehrapy_settings)
_set_log_level(ehrapy_settings, level=20)  # default level info

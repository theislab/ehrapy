from __future__ import annotations

import inspect
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Literal

from ehrdata._logger import logger
from matplotlib import pyplot as plt
from scanpy.plotting import set_rcParams_scanpy

if TYPE_CHECKING:
    from collections.abc import Iterable

VERBOSITY_TO_INT = {
    "error": 0,  # 40
    "warning": 1,  # 30
    "success": 2,  # 25
    "info": 3,  # 20
    "hint": 4,  # 15
    "debug": 5,  # 10
}
VERBOSITY_TO_STR: dict[int, str] = dict(
    [reversed(i) for i in VERBOSITY_TO_INT.items()]  # type: ignore
)

# Collected from the print_* functions in matplotlib.backends
# fmt: off
_Format = Literal[
    'png', 'jpg', 'tif', 'tiff',
    'pdf', 'ps', 'eps', 'svg', 'svgz', 'pgf',
    'raw', 'rgba',
]
# fmt: on


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
    """Configuration manager for ehrapy."""

    def __init__(
        self,
        *,
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
        n_jobs: int = -1,
        categories_to_ignore: Iterable[str] = ("N/A", "dontknow", "no_gate", "?"),
        _frameon: bool = True,
        _vector_friendly: bool = False,
        _low_resolution_warning: bool = True,
        n_pcs=50,
    ):
        # logging
        self._verbosity_int: int = 1  # warning-level logging
        logger.set_verbosity(self._verbosity_int)
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
    def verbosity(self) -> str:
        """Logger verbosity (default 'warning').

        - 'error': ❌ only show error messages
        - 'warning': ❗ also show warning messages
        - 'success': ✅ also show success and save messages
        - 'info': 💡 also show info messages
        - 'hint': 💡 also show hint messages
        - 'debug': 🐛 also show detailed debug messages
        """
        return VERBOSITY_TO_STR[self._verbosity_int]

    @verbosity.setter
    def verbosity(self, verbosity: str | int):
        if isinstance(verbosity, str):
            verbosity_int = VERBOSITY_TO_INT[verbosity]
        else:
            verbosity_int = verbosity
        self._verbosity_int = verbosity_int
        logger.set_verbosity(verbosity_int)

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
            raise ValueError(f"Cannot set file_format_data to {file_format}. Must be one of {file_format_options}")
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
        """Automatically save figures to the default fig dir.

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
            raise ValueError(f"`cache_compression` ({cache_compression}) must be in {{'lzf', 'gzip', None}}")
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
            if isinstance(ipython_format, str):
                ipython_format = [ipython_format]  # type: ignore
            from matplotlib_inline.backend_inline import set_matplotlib_formats

            set_matplotlib_formats(*ipython_format)

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

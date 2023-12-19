from __future__ import annotations

import sys
from datetime import datetime

import session_info
from rich import print

from ehrapy import __version__
from ehrapy.logging import _versions_dependencies


def print_versions():  # pragma: no cover
    """Print print versions of imported packages.

    Examples:
        >>> import ehrapy as ep
        >>> ep.print_versions()
    """
    try:
        session_info.show(
            dependencies=True,
            html=False,
            excludes=[
                "builtins",
                "stdlib_list",
                "importlib_metadata",
                "jupyter_core"
                # Special module present if test coverage being calculated
                # https://gitlab.com/joelostblom/session_info/-/issues/10
                "$coverage",
            ],
        )
    except AttributeError:
        print("[bold yellow]Unable to fetch versions for one or more dependencies.")
        pass


def print_version_and_date(*, file=None):  # pragma: no cover
    """Useful for starting a notebook so you see when you started working."""
    if file is None:
        file = sys.stdout
    print(
        f"Running ehrapy {__version__}, " f"on {datetime.now():%Y-%m-%d %H:%M}.",
        file=file,
    )


def print_header(*, file=None):  # pragma: no cover
    """Versions that might influence the numerical results.

    Matplotlib and Seaborn are excluded from this.
    """
    _DEPENDENCIES_NUMERICS = [
        "scanpy",
        "anndata",
        "umap",
        "numpy",
        "scipy",
        "pandas",
        ("sklearn", "scikit-learn"),
        "statsmodels",
        ("igraph", "python-igraph"),
        "leidenalg",
        "pynndescent",
    ]

    modules = ["ehrapy"] + _DEPENDENCIES_NUMERICS
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=file or sys.stdout,
    )

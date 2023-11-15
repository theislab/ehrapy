from __future__ import annotations

import sys
from contextlib import closing
from datetime import datetime
from io import StringIO

import session_info
from IPython.utils.io import Tee
from rich import print

from ehrapy import __version__
from ehrapy.logging import _versions_dependencies


def print_versions(*, output_file=None) -> None:  # pragma: no cover
    """Print print versions of imported packages.

    Args:
        output_file: Path to output file

    Examples:
        >>> import ehrapy as ep
        >>> ep.print_versions()
    """
    stdout = sys.stdout
    try:
        buf = sys.stdout = StringIO()
        session_info.show(
            dependencies=True,
            excludes=[
                "builtins",
                "stdlib_list",
                "importlib_metadata",
                # Special module present if test coverage being calculated
                # https://gitlab.com/joelostblom/session_info/-/issues/10
                "transformers",
                "$coverage",
            ],
            write_req_file=False,
            html=False,
        )
    finally:
        sys.stdout = stdout
    output = buf.getvalue()

    if output_file:
        with closing(Tee(output_file, "w", channel="stdout")):
            print(output)
    else:
        print(output)


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

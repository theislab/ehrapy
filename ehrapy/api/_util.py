from __future__ import annotations

import importlib
import sys
from contextlib import closing
from datetime import datetime
from io import StringIO
from subprocess import PIPE, Popen

from IPython.utils.io import Tee
from rich import print
from scanpy.logging import _versions_dependencies
from sinfo import sinfo

from ehrapy import __version__


def print_versions(*, output_file=None) -> None:  # pragma: no cover
    """Print print versions of imported packages.

    Args:
        output_file: Path to output file
    """
    stdout = sys.stdout
    try:
        buf = sys.stdout = StringIO()
        sinfo(
            dependencies=True,
            excludes=[
                "builtins",
                "stdlib_list",
                "importlib_metadata",
                # Special module present if test coverage being calculated
                # https://gitlab.com/joelostblom/sinfo/-/issues/10
                "$coverage",
            ],
            write_req_file=False,
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
        "anndata",  # anndata actually shouldn't, but as long as it's in development
        "umap",
        "numpy",
        "scipy",
        "pandas",
        ("sklearn", "scikit-learn"),
        "statsmodels",
        ("igraph", "python-igraph"),
        "louvain",
        "leidenalg",
        "pynndescent",
    ]

    modules = ["ehrapy"] + _DEPENDENCIES_NUMERICS
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=file or sys.stdout,
    )


def check_module_importable(package: str) -> bool:  # pragma: no cover
    """Checks whether a module is installed and can be loaded.

    Args:
        package: The package to check.

    Returns:
        True if the package is installed, false elsewise
    """
    module_information = importlib.util.find_spec(package)
    module_available = module_information is not None

    return module_available


def shell_command_accessible(command: list[str]) -> bool:  # pragma: no cover
    """Checks whether the provided command is accessible in the current shell.

    Args:
        command: The command to check. Spaces are separated as list elements.

    Returns:
        True if the command is accessible, false otherwise.
    """
    command_accessible = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    (commmand_stdout, command_stderr) = command_accessible.communicate()
    if command_accessible.returncode != 0:
        return False

    return True

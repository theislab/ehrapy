import importlib
import sys
from datetime import datetime
from io import StringIO

from rich import print
from scanpy.logging import _versions_dependencies
from sinfo import sinfo

from ehrapy import __version__


def print_versions(*, file=None):
    """Print print versions of imported packages."""
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
        )
    finally:
        sys.stdout = stdout
    output = buf.getvalue()
    print(output, file=file)


def print_version_and_date(*, file=None):
    """Useful for starting a notebook so you see when you started working."""
    if file is None:
        file = sys.stdout
    print(
        f"Running ehrapy {__version__}, " f"on {datetime.now():%Y-%m-%d %H:%M}.",
        file=file,
    )


def print_header(*, file=None):
    """Versions that might influence the numerical results.

    Matplotlib and Seaborn are excluded from this.
    """
    _DEPENDENCIES_NUMERICS = [
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

    modules = ["scanpy"] + _DEPENDENCIES_NUMERICS
    print(
        " ".join(f"{mod}=={ver}" for mod, ver in _versions_dependencies(modules)),
        file=file or sys.stdout,
    )


def check_module_importable(package: str) -> bool:
    """Checks whether a module is installed and can be loaded.

    Args:
        package: The package to check.

    Returns:
        True if the package is installed, false elsewise
    """
    module_information = importlib.util.find_spec(package)
    module_available = module_information is not None

    return module_available

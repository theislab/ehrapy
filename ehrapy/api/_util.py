import importlib
import sys
from io import StringIO

from rich import print
from sinfo import sinfo


def print_versions(*, file=None):
    """Print print versions of imported packages"""
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

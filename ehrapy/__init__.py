"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.15.0"

import os

# https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
os.environ["SCIPY_ARRAY_API"] = "1"

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence '\\")

from ehrapy import get
from ehrapy import plot as pl
from ehrapy import preprocessing as pp
from ehrapy import tools as tl
from ehrapy._settings import settings
from ehrapy.core.meta_information import print_versions


def __getattr__(name: str):
    """Lazy-load optional layers such as ``ehrapy.mcp``."""
    if name == "mcp":
        import importlib

        return importlib.import_module("ehrapy.mcp")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

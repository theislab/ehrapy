import io
import sys

from rich import print
from sinfo import sinfo

from ehrapy.api._settings import EhrapyConfig, ehrapy_settings

settings: EhrapyConfig = ehrapy_settings

from ehrapy.api import data as dt
from ehrapy.api import encode as ec
from ehrapy.api import plot as pl
from ehrapy.api import preprocessing as pp
from ehrapy.api import tools as tl


def print_versions(*, file=None):
    """Print print versions of imported packages"""
    stdout = sys.stdout
    try:
        buf = sys.stdout = io.StringIO()
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

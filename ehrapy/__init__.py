"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"

from pypi_latest import PypiLatest
from rich import traceback

ehrapy_pypi_latest = PypiLatest("ehrapy", __version__)
ehrapy_pypi_latest.check_latest()

traceback.install(width=200, word_wrap=True)

from ehrapy.api import data, io, plot, preprocessing, tools

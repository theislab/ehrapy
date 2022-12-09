"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.4.0"

from pypi_latest import PypiLatest
from rich import traceback

ehrapy_pypi_latest = PypiLatest("ehrapy", __version__)
ehrapy_pypi_latest.check_latest()

traceback.install(width=200, word_wrap=True)

from ehrapy._settings import EhrapyConfig, ehrapy_settings
from ehrapy.core.meta_information import print_versions

settings: EhrapyConfig = ehrapy_settings

from ehrapy import anndata as ad
from ehrapy import data as dt
from ehrapy import io
from ehrapy import plot as pl
from ehrapy import preprocessing as pp
from ehrapy import tools as tl

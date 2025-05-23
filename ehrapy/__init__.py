"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.13.0"

import os

# https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
os.environ["SCIPY_ARRAY_API"] = "1"

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, message=r"invalid escape sequence '\\")

from ehrapy._settings import EhrapyConfig, ehrapy_settings

settings: EhrapyConfig = ehrapy_settings

from ehrapy import anndata as ad
from ehrapy import data as dt
from ehrapy import get, io
from ehrapy import plot as pl
from ehrapy import preprocessing as pp
from ehrapy import tools as tl
from ehrapy.core.meta_information import print_versions

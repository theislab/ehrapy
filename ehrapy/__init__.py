"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.14.0"

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

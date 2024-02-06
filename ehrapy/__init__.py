"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.7.0"

from ehrapy._settings import EhrapyConfig, ehrapy_settings

settings: EhrapyConfig = ehrapy_settings

from ehrapy import anndata as ad
from ehrapy import data as dt
from ehrapy import io
from ehrapy import plot as pl
from ehrapy import preprocessing as pp
from ehrapy import tools as tl
from ehrapy.core.meta_information import print_versions

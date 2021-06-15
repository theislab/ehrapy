"""Top-level package for ehrapy."""

__author__ = "Lukas Heumos"
__email__ = "lukas.heumos@posteo.net"
__version__ = "0.1.0"

from ehrapy.api import plot, preprocessing, tools
from ehrapy.cli.upgrade import UpgradeCommand

UpgradeCommand.check_ehrapy_latest()

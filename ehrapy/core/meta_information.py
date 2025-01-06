from __future__ import annotations

import sys
from datetime import datetime

from rich import print
from scanpy.logging import _versions_dependencies

from ehrapy import __version__


def print_versions():  # pragma: no cover
    """Print versions of imported packages.

    Examples:
        >>> import ehrapy as ep
        >>> ep.print_versions()
    """
    print_header()


def print_version_and_date(*, file=None):  # pragma: no cover
    """Useful for starting a notebook so you see when you started working."""
    if file is None:
        file = sys.stdout
    print(
        f"Running ehrapy {__version__}, " f"on {datetime.now():%Y-%m-%d %H:%M}.",
        file=file,
    )


def print_header(*, file=None):  # pragma: no cover
    """Versions that might influence the numerical results."""
    from session_info2 import session_info

    sinfo = session_info(os=True, cpu=True, gpu=True, dependencies=True)

    if file is not None:
        print(sinfo, file=file)
        return

    return sinfo

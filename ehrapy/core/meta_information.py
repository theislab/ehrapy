from __future__ import annotations

import sys
from datetime import datetime

from rich import print

from ehrapy import __version__


def print_versions(*, file=None):  # pragma: no cover
    """Print versions of imported packages.

    Examples:
        >>> import ehrapy as ep
        >>> ep.print_versions()
    """
    from session_info2 import session_info

    sinfo = session_info(os=True, cpu=True, gpu=True, dependencies=True)

    if file is not None:
        print(sinfo, file=file)
        return

    print(sinfo)

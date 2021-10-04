from collections import Generator
from contextlib import contextmanager


@contextmanager
def verbosity(level: int) -> Generator[None, None, None]:
    """Temporarily set the verbosity level of :mod:`scanpy`.

    Verbosity level (default `warning`)
    Level 0: only show 'error' messages.
    Level 1: also show 'warning' messages.
    Level 2: also show 'info' messages.
    Level 3: also show 'hint' messages.
    Level 4: also show very detailed progress for 'debug'ging.

    Args:
        level: The new verbosity level
    """
    import scanpy as sc

    current_verbosity = sc.settings.verbosity
    sc.settings.verbosity = level
    try:
        yield
    finally:
        sc.settings.verbosity = current_verbosity

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import holoviews as hv
import matplotlib as mpl
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")


def load_hv_extensions() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Load the bokeh and matplotlib holoviews extensions when a holoviews-backed plot function is first called."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if missing := [backend for backend in ("bokeh", "matplotlib") if backend not in hv.Store.renderers]:
                # hv.extension switches to its first backend, so restore one the user already selected
                current_backend = hv.Store.current_backend if hv.Store.renderers else "bokeh"
                mpl_backend = mpl.get_backend()
                # the inline backend registers its IPython display hooks on import only while rcParams names it
                plt.switch_backend(mpl_backend)
                hv.extension(*missing)
                hv.Store.set_current_backend(current_backend)
                plt.switch_backend(mpl_backend)
            return func(*args, **kwargs)

        return wrapper

    return decorator

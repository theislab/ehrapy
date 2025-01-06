# Since we might check whether an object is an instance of dask.array.Array
# without requiring dask installed in the environment.
from collections.abc import Callable

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def _raise_array_type_not_implemented(func: Callable, type_: type) -> NotImplementedError:
    return NotImplementedError(
        f"{func.__name__} does not support array type {type_}. Must be of type {func.registry.keys()}."  # type: ignore
    )


def is_dask_array(array):
    if DASK_AVAILABLE:
        return isinstance(array, da.Array)
    else:
        return False

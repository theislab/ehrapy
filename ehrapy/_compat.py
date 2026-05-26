from __future__ import annotations

import warnings
from functools import wraps
from importlib.util import find_spec
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import holoviews as hv
import numpy as np
import scipy.sparse as sp

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

if TYPE_CHECKING:
    # type checkers are confused and can only see …core.Array
    from dask.array.core import Array as DaskArray
elif find_spec("dask"):
    from dask.array import Array as DaskArray
else:
    DaskArray = type("Array", (), {})
    DaskArray.__module__ = "dask.array"

if TYPE_CHECKING:
    from collections.abc import Callable

    from ehrdata import EHRData


def _raise_array_type_not_implemented(func: Callable, type_: type) -> NotImplementedError:
    raise NotImplementedError(
        f"{func.__name__} does not support array type {type_}. Must be of type {func.registry.keys()}."  # type: ignore
    )


def _shell_command_accessible(command: list[str]) -> bool:
    """Checks whether the provided command is accessible in the current shell.

    Args:
        command: The command to check. Spaces are separated as list elements.

    Returns:
        True if the command is accessible, False otherwise.
    """
    command_accessible = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    command_accessible.communicate()
    if command_accessible.returncode != 0:
        return False

    return True


def _apply_over_time_axis(f: Callable) -> Callable:
    """Decorator to allow functions to handle both 2D and 3D arrays.

    - If the input is 2D: pass it through unchanged.
    - If the input is 3D: reshape to 2D before calling the function, then reshape the result back to 3D.
    """

    @wraps(f)
    def wrapper(arr, *args, **kwargs):
        if arr.ndim == 2:
            return f(arr, *args, **kwargs)

        elif arr.ndim == 3:
            n_obs, n_vars, n_time = arr.shape
            arr_2d = np.moveaxis(arr, 1, 2).reshape(-1, n_vars)
            arr_modified_2d = f(arr_2d, *args, **kwargs)
            return np.moveaxis(arr_modified_2d.reshape(n_obs, n_time, n_vars), 1, 2)

        else:
            raise ValueError(f"Unsupported array dimensionality: {arr.ndim}. Please reshape the array to 2D or 3D.")

    return wrapper


def function_future_warning(old_function_name: str, new_function_name: str | None = None):
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            warn_msg = f"{old_function_name} is deprecated, and will be removed in v1.0.0."
            if new_function_name:
                warn_msg += f" Use {new_function_name} instead."
            warnings.warn(warn_msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def function_2D_only():
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            data: EHRData | None
            if args and len(args) >= 1:
                data = args[0]
            elif kwargs:
                data = kwargs.get("edata")

            layer = kwargs.get("layer")
            use_rep = kwargs.get("use_rep")

            if data is not None:
                array = data.X if layer is None else data.layers[layer]
                if use_rep is not None:
                    array = data.obsm[use_rep]

                if array.ndim != 2 and array.shape[2] != 1:
                    raise ValueError(
                        f"{func.__name__}() only supports 2D data, got {'data.X' if layer is None else f'data.layers[{layer}]'} with shape {array.shape}"
                    )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def as_dense_dask_array(a, chunk_size=1000):
    """Convert input to a dense Dask array."""
    import dask.array as da

    return da.from_array(a, chunks=chunk_size)


def choose_hv_backend() -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if hv.Store.current_backend is None:
                raise RuntimeError(
                    "No holoviews backend selected. "
                    "Call holoviews.extension('matplotlib') or "
                    "holoviews.extension('bokeh') before using this function."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def nanmean_array_api(xp, arr, axes):
    """Compute mean ignoring NaN values using Array API operations."""
    mask = xp.isnan(arr)
    zero_filled = xp.where(mask, xp.zeros_like(arr), arr)
    count = xp.sum(xp.astype(~mask, arr.dtype), axis=axes)
    return xp.sum(zero_filled, axis=axes) / count


def nanmedian_array_api(xp, arr):
    """Compute per-feature median ignoring NaN values using Array API operations.

    Computes the median for each feature across all patients and time steps
    for a 3D array of shape ``(n_obs, n_vars, n_time)``.
    """
    if arr.ndim == 2:
        arr = xp.reshape(arr, (arr.shape[0], arr.shape[1], 1))

    n_obs, n_vars, n_time = arr.shape
    arr_flat = xp.reshape(xp.permute_dims(arr, (1, 0, 2)), (n_vars, -1))
    medians = []
    for i in range(n_vars):
        row = arr_flat[i, :]
        not_nan = ~xp.isnan(row)
        n = int(xp.sum(xp.astype(not_nan, xp.float64)))
        if n == 0:
            medians.append(float("nan"))
            continue
        filled = xp.where(not_nan, row, xp.asarray(float("inf"), dtype=arr.dtype))
        sorted_row = xp.sort(filled)
        if n % 2 == 1:
            medians.append(float(sorted_row[n // 2]))
        else:
            medians.append(float((sorted_row[n // 2 - 1] + sorted_row[n // 2]) / 2))

    return xp.asarray(medians, dtype=arr.dtype)


def nanstd_array_api(xp, arr, axes):
    """Compute standard deviation ignoring NaN values using Array API operations."""
    nan_mask = xp.isnan(arr)
    mean = nanmean_array_api(xp, arr, axes=axes)
    # expand mean dims to broadcast against arr (assumes axes is an int or 0)
    diff = xp.where(nan_mask, xp.zeros_like(arr), arr - xp.expand_dims(mean, axis=axes))
    count = xp.sum(xp.astype(~nan_mask, arr.dtype), axis=axes)
    return xp.sqrt(xp.sum(diff**2, axis=axes) / count)


def nanmin_array_api(xp, arr, axis):
    """Compute min ignoring NaN values using Array API operations.

    Returns NaN for slices where all values are NaN.
    """
    nan_mask = xp.isnan(arr)

    # Replace NaNs with +inf so they don't affect min
    arr_for_min = xp.where(nan_mask, xp.full_like(arr, xp.inf), arr)
    minv = xp.min(arr_for_min, axis=axis)

    # Count non NaN entries per slice
    count = xp.sum(xp.astype(~nan_mask, xp.int64), axis=axis)
    nan_scalar = xp.asarray(float("nan"), dtype=arr.dtype)

    return xp.where(count == 0, nan_scalar, minv)


def nanmax_array_api(xp, arr, axis):
    """Compute max ignoring NaN values using Array API operations.

    Returns NaN for slices where all values are NaN.
    """
    nan_mask = xp.isnan(arr)

    # Replace NaNs with -inf so they don't affect max
    arr_for_max = xp.where(nan_mask, xp.full_like(arr, -xp.inf), arr)
    maxv = xp.max(arr_for_max, axis=axis)

    # Count non NaN entries per slice
    count = xp.sum(xp.astype(~nan_mask, xp.int64), axis=axis)
    nan_scalar = xp.asarray(float("nan"), dtype=arr.dtype)

    return xp.where(count == 0, nan_scalar, maxv)

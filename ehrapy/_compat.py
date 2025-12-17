from __future__ import annotations

import warnings
from functools import wraps
from importlib.util import find_spec
from inspect import signature
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, ParamSpec, TypeVar, cast

import holoviews as hv
import numpy as np
import scipy.sparse as sp

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

from anndata import AnnData
from ehrdata import EHRData

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


def use_ehrdata(
    deprecated_after: str | None = None,
    old_param: str = "adata",
    new_param: str = "edata",
    edata_None_allowed: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to migrate functions from AnnData to EHRData."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        sig = signature(func)

        has_new_param = new_param in sig.parameters
        has_old_param = old_param in sig.parameters

        if not has_new_param and not has_old_param:
            raise ValueError(f"Function {func.__name__} does not have parameter {old_param} or {new_param}")

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Check if we're passing data as first positional argument
            if args and len(args) >= 1:
                data = args[0]
                new_args = args[1:]

                # Issue warning if AnnData but not EHRData
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead. Please review the 0.13.0 changelog for more information.",
                        FutureWarning,
                        stacklevel=2,
                    )

                return func(data, *new_args, **kwargs)  # type: ignore

            # Handle keyword arguments - convert between old and new
            if has_old_param and new_param in kwargs:
                # Function expects old_param but got new_param
                data = kwargs.pop(new_param)

                # Issue warning if AnnData but not EHRData
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead. Please review the 0.13.0 changelog for more information.",
                        FutureWarning,
                        stacklevel=2,
                    )

                return func(**{old_param: data, **kwargs})  # type: ignore

            if has_new_param and old_param in kwargs:
                # Function expects new_param but got old_param
                data = kwargs.pop(old_param)

                # Always issue warning for old parameter name
                warnings.warn(
                    f"Parameter '{old_param}' is deprecated"
                    + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                    + f". Please use '{new_param}' instead.",
                    FutureWarning,
                    stacklevel=2,
                )

                # Additionally warn about AnnData
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead. Please review the 0.13.0 changelog for more information.",
                        FutureWarning,
                        stacklevel=2,
                    )

                return func(**{new_param: data, **kwargs})  # type: ignore

            # Pass through if using the correct parameter, but still check types
            if has_old_param and old_param in kwargs:
                data = kwargs[old_param]
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead. Please review the 0.13.0 changelog for more information.",
                        FutureWarning,
                        stacklevel=2,
                    )
                return func(**kwargs)  # type: ignore

            if has_new_param and new_param in kwargs:
                data = kwargs[new_param]
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead. Please review the 0.13.0 changelog for more information.",
                        FutureWarning,
                        stacklevel=2,
                    )
                return func(**kwargs)  # type: ignore

            # If neither parameter is provided
            if not edata_None_allowed:
                param_name = new_param if has_new_param else old_param
                alt_name = old_param if has_new_param else new_param
                raise TypeError(f"{func.__name__}() missing required argument: '{param_name}' (or '{alt_name}')")
            return func(**kwargs)  # type: ignore

        return cast("Callable[P, R]", wrapper)

    return decorator


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


def _cast_adata_to_match_data_type(input_data: AnnData, target_type_reference: EHRData | AnnData) -> EHRData | AnnData:
    """Cast the data object to the type used by the function."""
    if isinstance(input_data, type(target_type_reference)):
        return input_data

    if isinstance(target_type_reference, AnnData) and not isinstance(target_type_reference, EHRData):
        return input_data

    if isinstance(target_type_reference, EHRData):
        return EHRData.from_adata(input_data)

    raise ValueError(f"Used data object must be an AnnData or EHRData, got {type(target_type_reference)}")


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
            data: AnnData | EHRData | None
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

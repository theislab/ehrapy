# Since we might check whether an object is an instance of dask.array.Array
# without requiring dask installed in the environment.
from __future__ import annotations

import warnings
from functools import wraps
from importlib.util import find_spec
from inspect import signature
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar, cast

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
) -> Callable[[Callable[Concatenate[EHRData | AnnData, P], R]], Callable[P, R]]:
    """Decorator to migrate functions from AnnData to EHRData."""

    def decorator(func: Callable[Concatenate[EHRData | AnnData, P], R]) -> Callable[P, R]:
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
                        + ". Please use EHRData instead.",
                        DeprecationWarning,
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
                        + ". Please use EHRData instead.",
                        DeprecationWarning,
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
                    DeprecationWarning,
                    stacklevel=2,
                )

                # Additionally warn about AnnData
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead.",
                        DeprecationWarning,
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
                        + ". Please use EHRData instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                return func(**kwargs)  # type: ignore

            if has_new_param and new_param in kwargs:
                data = kwargs[new_param]
                if isinstance(data, AnnData) and not isinstance(data, EHRData):
                    warnings.warn(
                        f"Using AnnData with {func.__name__} is deprecated"
                        + (f" and will be removed after version {deprecated_after}" if deprecated_after else "")
                        + ". Please use EHRData instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                return func(**kwargs)  # type: ignore

            # If neither parameter is provided
            param_name = new_param if has_new_param else old_param
            alt_name = old_param if has_new_param else new_param
            raise TypeError(f"{func.__name__}() missing required argument: '{param_name}' (or '{alt_name}')")

        return cast("Callable[P, R]", wrapper)

    return decorator

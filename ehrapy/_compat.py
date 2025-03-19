# Since we might check whether an object is an instance of dask.array.Array
# without requiring dask installed in the environment.
from __future__ import annotations

import importlib.util
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


def _raise_array_type_not_implemented(func: Callable, type_: type) -> NotImplementedError:
    raise NotImplementedError(
        f"{func.__name__} does not support array type {type_}. Must be of type {func.registry.keys()}."  # type: ignore
    )


def is_dask_array(array):
    if DASK_AVAILABLE:
        return isinstance(array, da.Array)
    else:
        return False


def _check_module_importable(package: str) -> bool:
    """Checks whether a module is installed and can be loaded.

    Args:
        package: The package to check.

    Returns:
        True if the package is installed, False otherwise.
    """
    module_information = importlib.util.find_spec(package)
    module_available = module_information is not None

    return module_available


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

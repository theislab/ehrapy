from __future__ import annotations

import importlib.util
from subprocess import PIPE, Popen


def _check_module_importable(package: str) -> bool:  # pragma: no cover
    """Checks whether a module is installed and can be loaded.

    Args:
        package: The package to check.

    Returns:
        True if the package is installed, False otherweise
    """
    module_information = importlib.util.find_spec(package)
    module_available = module_information is not None

    return module_available


def _shell_command_accessible(command: list[str]) -> bool:  # pragma: no cover
    """Checks whether the provided command is accessible in the current shell.

    Args:
        command: The command to check. Spaces are separated as list elements.

    Returns:
        True if the command is accessible, False otherwise.
    """
    command_accessible = Popen(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    (commmand_stdout, command_stderr) = command_accessible.communicate()
    if command_accessible.returncode != 0:
        return False

    return True

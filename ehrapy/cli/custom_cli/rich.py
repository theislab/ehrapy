import os
import sys

from rich.console import Console


def force_terminal_in_github_action() -> Console:
    """Check, whether the GITHUB_ACTIONS environment variable is set or not.
    If it is set, the process runs in a workflow file and we need to tell rich, in order to get colored output as well.

    Returns:
        Rich Console object
    """
    if "GITHUB_ACTIONS" in os.environ:
        return Console(file=sys.stderr, force_terminal=True)
    else:
        return Console(file=sys.stderr)


# the console used for printing with rich
console = force_terminal_in_github_action()

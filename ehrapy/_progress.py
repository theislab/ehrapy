import functools
from collections.abc import Callable

from rich.progress import Progress, SpinnerColumn


def spinner(message: str = "Running task") -> Callable:
    def wrap(func):
        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            with Progress(
                "[progress.description]{task.description}",
                SpinnerColumn(),
                refresh_per_second=1500,
            ) as progress:
                progress.add_task(f"[blue]{message}", total=1)
                result = func(*args, **kwargs)
            return result

        return wrapped_f

    return wrap

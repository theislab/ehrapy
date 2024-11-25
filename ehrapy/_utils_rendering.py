import functools

from yaspin import yaspin


def spinner(message: str = "Running task"):
    def wrap(func):
        @functools.wraps(func)
        def wrapped_f(*args, **kwargs):
            with yaspin() as sp:
                sp.text = f"{message}..."
                result = func(*args, **kwargs)
                sp.ok("✔")
            return result

        return wrapped_f

    return wrap

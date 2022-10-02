import logging
from logging import CRITICAL, ERROR, WARNING, INFO, DEBUG
from datetime import datetime, timezone, timedelta
from functools import partial, update_wrapper
from typing import Optional

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")


class _RootLogger(logging.RootLogger):
    def __init__(self, level):
        super().__init__(level)
        self.propagate = False
        _RootLogger.manager = logging.Manager(self)

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: Optional[dict] = None,
        time: datetime = None,
        deep: Optional[str] = None,
    ) -> datetime:
        from ehrapy import settings
        now = datetime.now(timezone.utc)
        time_passed: timedelta = None if time is None else now - time
        extra = {
            **(extra or {}),
            "deep": deep if settings.verbosity.level < level else None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)


def _set_log_file(settings):
    file = settings.logfile
    name = settings.logpath
    root = settings._root_logger
    h = logging.StreamHandler(file) if name is None else logging.FileHandler(name)
    h.setFormatter(_LogFormatter())
    h.setLevel(root.level)
    if len(root.handlers) == 1:
        root.removeHandler(root.handlers[0])
    elif len(root.handlers) > 1:
        raise RuntimeError('Ehrapy’s root logger got more than one handler')
    root.addHandler(h)


def _set_log_level(settings, level: int):
    root = settings._root_logger
    root.setLevel(level)
    (h,) = root.handlers  # may only be 1
    h.setLevel(level)


class _LogFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def _versions_dependencies(dependencies):
    for mod in dependencies:
        mod_name, dist_name = mod if isinstance(mod, tuple) else (mod, mod)
        try:
            imp = __import__(mod_name)
            yield dist_name, imp.__version__
        except (ImportError, AttributeError):
            pass


def _copy_docs_and_signature(fn):
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])


def error(
    msg: str,
    *,
    time: datetime = None,
    deep: Optional[str] = None,
    extra: Optional[dict] = None,
) -> datetime:
    """
    Log message with specific level and return current time.

    Args:
        msg: Message to display.
        time: A time in the past.
              If this is passed, the time difference from then to now is appended to `msg` as `(HH:MM:SS)`.
              If `msg` contains `{time_passed}`, the time difference is instead inserted at that position
        deep: If the current verbosity is higher than the log function’s level, this gets displayed as well
        extra: Additional values you can specify in `msg` like `{time_passed}`.

    Returns:
        :class:`datetime.datetime` The current time.
    """
    from ehrapy import settings
    return settings._root_logger.error(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def warning(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa
    from ehrapy import settings
    return settings._root_logger.warning(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def info(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa
    from ehrapy import settings
    return settings._root_logger.info(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def hint(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa
    from ehrapy import settings
    return settings._root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(msg: str, *, time=None, deep=None, extra=None) -> datetime:  # noqa
    from ehrapy import settings
    return settings._root_logger.debug(msg, time=time, deep=deep, extra=extra)

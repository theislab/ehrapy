from pathlib import Path, PurePath
from typing import Union

supported_extensions = {"csv", "tsv", "h5ad", "pdf"}


def _get_file_extension(file_path: Path) -> str:
    """Check whether the argument is a filename.

    Args:
        file_path: Path to the file

    Returns:
        File extension of the specified file
    """
    ext = file_path.suffixes

    if len(ext) > 2:
        ext = ext[-2:]

    if ext and ext[-1][1:] in supported_extensions:
        return ext[-1][1:]
    raise ValueError(
        f"""\
{file_path!r} does not end on a valid extension.
Please, provide one of the available extensions.
{supported_extensions}
"""
    )


def _slugify(file_path: Union[str, PurePath]) -> str:
    """Transforms a Path into a string representation which is machine readable.

    Args:
        file_path: Path to the file

    Returns:
        Machine readable path representation as String
    """
    """Make a path into a filename."""
    if not isinstance(file_path, PurePath):
        file_path = PurePath(file_path)
    parts: list = list(file_path.parts)
    if parts[0] == "/":
        parts.pop(0)
    elif len(parts[0]) == 3 and parts[0][1:] == ":\\":
        parts[0] = parts[0][0]  # C:\ → C
    filename = "-".join(parts)
    assert "/" not in filename, filename
    assert not filename[1:].startswith(":"), filename

    return filename


def _is_float_convertable(string) -> bool:
    """Checks whether a string can be converted into a float

    http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python

    Args:
        string: The string to check for

    Returns:
        True if the string is float convertable, False otherwise
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def _is_int_convertable(string: Union[str, float]) -> bool:
    """Checks whether a string can be converted into an integer

    Args:
        string: The string to check for

    Returns:
        True if the string is int convertable, False otherwise
    """
    try:
        int(string)
        return True
    except ValueError:
        return False

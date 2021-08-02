from pathlib import Path, PurePath
from typing import Union

avail_exts = {"csv", "tsv", "tab", "txt"}


def is_valid_filename(filename: Path, return_ext=False):
    """Check whether the argument is a filename."""
    ext = filename.suffixes

    if len(ext) > 2:
        ext = ext[-2:]

    if ext and ext[-1][1:] in avail_exts:
        return ext[-1][1:] if return_ext else True
    elif not return_ext:
        return False
    raise ValueError(
        f"""\
{filename!r} does not end on a valid extension.
Please, provide one of the available extensions.
{avail_exts}
"""
    )


def _slugify(path: Union[str, PurePath]) -> str:
    """Make a path into a filename."""
    if not isinstance(path, PurePath):
        path = PurePath(path)
    parts = list(path.parts)
    if parts[0] == '/':
        parts.pop(0)
    elif len(parts[0]) == 3 and parts[0][1:] == ':\\':
        parts[0] = parts[0][0]  # C:\ → C
    filename = '-'.join(parts)
    assert '/' not in filename, filename
    assert not filename[1:].startswith(':'), filename
    return filename


def is_float(string):
    """\
    Check whether string is float.
    See also
    --------
    http://stackoverflow.com/questions/736043/checking-if-a-string-can-be-converted-to-float-in-python
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def is_int(string):
    """\
    Check whether string is int.
    """
    try:
        return int(string)
    except ValueError:
        return False

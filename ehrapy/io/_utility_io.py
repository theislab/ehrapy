from __future__ import annotations

from pathlib import Path

supported_extensions = {"csv", "tsv", "h5ad"}


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

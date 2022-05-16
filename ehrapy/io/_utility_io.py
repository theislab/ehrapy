from __future__ import annotations

from pathlib import Path

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


def _convert_x_only_to_obs_only(x_only: dict[str, list[str]] | list[str]) -> dict[str, list[str]] | list[str]:
    """Convert X only feature names to obs only names by adding all columns to obs only that are not in x_only.
    X only is later on never used in reading or writing, but its counterpart obs_only is.
    """
    if isinstance(x_only, list):
        pass
    else:
        pass

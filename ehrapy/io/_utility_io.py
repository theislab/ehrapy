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


def _check_columns_only_params(
    obs_only: dict[str, list[str]] | list[str] | None, x_only: dict[str, list[str]] | list[str] | None
) -> None:
    """Check whether columns_obs_only and columns_x_only can be used as passed. For a single anndata object (thus
    parameters being a list of strings) it's not possible to pass both, obs_only and x_only.
    For multiple anndata objects (thus the parameters being dicts of string keys with a list value), it is possible to pass both. But the keys
    (unique identifiers of the anndata objects, basically its names) should share no common identifier, thus a single anndata object is either in x_only OR
    obs_only, but not in both.
    """
    # at least one parameter is None
    if not obs_only or not x_only:
        return
    # cannot use both for a single anndata object
    if obs_only and x_only and isinstance(obs_only, list):
        raise ValueError(
            "Can not use columns_obs_only together with columns_x_only with a single AnnData object. At least one has to be None!"
        )
    # check for duplicates in the two dicts for multiple AnnData objects
    else:
        common_keys = obs_only.keys() & x_only.keys()  # type: ignore
        # at least one duplicated key has been found
        if common_keys:
            raise ValueError(
                "Can not use columns_obs_only together with columns_x_only for a single AnnData object. The following anndata identifiers where found"
                f"in both: {','.join(key for key in common_keys)}!"
            )

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ehrapy import settings
from ehrapy._compat import function_future_warning
from ehrapy.preprocessing._encoding import encode

if TYPE_CHECKING:
    from anndata import AnnData

supported_extensions = {"csv", "tsv", "h5ad"}


@function_future_warning("ep.io.write", "ehrdata.io.write_<format>")
def write(
    filename: str | Path,
    adata: AnnData,
    extension: str | bool = None,
    compression: Literal["gzip", "lzf"] | None = "gzip",
    compression_opts: int | None = None,
) -> None:
    """Write :class:`~anndata.AnnData` objects to file.

    It is possbile to either write an :class:`~anndata.AnnData` object to a .csv file or a .h5ad file.
    The .h5ad file can be used as a cache to save the current state of the object and to retrieve it faster once needed.
    This preserves the object state at the time of writing. It is possible to write both, encoded and unencoded objects.

    Args:
        filename: File name or path to write the file to.
        adata: Central data object.
        extension: File extension. One of 'h5ad', 'csv'. Defaults to `None` which infers the extension from the filename.
        compression: Optional file compression. One of 'gzip', 'lzf'.
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ed.dt.mimic_2()
        >>> ep.io.write("mimic_2.h5ad", adata)
    """
    function_future_warning("ehrapy.io.write", "ehrdata.io.write")
    filename = Path(filename)  # allow passing strings
    if _get_file_extension(filename):
        filename = filename
        _extension = _get_file_extension(filename)
        if extension is None:
            extension = _extension
        elif extension != _extension:
            raise ValueError(
                "It suffices to provide the file type by "
                "providing a proper extension to the filename."
                'One of "csv", "h5".'
            )
    else:
        key = filename
        extension = settings.file_format_data if extension is None else extension
        filename = _get_filename_from_key(key, extension)
    if extension == "csv":
        adata.write_csvs(filename)
    else:
        # dummy encoding when there is non numerical data in X
        if not np.issubdtype(adata.X.dtype, np.number) and extension == "h5ad":
            # flag to indicate an Anndata object has been dummy encoded to write it to .h5ad file
            # Case of writing an unencoded non numerical AnnData object
            encoded_adata = encode(adata, autodetect=True)
            encoded_adata.uns["ehrapy_dummy_encoding"] = True
            encoded_adata.uns["columns_obs_only"] = list(adata.obs.columns)
            encoded_adata.write(filename, compression=compression, compression_opts=compression_opts)
        else:
            adata.write(filename, compression=compression, compression_opts=compression_opts)


def _get_file_extension(file_path: Path) -> str:
    """Check whether the argument is a filename.

    Args:
        file_path: Path to the file.

    Returns:
        File extension of the specified file.
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


def _get_filename_from_key(key, extension=None) -> Path:
    """Gets full file name from a key.

    Args:
        key: Key to get file name for.
        extension: file extension.

    Returns:
        Path to the full file.
    """
    extension = settings.file_format_data if extension is None else extension
    extension = "csv" if extension is None else extension

    return settings.datasetdir / f"{key}.{extension}"

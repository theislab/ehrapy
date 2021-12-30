from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData

from ehrapy.api import settings
from ehrapy.api.io._utility_io import _get_file_extension


def write(
    filename: Union[str, Path],
    adata: AnnData,
    extension: Union[str, bool] = None,
    compression: Optional[Literal["gzip", "lzf"]] = "gzip",
    compression_opts: Optional[int] = None,
) -> None:
    """Write :class:`~anndata.AnnData` objects to file.

    Args:
        filename: File name or path to write the file to
        adata: Annotated data matrix.
        extension: File extension. One of h5, csv, txt
        compression: Optional file compression. One of gzip, lzf
        compression_opts: See http://docs.h5py.org/en/latest/high/dataset.html.

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = eh.data.mimic_2(encode=True)
            ep.io.write("mimic_2.h5ad", adata)
    """
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
                'One of "txt", "csv", "h5".'
            )
    else:
        key = filename
        extension = settings.file_format_data if extension is None else extension
        filename = _get_filename_from_key(key, extension)
    if extension == "csv":
        adata.write_csvs(filename)
    else:
        if not np.issubdtype(adata.X.dtype, np.number) and extension == "h5ad":
            raise ValueError(
                "Cannot write AnnData object containing non-numerical values to .h5ad file. Please "
                "encode your AnnData object before writing!"
            )
        adata.write(filename, compression=compression, compression_opts=compression_opts)


def _get_filename_from_key(key, extension=None) -> Path:
    """Gets full file name from a key.

    Args:
        key: Key to get file name for
        extension: file extension

    Returns:
        Path to the full file
    """
    extension = settings.file_format_data if extension is None else extension
    extension = "csv" if extension is None else extension
    return settings.datasetdir / f"{key}.{extension}"

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from anndata import AnnData

from ehrapy.api.io.utility_io import is_valid_filename


class DataWriter:
    @staticmethod
    def write(
        filename: Union[str, Path],
        adata: AnnData,
        extension: Union[str, bool] = None,
        compression: Optional[Literal["gzip", "lzf"]] = "gzip",
        compression_opts: Optional[int] = None,
    ) -> None:
        filename = Path(filename)  # allow passing strings
        if is_valid_filename(filename):
            filename = filename
            ext_ = is_valid_filename(filename, return_ext=True)
            if extension is None:
                extension = ext_
            elif extension != ext_:
                raise ValueError(
                    "It suffices to provide the file type by "
                    "providing a proper extension to the filename."
                    'One of "txt", "csv", "h5".'
                )
        else:
            key = filename
            # TODO get default format from settings
            extension = "csv" if extension is None else extension
            filename = DataWriter._get_filename_from_key(key, extension)
        if extension == "csv":
            adata.write_csvs(filename)
        else:
            if not np.issubdtype(adata.X.dtype, np.number) and extension == "h5ad":
                raise ValueError(
                    "Cannot write AnnData object containing non-numerical values to .h5ad file. Please "
                    "encode your AnnData object before writing!"
                )
            adata.write(filename, compression=compression, compression_opts=compression_opts)

    @staticmethod
    def _get_filename_from_key(key, extension=None) -> Path:
        """Gets full file name from a key.

        Args:
            key: Key to get file name for
            extension: file extension

        Returns:
            Path to the full file
        """
        # TODO replace with settings (default_file_format_data and write_dir)
        extension = "csv" if extension is None else extension
        return Path.cwd() / f"{key}.{extension}"

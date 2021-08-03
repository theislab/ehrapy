from pathlib import Path
from typing import Literal, Optional, Union

from anndata import AnnData

from ehrapy.api.io.utility_io import is_valid_filename


class Datawriter:
    @staticmethod
    def write(
        filename: Union[str, Path],
        adata: AnnData,
        ext: Optional[Literal["h5", "csv", "txt"]] = None,
        compression: Optional[Literal["gzip", "lzf"]] = "gzip",
        compression_opts: Optional[int] = None,
    ):
        """\
        Write :class:`~anndata.AnnData` objects to file.

        Parameters
        ----------
        filename
            If the filename has no file extension, it is interpreted as a key for
            generating a filename via `TODO file_format_data and write_dir`.

        adata
            Annotated data matrix.
        ext
            File extension from wich to infer file format. If `None`, defaults to
            `TODO`.
        compression
            See http://docs.h5py.org/en/latest/high/dataset.html.
        compression_opts
            See http://docs.h5py.org/en/latest/high/dataset.html.
        """
        filename = Path(filename)  # allow passing strings
        if is_valid_filename(filename):
            filename = filename
            ext_ = is_valid_filename(filename, return_ext=True)
            if ext is None:
                ext = ext_
            elif ext != ext_:
                raise ValueError(
                    "It suffices to provide the file type by "
                    "providing a proper extension to the filename."
                    'One of "txt", "csv", "h5".'
                )
        else:
            key = filename
            # TODO get default format from settings
            ext = "csv" if ext is None else ext
            filename = Datawriter._get_filename_from_key(key, ext)
        if ext == "csv":
            adata.write_csvs(filename)
        else:
            adata.write(filename, compression=compression, compression_opts=compression_opts)

    @staticmethod
    def _get_filename_from_key(key, ext=None) -> Path:
        """ """
        # TODO replace with settings (default_file_format_data and write_dir)
        ext = "csv" if ext is None else ext
        return Path.cwd() / f"{key}.{ext}"

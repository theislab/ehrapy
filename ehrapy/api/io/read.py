from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional, Union

import numpy as np
from anndata import AnnData
from anndata import read as read_h5ad
from rich import print

from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.io.utility_io import _slugify, avail_exts, is_float, is_int, is_valid_filename


class Datareader:
    @staticmethod
    def read(
        filename: Union[Path, str],
        extension: Optional[str] = None,
        delimiter: Optional[str] = None,
        cache: bool = False,
        backup_url: Optional[str] = None,
    ) -> AnnData:
        """Read file and return :class:`~anndata.AnnData` object.

        To speed up reading, consider passing ``cache=True``, which creates an hdf5 cache file.

        Args:
            filename: Name of the input file to read
            extension: Extension that indicates the file type. If ``None``, uses extension of filename.
            delimiter: Delimiter that separates data within text file. If ``None``, will split at arbitrary number of white spaces,
                       which is different from enforcing splitting at any single white space ``' '``.
            cache: If `False`, read from source, if `True`, read from fast 'h5ad' cache.
            backup_url: Retrieve the file from an URL if not present on disk.

        Returns:
            An :class:`~anndata.AnnData` object
        """
        file = Path(filename)
        if not file.exists():
            print("[bold yellow]Path or dataset does not yet exist. Attempting to download...")
            output_file_name = backup_url.split("/")[-1]
            is_zip: bool = output_file_name.endswith(".zip")
            Dataloader.download(backup_url, output_file_name=output_file_name, is_zip=is_zip)

        raw_anndata = Datareader._read(file, ext=extension, delimiter=delimiter, cache=cache)

        return raw_anndata

    @staticmethod
    def _read(
        filename: Path,
        ext=None,
        delimiter=None,
        cache: bool = False,
        backup_url: Optional[str] = None,
    ) -> AnnData:
        if ext is not None and ext not in avail_exts:
            raise ValueError("Please provide one of the available extensions.\n" f"{avail_exts}")
        else:
            ext = is_valid_filename(filename, return_ext=True)
        # read hdf5 files
        if ext in {"h5", "h5ad"}:
            return read_h5ad(filename)

        is_present = Datareader._check_datafile_present_and_download(filename, backup_url=backup_url)
        if not is_present:
            print(f"[bold red]... did not find original file {filename}")
        # TODO REPLACE WITH SETTINGS cachedir
        path_cache = Path.cwd() / _slugify(filename).replace("." + ext, ".h5ad")  # type: Path
        if path_cache.suffix in {".gz", ".bz2"}:
            path_cache = path_cache.with_suffix("")
        if cache and path_cache.is_file():
            return read_h5ad(path_cache)

        if not is_present:
            raise FileNotFoundError(f"Did not find file {filename}.")

        # do the actual reading
        if ext == "csv":
            raw_anndata = Datareader.read_csv(filename, dtype="object")
        elif ext in {"txt", "tab", "data", "tsv"}:
            raw_anndata = Datareader.read_text(filename, delimiter, dtype="object")
        else:
            raise ValueError(f"Unknown extension {ext}.")

        # TODO: FIX, does not work currently
        # if cache:
        #  if not path_cache.parent.is_dir():
        #     path_cache.parent.mkdir(parents=True)
        # write for faster reading when calling the next time
        # raw_anndata.write(path_cache)

        return raw_anndata

    @staticmethod
    def read_csv(
        filename: Union[Path, Iterator[str]],
        delimiter: Optional[str] = ",",
        dtype: str = "float32",
    ) -> AnnData:
        """\
        Read `.csv` file.
        Same as :func:`~anndata.read_text` but with default delimiter `','`.
        Parameters
        ----------
        filename
            Data file.
        delimiter
            Delimiter that separates data within text file.
            If `None`, will split at arbitrary number of white spaces,
            which is different from enforcing splitting at single white space `' '`.

        dtype
            Numpy data type.
        """
        return Datareader.read_text(filename, delimiter, dtype)

    @staticmethod
    def read_text(
        filename: Union[Path, Iterator[str]],
        delimiter: Optional[str] = None,
        dtype: str = "float32",
    ) -> AnnData:
        """\
        Read `.txt`, `.tab`, `.data` (text) file.
        Same as :func:`~anndata.read_csv` but with default delimiter `None`.
        Parameters
        ----------
        filename
            Data file, filename or stream.
        delimiter
            Delimiter that separates data within text file. If `None`, will split at
            arbitrary number of white spaces, which is different from enforcing
            splitting at single white space `' '`.
        dtype
            Numpy data type.
        """
        if not isinstance(filename, (Path, str, bytes)):
            return Datareader._read_text(filename, delimiter, dtype)

        filename = Path(filename)
        with filename.open() as f:
            return Datareader._read_text(f, delimiter, dtype)

    @staticmethod
    def iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
        """Helper for iterating only nonempty lines without line breaks"""
        for line in file_like:
            line = line.rstrip("\r\n")
            if line:
                yield line

    @staticmethod
    def _read_text(  # noqa:C901
        f: Iterator[str],
        delimiter: Optional[str],
        dtype: str,
    ) -> AnnData:
        comments = []
        data = []
        lines = Datareader.iter_lines(f)
        col_names = []
        row_names = []
        id_column_avail = False
        # read header and column names
        for line in lines:
            if line.startswith("#"):
                comment = line.lstrip("# ")
                if comment:
                    comments.append(comment)
            else:
                if delimiter is not None and delimiter not in line:
                    raise ValueError(f"Did not find delimiter {delimiter!r} in first line.")
                line_list = line.split(delimiter)
                # the first column might be row names, so check the last
                if not is_float(line_list[-1]):
                    col_names = line_list
                    # TODO: Throw warning exception here that no ID column found? -> We expect it to be the first col!
                    if "patient_id" == col_names[0].lower():
                        id_column_avail = True
                    # logg.msg("    assuming first line in file stores column names", v=4)
                else:
                    if not is_float(line_list[0]):
                        row_names.append(line_list[0])
                        Datareader._cast_vals_to_numeric(line_list[1:])
                        data.append(np.array(line_list[1:], dtype=dtype))
                    else:
                        Datareader._cast_vals_to_numeric(line_list)
                        data.append(np.array(line_list, dtype=dtype))
                break
        if not col_names:
            # try reading col_names from the last comment line
            if len(comments) > 0:
                # logg.msg("    assuming last comment line stores variable names", v=4)
                col_names = np.array(comments[-1].split())
            # just numbers as col_names
            else:
                # logg.msg("    did not find column names in file", v=4)
                col_names = np.arange(len(data[0])).astype(str)
        col_names = np.array(col_names, dtype=str)
        # read another line to check if first column contains row names or not
        for line in lines:
            line_list = line.split(delimiter)
            if id_column_avail:
                # logg.msg("    assuming first column in file stores row names", v=4)
                row_names.append(line_list[0])
                Datareader._cast_vals_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                Datareader._cast_vals_to_numeric(line_list)
                data.append(np.array(line_list, dtype=dtype))
            break
        # if row names are just integers
        if len(data) > 1 and data[0].size != data[1].size:
            # logg.msg(
            #     "    assuming first row stores column names and first column row names",
            #     v=4,
            # )
            col_names = np.array(data[0]).astype(int).astype(str)
            row_names.append(data[1][0].astype(int).astype(str))
            data = [data[1][1:]]
        # parse the file
        for line in lines:
            line_list = line.split(delimiter)
            if id_column_avail:
                row_names.append(line_list[0])
                Datareader._cast_vals_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                Datareader._cast_vals_to_numeric(line_list)
                data.append(np.array(line_list, dtype=dtype))
        # logg.msg("    read data into list of lists", t=True, v=4)
        # transfrom to array, this takes a long time and a lot of memory
        # but it’s actually the same thing as np.genfromtxt does
        # - we don’t use the latter as it would involve another slicing step
        #   in the end, to separate row_names from float data, slicing takes
        #   a lot of memory and CPU time
        if data[0].size != data[-1].size:
            raise ValueError(
                f"Length of first line ({data[0].size}) is different " f"from length of last line ({data[-1].size})."
            )
        data = np.array(data, dtype=dtype)
        # logg.msg("    constructed array from list of list", t=True, v=4)
        # transform row_names
        if not row_names:
            row_names = np.arange(len(data)).astype(str)
            # logg.msg("    did not find row names in file", v=4)
        else:
            row_names = np.array(row_names)
            for iname, name in enumerate(row_names):
                row_names[iname] = name.strip('"')
        # adapt col_names if necessary
        if col_names.size > data.shape[1]:
            col_names = col_names[1:]
        for iname, name in enumerate(col_names):
            col_names[iname] = name.strip('"')
        return AnnData(
            data,
            obs=dict(obs_names=row_names),
            var=dict(var_names=col_names),
            dtype=dtype,
            layers={"original": data.copy()},
        )

    @staticmethod
    def _check_datafile_present_and_download(path, backup_url=None):
        """Check whether the file is present, otherwise download."""
        path = Path(path)
        if path.is_file():
            return True
        if backup_url is None:
            return False
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True)

        Dataloader.download(backup_url, output_file_name=str(path))
        return True

    @staticmethod
    def _cast_vals_to_numeric(row: List[Optional[Union[str, int, float]]]) -> List[Optional[Union[str, int, float]]]:
        """Cast values to numerical datatype if possible"""
        for idx, val in enumerate(row):
            _is_int = is_int(val)
            if val == "0":
                row[idx] = 0
            elif val == "":
                row[idx] = None
            elif _is_int:
                row[idx] = _is_int
            elif is_float(val):
                row[idx] = float(val)
        return row

    @staticmethod
    def homogeneous_type(seq):
        """Check, whether all elements in an iterable are of the same type"""
        iseq = iter(seq)
        first_type = type(next(iseq))
        return first_type if all((type(x) is first_type) for x in iseq) else False

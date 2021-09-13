import warnings
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
from _collections import OrderedDict
from anndata import AnnData
from anndata import read as read_h5ad
from rich import print

from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.encode.encode import Encoder
from ehrapy.api.io._utility_io import _get_file_extension, _is_float, _is_int, _slugify, supported_extensions


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


class DataReader:
    suppress_warnings = False

    @staticmethod
    def read(
        filename: Union[Path, str],
        extension: Optional[str] = None,
        delimiter: Optional[str] = None,
        index_column: Union[str, Optional[int]] = None,
        columns_obs_only: Optional[List[Union[str]]] = None,
        cache: bool = False,
        backup_url: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> AnnData:
        DataReader.suppress_warnings = suppress_warnings
        file = Path(filename)
        if not file.exists():
            print("[bold yellow]Path or dataset does not yet exist. Attempting to download...")
            output_file_name = backup_url.split("/")[-1]
            is_zip: bool = output_file_name.endswith(".zip")  # TODO can we generalize this to tar files as well?
            Dataloader.download(backup_url, output_file_name=str(filename), output_path=str(Path.cwd()), is_zip=is_zip)

        raw_anndata = DataReader._read(
            filename=file,
            extension=extension,
            delimiter=delimiter,
            index_column=index_column,
            columns_obs_only=columns_obs_only,
            cache=cache,
        )
        return raw_anndata

    @staticmethod
    def _read(
        filename: Path,
        extension: Optional[str] = None,
        delimiter: Optional[str] = None,
        index_column: Union[str, Optional[int]] = None,
        columns_obs_only: Optional[List[Union[str]]] = None,
        cache: bool = False,
        backup_url: Optional[str] = None,
    ) -> Union[AnnData, np.ndarray]:
        """Internal interface of the read method."""
        if extension is not None and extension not in supported_extensions:
            raise ValueError("Please provide one of the available extensions.\n" f"{supported_extensions}")
        else:
            extension = _get_file_extension(filename)
        # read hdf5 files
        if extension in {"h5", "h5ad"}:
            return read_h5ad(filename)

        is_present = DataReader._check_datafile_present_and_download(filename, backup_url=backup_url)
        if not is_present:
            raise FileNotFoundError(f"Did not find file {filename}.")
        # TODO REPLACE WITH SETTINGS cachedir
        path_cache = Path.cwd() / _slugify(filename).replace("." + extension, ".h5ad")  # type: Path
        if path_cache.suffix in {".gz", ".bz2"}:
            path_cache = path_cache.with_suffix("")
        if cache and path_cache.is_file():
            cached_adata = read_h5ad(path_cache)
            cached_adata.X = cached_adata.X.astype("object")
            cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)

            return cached_adata

        # do the actual reading
        if extension in {"csv", "tsv"}:
            raw_anndata = DataReader.read_csv(filename, delimiter, index_column, columns_obs_only)
        elif extension in {"txt", "tab", "data"}:
            raw_anndata = DataReader.read_text(filename, delimiter, dtype="object")
        else:
            raise ValueError(f"Unknown extension: {extension}.")

        if cache:
            if not path_cache.parent.is_dir():
                path_cache.parent.mkdir(parents=True)
            # write for faster reading when calling the next time
            cached_adata = Encoder.encode(ann_data=raw_anndata, autodetect=True)
            cached_adata.write(path_cache)
            cached_adata.X = cached_adata.X.astype("object")
            cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)
            return cached_adata

        return raw_anndata

    @staticmethod
    def read_csv(
        filename: Union[Path, Iterator[str]],
        delimiter: Optional[str] = ",",
        index_column: Union[str, Optional[int]] = None,
        columns_obs_only: Optional[List[Union[str]]] = None,
    ) -> AnnData:
        """Read `.csv` and `.tsv` file.

        Args:
            filename: File path to the csv file.
            delimiter: Delimiter separating the csv data within the file.
            index_column: Index or column name of the index column (obs)
            columns_obs_only: List of columns which only be stored in .obs, but not in X. Useful for free text annotations.

        Returns:
            An :class:`~anndata.AnnData` object
        """
        # read pandas dataframe
        initial_df = pd.read_csv(filename, delimiter=delimiter)
        # return the raw AnnData object
        return DataReader._df_to_anndata(initial_df, index_column, columns_obs_only)

    @staticmethod
    def read_text(
        filename: Union[Path, Iterator[str]],
        delimiter: Optional[str] = None,
        dtype: str = "float32",
    ) -> AnnData:
        """Read `.txt`, `.tab`, `.data` (text) file.

        Args:
            filename: File name or stream
            delimiter:  Delimiter that separates data within text file.
            If `None`, will split at arbitrary number of white spaces, which is different from enforcing splitting at single white space `' '`.
            dtype: Numpy data type.

        Returns:
            An :class:`~anndata.AnnData` object
        """
        if not isinstance(filename, (Path, str, bytes)):
            return DataReader._read_text(filename, delimiter, dtype)

        filename = Path(filename)
        with filename.open() as f:
            return DataReader._read_text(f, delimiter, dtype)

    @staticmethod
    def _iter_lines(file_like: Iterable[str]) -> Generator[str, None, None]:
        """Helper for iterating only nonempty lines without line breaks"""
        for line in file_like:
            line = line.rstrip("\r\n")
            if line:
                yield line

    @staticmethod
    def _read_text(  # noqa:C901
        file_iterator: Iterator[str],
        delimiter: Optional[str],
        dtype: str,
    ) -> AnnData:
        comments: List = []
        data: List = []
        lines: Generator = DataReader._iter_lines(file_iterator)
        column_names: List = []
        row_names: List = []
        id_column_avail: bool = False
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
                if not _is_float(line_list[-1]):
                    column_names = line_list
                    if "patient_id" == column_names[0].lower():
                        id_column_avail = True
                else:
                    if not _is_float(line_list[0]):
                        row_names.append(line_list[0])
                        DataReader._cast_vals_to_numeric(line_list[1:])
                        data.append(np.array(line_list[1:], dtype=dtype))
                    else:
                        DataReader._cast_vals_to_numeric(line_list)
                        data.append(np.array(line_list, dtype=dtype))
                break
        # TODO Fix the unused variables
        if not column_names:
            # try reading col_names from the last comment line
            if len(comments) > 0:
                column_names_arr = np.array(comments[-1].split())
            # just numbers as col_names
            else:
                column_names_arr = np.arange(len(data[0])).astype(str)
        column_names_arr = np.array(column_names, dtype=str)
        # read another line to check if first column contains row names or not
        for line in lines:
            line_list = line.split(delimiter)
            if id_column_avail:
                row_names.append(line_list[0])
                DataReader._cast_vals_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                DataReader._cast_vals_to_numeric(line_list)
                data.append(np.array(line_list, dtype=dtype))
            break
        # if row names are just integers
        if len(data) > 1 and data[0].size != data[1].size:
            column_names_arr = np.array(data[0]).astype(int).astype(str)
            row_names.append(data[1][0].astype(int).astype(str))
            data = [data[1][1:]]
        # parse the file
        for line in lines:
            line_list = line.split(delimiter)
            if id_column_avail:
                row_names.append(line_list[0])
                DataReader._cast_vals_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                DataReader._cast_vals_to_numeric(line_list)
                data.append(np.array(line_list, dtype=dtype))
        if data[0].size != data[-1].size:
            raise ValueError(
                f"Length of first line ({data[0].size}) is different " f"from length of last line ({data[-1].size})."
            )
        data_array = np.array(data, dtype=dtype)
        # transform row_names
        if not row_names:
            row_names_arr = np.arange(len(data_array)).astype(str)
        else:
            row_names_arr = np.array(row_names)
            for iname, name in enumerate(row_names_arr):
                row_names_arr[iname] = name.strip('"')
        # adapt col_names if necessary
        if column_names_arr.size > data_array.shape[1]:
            column_names_arr = column_names_arr[1:]
        for iname, name in enumerate(column_names_arr):
            column_names_arr[iname] = name.strip('"')

        return AnnData(
            X=data_array,
            obs=dict(obs_names=row_names_arr),
            var=dict(var_names=column_names_arr),
            dtype=dtype,
            layers={"original": data_array.copy()},
        )

    @staticmethod
    def _df_to_anndata(
        df: pd.DataFrame, index_column: Union[str, Optional[int]], columns_obs_only: Optional[List[Union[str]]]
    ) -> AnnData:
        """Create an AnnData object from the initial dataframe"""
        # set index given or default
        df = DataReader._set_index(df, index_column)
        # move columns from the input dataframe to later obs
        dataframes = DataReader.move_columns_to_obs(df, columns_obs_only)
        X = dataframes.df.to_numpy(copy=True)

        return AnnData(
            X,
            obs=dataframes.obs,
            var=pd.DataFrame(index=list(dataframes.df.columns)),
            dtype="object",
            layers={"original": X.copy()},
        )

    @staticmethod
    def _decode_cached_adata(adata: AnnData, column_obs_only: List[str]) -> AnnData:
        """Decode the label encoding of initial AnnData object

        Args:
            adata: The label encoded AnnData object
            column_obs_only: The columns, that should be kept in obs

        Returns:
            The decoded, initial AnnData object
        """
        var_names = list(adata.var_names)
        # for each encoded categorical, replace its encoded values with its original values in X
        for idx, var_name in enumerate(var_names):
            if not var_name.startswith("ehrapycat_"):
                break
            value_name = var_name[10:]
            original_values = adata.uns["original_values_categoricals"][value_name]
            adata.X[:, idx : idx + 1] = original_values
            # update var name per categorical
            var_names[idx] = value_name
        # drop all columns, that are not obs only in obs
        if column_obs_only:
            adata.obs = adata.obs[column_obs_only]
        else:
            adata.obs = pd.DataFrame(index=adata.obs.index)
        # set the new var names (unencoded ones)
        adata.var.index = var_names
        # update original layer as well
        adata.layers["original"] = adata.X.copy()
        # reset uns
        adata.uns = OrderedDict()

        return adata

    @staticmethod
    def _check_datafile_present_and_download(path: Union[str, Path], backup_url=None) -> bool:
        """Check whether the file is present, otherwise download.

        Args:
            path: Path to the file to check
            backup_url: Backup URL if the file cannot be found and has to be downloaded

        Returns:
            True if the file was present. False if not.
        """
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
    def _set_index(df: pd.DataFrame, index_column: Union[str, Optional[int]]) -> pd.DataFrame:
        """Try to set the index, if any given by the index_column parameter."""
        column_names = list(df.columns)
        if isinstance(index_column, str):
            df = df.set_index(index_column)
        elif isinstance(index_column, int):
            df = df.set_index(column_names[index_column])
        else:
            if "patient_id" == column_names[0]:
                df = df.set_index("patient_id")
            else:
                if not DataReader.suppress_warnings:
                    warnings.warn(
                        "Did not find patient_id column at column 0 and no index column was passed. Using default, numerical indices instead!",
                        IndexColumnWarning,
                    )

        return df

    @staticmethod
    def move_columns_to_obs(df: pd.DataFrame, columns_obs_only: Optional[List[Union[str]]]) -> BaseDataframes:
        """
        Move the given columns from the original dataframe (and therefore X) to obs. By doing so, those values will not get lost
        and will be stored in obs, but will not appear in X. This may be useful for textual values like free text.
        """
        if columns_obs_only:
            try:
                obs = df[columns_obs_only].copy()
                obs = obs.set_index(df.index.map(str))
                df = df.drop(columns_obs_only, axis=1)
            # TODO Key error traceback still prints, have no idea why
            except KeyError:
                raise ColumnNotFoundError(
                    "One or more column names passed to column_obs_only were not found in the input data. "
                    "Make sure you spelled the column names correctly."
                )
        else:
            obs = pd.DataFrame(index=df.index.map(str))

        return BaseDataframes(obs, df)

    @staticmethod
    def _cast_vals_to_numeric(row: List[Optional[Union[str, int, float]]]) -> List[Optional[Union[str, int, float]]]:
        """Cast values to numerical datatype if possible.

        Args:
            row: List of values to cast

        Returns:
            A new List of values casted into the appropriate data type
        """
        for idx, val in enumerate(row):
            __is_int: bool = _is_int(val)
            if val == "0":
                row[idx] = 0
            elif val == "":
                row[idx] = None
            elif __is_int:
                row[idx] = int(val)
            elif _is_float(val):
                row[idx] = float(val)

        return row

    @staticmethod
    def homogeneous_type(sequence):
        """Check, whether all elements in an iterable are of the same type.

        Args:
            sequence: Sequence to check

        Returns:
            True if all elements are of the same type, False otherwise.
        """
        iseq = iter(sequence)
        first_type = type(next(iseq))

        return first_type if all((type(x) is first_type) for x in iseq) else False


class IndexColumnWarning(UserWarning):
    pass


class ColumnNotFoundError(Exception):
    pass

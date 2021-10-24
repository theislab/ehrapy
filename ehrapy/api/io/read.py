from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from _collections import OrderedDict
from anndata import AnnData
from anndata import read as read_h5ad
from mudata import MuData
from rich import print

from ehrapy.api import settings
from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.encode.encode import Encoder
from ehrapy.api.io._utility_io import (
    _get_file_extension,
    _is_float_convertable,
    _is_int_convertable,
    _slugify,
    supported_extensions,
)


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
        return_mudata: bool = False,
        cache: bool = False,
        backup_url: Optional[str] = None,
        suppress_warnings: bool = False,
    ) -> Union[AnnData, Dict[str, AnnData], MuData]:
        DataReader.suppress_warnings = suppress_warnings
        file = Path(filename)
        if not file.exists():
            print("[bold yellow]Path or dataset does not yet exist. Attempting to download...")
            output_file_name = backup_url.split("/")[-1]
            is_zip: bool = output_file_name.endswith(".zip")  # TODO can we generalize this to tar files as well?
            Dataloader.download(backup_url, output_file_name=str(filename), output_path=str(Path.cwd()), is_zip=is_zip)
            # TODO: temporary fix for demo
            file = Path.cwd() / "mimic-iii-clinical-database-demo-1.4"

        raw_object = DataReader._read(
            filename=file,
            extension=extension,
            delimiter=delimiter,
            index_column=index_column,
            columns_obs_only=columns_obs_only,
            return_mudata=return_mudata,
            cache=cache,
        )
        return raw_object

    @staticmethod
    def _read(
        filename: Path,
        extension: Optional[str] = None,
        delimiter: Optional[str] = None,
        index_column: Union[str, Optional[int]] = None,
        columns_obs_only: Optional[List[Union[str]]] = None,
        return_mudata: bool = False,
        cache: bool = False,
        backup_url: Optional[str] = None,
    ) -> Union[MuData, Dict[str, AnnData], AnnData]:
        """Internal interface of the read method."""
        # check, whether the datafile(s) is/are present or not
        is_present = DataReader._check_datafiles_present_and_download(filename, backup_url=backup_url)
        if not is_present and not filename.is_dir() and not filename.is_file():
            print(
                "[bold red]Attempted download of missing file(s) failed. Please file an issue at our repository "
                "[blue]https://github.com/theislab/ehrapy!"
            )
        # If the filename is a directory, assume it is a dataset with multiple files
        if filename.is_dir():
            return DataReader._read_multiple_csv(filename, delimiter, index_column, columns_obs_only, return_mudata)

        if extension is not None and extension not in supported_extensions:
            raise ValueError("Please provide one of the available extensions.\n" f"{supported_extensions}")
        else:
            extension = _get_file_extension(filename)
        # read hdf5 files
        if extension in {"h5", "h5ad"}:
            return read_h5ad(filename)

        if not is_present:
            raise FileNotFoundError(f"Did not find file {filename}.")
        path_cache = settings.cachedir / _slugify(filename).replace("." + extension, ".h5ad")  # type: Path
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
            cached_adata = Encoder.encode(data=raw_anndata, autodetect=True)
            cached_adata.write(path_cache)
            cached_adata.X = cached_adata.X.astype("object")
            cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)
            return cached_adata

        return raw_anndata

    @staticmethod
    def _read_multiple_csv(  # noqa: N802
        filename: Path,
        delimiter: Optional[str] = None,
        index_column: Union[Union[str, Optional[int]], Dict[str, Union[str, Optional[int]]]] = None,
        columns_obs_only: Union[Optional[List[Union[str]]], Dict[str, Optional[List[Union[str]]]]] = None,
        return_mudata_object: bool = False,
    ) -> Union[MuData, Dict[str, AnnData]]:
        """Read a dataset containing multiple files (in this case .csv or .tsv files).

        Args:
            filename: File path to the directory containing multiple csvs dataset.
            delimiter: Delimiter separating the data within the file.
            index_column: Indices or column names of the index columns (obs)
            columns_obs_only: List of columns per file (thus AnnData object) which should only be stored in .obs, but not in X. Useful for free text annotations.
            return_mudata_object: If set to True, return a :class:`~mudata.MuData` object, else a dict of :class:`~anndata.AnnData` objects

        Returns:
            An :class:`~mudata.MuData` object or a dict mapping the filename (object name) to the corresponding :class:`~anndata.AnnData` object.
        """
        if not return_mudata_object:
            anndata_dict = {}
        else:
            mudata = None
        for file in filename.iterdir():
            if file.is_file() and file.suffix in [".csv", ".tsv"]:
                # slice off the file suffix as this is not needed for identifier
                adata_identifier = file.name[:-4]
                index_col, col_obs_only = DataReader._extract_index_and_columns_obs_only(
                    adata_identifier, index_column, columns_obs_only
                )
                adata = DataReader.read_csv(file, delimiter, index_col, col_obs_only)
                # obs indices have to be unique otherwise updating and working with the MuData object will fail
                if index_col:
                    adata.obs_names_make_unique()

                if return_mudata_object:
                    if not mudata:
                        mudata = MuData({adata_identifier: adata})
                    else:
                        mudata.mod[adata_identifier] = adata
                else:
                    anndata_dict[adata_identifier] = adata
        if return_mudata_object:
            # create the MuData object with the AnnData objects as modalities
            mudata.update()
            return mudata
        else:
            return anndata_dict

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
        try:
            if index_column and columns_obs_only and index_column in columns_obs_only:
                print(
                    f"[bold yellow]Index column [blue]{index_column} [yellow]is also used as a column "
                    f"for obs only. Using default indices instead and moving [blue]{index_column} [yellow]to column_obs_only."
                )
                index_column = None
            initial_df = pd.read_csv(filename, delimiter=delimiter, index_col=index_column)
        # possible cause: index column is misspelled (or does not exist at all in this file)
        except ValueError:
            raise IndexNotFoundError(
                f"Could not create AnnData object while reading file {filename}. Does index_column named {index_column} "
                f"exist in {filename}?"
            ) from None

        # get all object dtype columns
        object_type_columns = [col_name for col_name in initial_df.columns if initial_df[col_name].dtype == "object"]
        # if columns_obs_only is None, initialize it as datetime columns need to be included here
        if not columns_obs_only:
            columns_obs_only = []

        for col in object_type_columns:
            try:
                pd.to_datetime(initial_df[col])
                columns_obs_only.append(col)
            except (ValueError, TypeError):
                pass

        # return the initial AnnData object
        return DataReader._df_to_anndata(initial_df, columns_obs_only)

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
                if not _is_float_convertable(line_list[-1]):
                    column_names = line_list
                    if "patient_id" == column_names[0].lower():
                        id_column_avail = True
                else:
                    if not _is_float_convertable(line_list[0]):
                        row_names.append(line_list[0])
                        DataReader._cast_values_to_numeric(line_list[1:])
                        data.append(np.array(line_list[1:], dtype=dtype))
                    else:
                        DataReader._cast_values_to_numeric(line_list)
                        data.append(np.array(line_list, dtype=dtype))
                break
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
                DataReader._cast_values_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                DataReader._cast_values_to_numeric(line_list)
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
                DataReader._cast_values_to_numeric(line_list[1:])
                data.append(np.array(line_list[1:], dtype=dtype))
            else:
                DataReader._cast_values_to_numeric(line_list)
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
    def _df_to_anndata(df: pd.DataFrame, columns_obs_only: Optional[List[Union[str]]]) -> AnnData:
        """Create an AnnData object from the initial dataframe"""
        # move columns from the input dataframe to later obs
        dataframes = DataReader._move_columns_to_obs(df, columns_obs_only)
        X = dataframes.df.to_numpy(copy=True)

        return AnnData(
            X=X,
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
    def _check_datafiles_present_and_download(path: Union[str, Path], backup_url=None) -> bool:
        """Check whether the file or directory is present, otherwise download.

        Args:
            path: Path to the file or directory to check
            backup_url: Backup URL if the file cannot be found and has to be downloaded

        Returns:
            True if the file or directory was present. False if not.
        """
        path = Path(path)
        if path.is_file() or path.is_dir():
            return True
        if backup_url is None:
            return False
        if not path.is_dir() and not path.parent.is_dir():
            path.parent.mkdir(parents=True)

        Dataloader.download(backup_url, output_file_name=str(path))

        return True

    @staticmethod
    def _extract_index_and_columns_obs_only(identifier: str, index_columns, columns_obs_only):
        """
        Extract the index column (if any) and the columns, for obs only (if any) from the given user input.

        For each file, `index_columns` and `columns_obs_only` can provide three cases:
            1.) The filename (thus the identifier) is not present as a key and no default key is provided or one or both dicts are empty:
                --> No index column will be set and/or no columns are obs only (based on user input)

                .. code-block:: python
                       # some setup code here
                       ...
                       # filename
                       identifier1 = "MyFile"
                       identifier2 = "MyOtherFile"
                       # no default key and identifier1 is not in the index or columns_obs_only keys
                       # -> no index column will be set and no columns will be obs only (except datetime, if any)
                       index_columns = {"MyOtherFile":["MyOtherColumn1"]}
                       columns_obs_only = {"MyOtherFile":["MyOtherColumn2"]}

            2.) The filename (thus the identifier) is not present as a key, but default key is provided
                --> The index column will be set and/or columns will be obs only according to the default key

                .. code-block:: python
                      # some setup code here
                       ...
                       # filename
                       identifier1 = "MyFile"
                       identifier2 = "MyOtherFile"
                       # identifier1 is not in the index or columns_obs_only keys, but default key is set for both
                       # -> index column will be set using MyColumn1 and column obs only will include MyColumn2
                       index_columns = {"MyOtherFile":["MyOtherColumn1"], "default": "MyColumn1"}
                       columns_obs_only = {"MyOtherFile":["MyOtherColumn2"], "default": "MyColumn2"}

            3.) The filename is present as a key
                --> The index column will be set and/or columns are obs only according to its value

                .. code-block:: python
                       # some setup code here
                       ...
                       # filename
                       identifier1 = "MyFile"
                       identifier2 = "MyOtherFile"
                       # identifier1 is in the index and columns_obs_only keys
                       # -> index column will be MyColumn1 and columns_obs_only will include MyColumn2 and MyColumn3
                       index_columns = {"MyFile":["MyColumn1"]}
                       columns_obs_only = {"MyFile":["MyColumn2", "MyColumn3"]}

        Args:
            identifier: The name of the
            index_columns: Index columns
            columns_obs_only: Columns for obs only

        Returns:
            Index column (if any) and columns obs only (if any) for this specific AnnData object
        """
        _index_column = None
        _columns_obs_only = None
        # get index column (if any)
        if index_columns and identifier in index_columns.keys():
            _index_column = index_columns[identifier]
        elif index_columns and "default" in index_columns.keys():
            _index_column = index_columns["default"]

        # get columns obs only (if any)
        if columns_obs_only and identifier in columns_obs_only.keys():
            _columns_obs_only = columns_obs_only[identifier]
        elif columns_obs_only and "default" in columns_obs_only.keys():
            _columns_obs_only = columns_obs_only["default"]

        # if there is only one obs only column, it might have been passed as single string
        if isinstance(_columns_obs_only, str):
            _columns_obs_only = [_columns_obs_only]

        # if index column is also found in column_obs_only, use default indices instead and only move it to obs only, but warn the user
        if _index_column and _index_column in _columns_obs_only:
            print(
                f"[bold yellow]Index column [blue]{_index_column} [yellow]for file [blue]{identifier} [yellow]is also used as a column "
                f"for obs only. Using default indices instead and moving [blue]{_index_column} [yellow]to column_obs_only."
            )
            _index_column = None

        return _index_column, _columns_obs_only

    @staticmethod
    def _move_columns_to_obs(df: pd.DataFrame, columns_obs_only: Optional[List[str]]) -> BaseDataframes:
        """Move the given columns from the original dataframe (and therefore X) to obs.

        By moving these values will not get lost and will be stored in obs, but will not appear in X.
        This may be useful for textual values like free text.

        Args:
            df: Pandas Dataframe to move the columns for
            columns_obs_only: Columns to move to obs only

        Returns:
            A modified :class:`~pd.DataFrame` object
        """
        if columns_obs_only:
            try:
                obs = df[columns_obs_only].copy()
                obs = obs.set_index(df.index.map(str))
                df = df.drop(columns_obs_only, axis=1)
            except KeyError:
                raise ColumnNotFoundError from KeyError(
                    "One or more column names passed to column_obs_only were not found in the input data. "
                    "Make sure you spelled the column names correctly."
                )
        else:
            obs = pd.DataFrame(index=df.index.map(str))

        return BaseDataframes(obs, df)

    @staticmethod
    def _cast_values_to_numeric(row: List[Optional[Union[str, int, float]]]) -> List[Optional[Union[str, int, float]]]:
        """Cast values to numerical datatype if possible.

        Args:
            row: List of values to cast

        Returns:
            A new List of values casted into the appropriate data type
        """
        for idx, val in enumerate(row):
            _is_int: bool = _is_int_convertable(val)
            if val == "0":
                row[idx] = 0
            elif val == "":
                row[idx] = None
            elif _is_int:
                row[idx] = int(val)
            elif _is_float_convertable(val):
                row[idx] = float(val)

        return row

    @staticmethod
    def _is_homogeneous_type(sequence: Sequence):
        """Check, whether all elements in an iterable are of the same type.

        Args:
            sequence: Sequence to check

        Returns:
            True if all elements are of the same type, False otherwise.
        """
        iseq = iter(sequence)
        first_type = type(next(iseq))

        return first_type if all((type(el) is first_type) for el in iseq) else False


class IndexNotFoundError(Exception):
    pass


class ColumnNotFoundError(Exception):
    pass

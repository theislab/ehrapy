import shutil
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
from _collections import OrderedDict
from anndata import AnnData
from anndata import read as read_h5ad
from mudata import MuData
from rich import print

from ehrapy.api import ehrapy_settings, settings
from ehrapy.api.data.dataloader import Dataloader
from ehrapy.api.io._utility_io import _get_file_extension, _slugify, supported_extensions
from ehrapy.api.preprocessing.encoding._encode import Encoder


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


class DataReader:
    suppress_warnings = False

    @staticmethod
    def read(
        dataset_path: Union[Path, str],
        extension: Optional[str] = None,
        delimiter: Optional[str] = None,
        index_column: Optional[Union[Dict[str, Union[str, int]], Union[str, int]]] = None,
        columns_obs_only: Optional[Union[Dict[str, List[str]], List[str]]] = None,
        return_mudata: bool = False,
        cache: bool = False,
        backup_url: Optional[str] = None,
        suppress_warnings: bool = False,
        download_dataset_name: Optional[str] = None,
    ) -> Union[AnnData, Dict[str, AnnData], MuData]:
        """Reads or downloads a desired directory or single file.

        Args:
            dataset_path: Path to the file or directory to read.
            extension: File extension. Required to select the appropriate file reader.
            delimiter: File delimiter. Required for e.g. csv vs tsv files.
            index_column: The index column of obs.
            columns_obs_only: Which columns to only add to obs and not X.
            return_mudata: Whether to create and return a MuData object.
            cache: Whether to use the cache when reading.
            download_dataset_name: Name of the file or directory in case the dataset is downloaded
            backup_url: URL to download the data file(s) from if not yet existing.
            suppress_warnings: Whether to suppress warnings.

        Returns:
            An :class:`~anndata.AnnData` object, a :class:`~mudata.MuData` object or a dict with an identifier (usually the filename, without extension)
            for each :class:`~anndata.AnnData` object in the dict


        """
        DataReader.suppress_warnings = suppress_warnings
        file: Path = Path(dataset_path)

        if not file.exists():
            if backup_url is not None:
                download_default_name = backup_url.split("/")[-1]
                download_dataset_name = download_dataset_name or download_default_name
                # currently supports zip, tar, gztar, bztar, xztar
                archive_formats, _ = zip(*shutil.get_archive_formats())
                is_archived = download_default_name[-3:] in archive_formats

            else:
                raise BackupURLNotProvidedError(
                    f"File or directory {file} does not exist and no backup_url was provided.\n"
                    f"Please provide a backup_url or check whether path is spelled correctly."
                )
            print("[bold yellow]Path or dataset does not yet exist. Attempting to download...")
            Dataloader.download(
                backup_url,
                output_file_name=download_default_name,
                output_path=ehrapy_settings.datasetdir,
                is_archived=is_archived,
            )
            # if archived, remove archive suffix
            archive_extension = download_default_name[-4:]
            output_path_name = (
                download_default_name.replace(archive_extension, "") if is_archived else download_default_name
            )
            output_file_or_dir = ehrapy_settings.datasetdir / output_path_name
            moved_path = Path(str(output_file_or_dir)[: str(output_file_or_dir).rfind("/") + 1]) / download_dataset_name
            shutil.move(output_file_or_dir, moved_path)  # type: ignore
            file = moved_path

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
        index_column: Optional[Union[Dict[str, Union[str, int]], Union[str, int]]] = None,
        columns_obs_only: Optional[Union[Dict[str, List[str]], List[str]]] = None,
        return_mudata: bool = False,
        cache: bool = False,
        backup_url: Optional[str] = None,
    ) -> Union[MuData, Dict[str, AnnData], AnnData]:
        """Internal interface of the read method."""
        if cache and return_mudata:
            raise MudataCachingNotSupportedError(
                "Caching is currently not supported for MuData objects. Consider setting return_mudata to False in order "
                "to use caching!"
            )
        # check, whether the datafile(s) is/are present or not
        is_present = DataReader._check_datafiles_present_and_download(filename, backup_url=backup_url)
        if not is_present and not filename.is_dir() and not filename.is_file():
            print(
                "[bold red]Attempted download of missing dataset file(s) failed. Please file an issue at our repository "
                "[blue]https://github.com/theislab/ehrapy!"
            )
        path_cache_dir = settings.cachedir / filename
        # read from cache directory if wanted and available
        if cache and path_cache_dir.is_dir():
            return DataReader._read_from_cache_dir(path_cache_dir)
        # If the filename is a directory, assume it is a dataset with multiple files
        elif filename.is_dir():
            adata_objects, columns_obs_only = DataReader._read_multiple_csv(
                filename, delimiter, index_column, columns_obs_only, return_mudata, cache
            )
            if cache:
                if not path_cache_dir.parent.is_dir():
                    path_cache_dir.parent.mkdir(parents=True)
                path_cache_dir.mkdir()
                return DataReader._write_cache_dir(adata_objects, path_cache_dir, columns_obs_only, index_column)  # type: ignore
            return adata_objects
        # dataset seems to be a single file, not a directory of multiple files
        else:
            if extension is not None and extension not in supported_extensions:
                raise ValueError("Please provide one of the available extensions.\n" f"{supported_extensions}")
            else:
                extension = _get_file_extension(filename)
            # read hdf5 files
            if extension in {"h5", "h5ad"}:
                return read_h5ad(filename)

            # read from cache file
            path_cache = settings.cachedir / _slugify(filename).replace("." + extension, ".h5ad")  # type: Path
            if path_cache.suffix in {".gz", ".bz2"}:
                path_cache = path_cache.with_suffix("")
            # previously cached data reading
            if cache and path_cache.is_file():
                return DataReader._read_from_cache(path_cache)

            # read from other files that are currently supported
            elif extension in {"csv", "tsv"}:
                raw_anndata, columns_obs_only = DataReader.read_csv(
                    filename, delimiter, index_column, columns_obs_only, cache  # type: ignore
                )
            else:
                raise NotImplementedError(f"There is currently no parser implemented for {extension} files!")
            # cache results if wanted
            if cache:
                if not path_cache.parent.is_dir():
                    path_cache.parent.mkdir(parents=True)
                return DataReader._write_cache(raw_anndata, path_cache, columns_obs_only)  # type: ignore
            return raw_anndata

    @staticmethod
    def _read_multiple_csv(  # noqa: N802
        filename: Path,
        delimiter: Optional[str] = None,
        index_column: Optional[Union[Dict[str, Union[str, int]], Union[str, int]]] = None,
        columns_obs_only: Union[Optional[List[Union[str]]], Dict[str, Optional[List[Union[str]]]]] = None,
        return_mudata_object: bool = False,
        cache: bool = False,
    ) -> Tuple[Union[MuData, Dict[str, AnnData]], Dict[str, List[str]]]:
        """Read a dataset containing multiple .csv/.tsv files.

        Args:
            filename: File path to the directory containing multiple .csv/.tsv files.
            delimiter: Delimiter separating the data within the file.
            index_column: Column names of the index columns for obs
            columns_obs_only: List of columns per file (AnnData object) which should only be stored in .obs, but not in X. Useful for free text annotations.
            return_mudata_object: When set to True, return a :class:`~mudata.MuData` object, otherwise a dict of :class:`~anndata.AnnData` objects
            cache: Whether to cache results or not

        Returns:
            An :class:`~mudata.MuData` object or a dict mapping the filename (object name) to the corresponding :class:`~anndata.AnnData` object and the columns
            that are obs only for each object
        """
        obs_only_all = {}
        if not return_mudata_object:
            anndata_dict = {}
        else:
            mudata = None
        for file in filename.iterdir():
            if file.is_file() and file.suffix in {".csv", ".tsv"}:
                # slice off the file suffix as this is not needed for identifier
                adata_identifier = file.name[:-4]
                index_col, col_obs_only = DataReader._extract_index_and_columns_obs_only(
                    adata_identifier, index_column, columns_obs_only
                )
                adata, single_adata_obs_only = DataReader.read_csv(
                    file, delimiter, index_col, col_obs_only, cache=cache
                )
                obs_only_all[adata_identifier] = single_adata_obs_only
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
            return anndata_dict, obs_only_all

    @staticmethod
    def read_csv(
        filename: Union[Path, Iterator[str]],
        delimiter: Optional[str] = ",",
        index_column: Optional[Union[str, int]] = None,
        columns_obs_only: Optional[List[Union[str]]] = None,
        cache: bool = False,
    ) -> Tuple[AnnData, Optional[List[str]]]:
        """Read `.csv` and `.tsv` file.

        Args:
            filename: File path to the csv file.
            delimiter: Delimiter separating the csv data within the file.
            index_column: Index or column name of the index column (obs)
            columns_obs_only: List of columns which only be stored in .obs, but not in X. Useful for free text annotations.
            cache: Whether the data should be written to cache or not

        Returns:
            An :class:`~anndata.AnnData` object and the column obs only for the object
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
        # in case the index column is misspelled or does not exist
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
        no_datetime_object_col = []
        for col in object_type_columns:
            try:
                pd.to_datetime(initial_df[col])
                # only add to column_obs_only if not present already to avoid duplicates
                if col not in columns_obs_only:
                    columns_obs_only.append(col)
            except (ValueError, TypeError):
                # we only need to replace NANs on non datetime, non numerical columns since datetime are obs only by default
                no_datetime_object_col.append(col)
        # writing to hd5a files requires non string to be empty in non numerical columns
        if cache:
            initial_df[no_datetime_object_col] = initial_df[no_datetime_object_col].fillna("")
            # temporary workaround needed; see https://github.com/theislab/anndata/issues/504 and https://github.com/theislab/anndata/issues/662
            # converting booleans to strings is needed for caching as writing to .h5ad files currently does not support writing boolean values
            bool_columns = {
                column_name: "str" for column_name in initial_df.columns if initial_df.dtypes[column_name] == "bool"
            }
            initial_df = initial_df.astype(bool_columns)
        # return the initial AnnData object
        return DataReader._df_to_anndata(initial_df, columns_obs_only), columns_obs_only

    @staticmethod
    def _read_from_cache_dir(cache_dir: Path) -> Dict[str, AnnData]:
        """Read AnnData objects from the cache directory"""
        adata_objects = {}
        # read each cache file in the cache directory and store it into a dict
        for cache_file in cache_dir.iterdir():
            if cache_file.name.endswith(".h5ad"):
                adata_objects[cache_file.stem] = DataReader._read_from_cache(cache_file)
        return adata_objects

    @staticmethod
    def _read_from_cache(path_cache: Path) -> AnnData:
        """Read AnnData object from cached file"""
        cached_adata = read_h5ad(path_cache)
        # type cast required; otherwise all values in X would be treated as strings
        cached_adata.X = cached_adata.X.astype("object")
        try:
            columns_obs_only = list(cached_adata.uns["cache_temp_obs_only"])
            del cached_adata.uns["cache_temp_obs_only"]
        # in case columns_obs_only has not been passed
        except KeyError:
            columns_obs_only = []
        # recreate the original AnnData object with the index column for obs and obs only columns
        cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)

        return cached_adata

    @staticmethod
    def _write_cache_dir(
        adata_objects: Dict[str, AnnData],
        path_cache: Path,
        columns_obs_only,
        index_column: Optional[Union[Dict[str, Union[str, int]]]],  # type ignore
    ) -> Dict[str, AnnData]:
        """Write multiple AnnData objects into a common cache directory keeping index column and columns_obs_only.

        Args:
            adata_objects: A dictionary with an identifier as key for each of the AnnData objects
            path_cache: Path to the cache directory
            columns_obs_only: Columns for obs only
            index_column: The index columns for each object (if any)

        Returns:
            A dict containing an unique identifier and an :class:`~anndata.AnnData` object for each file read
        """
        for identifier in adata_objects:
            # for each identifier (for the AnnData object), we need the index column and obs_only cols (if any) for reuse when reading cache
            index_col, cols_obs_only = DataReader._extract_index_and_columns_obs_only(
                identifier, index_column, columns_obs_only
            )
            adata_objects[identifier] = DataReader._write_cache(
                adata_objects[identifier], path_cache / (identifier + ".h5ad"), cols_obs_only
            )
        return adata_objects

    @staticmethod
    def _write_cache(
        raw_anndata: AnnData,
        path_cache: Path,
        columns_obs_only: Optional[List[Union[str]]],
    ) -> AnnData:
        """Write AnnData object to cache"""
        cached_adata = Encoder.encode(data=raw_anndata, autodetect=True)
        # temporary key that stores all column names that are obs only for this AnnData object
        cached_adata.uns["cache_temp_obs_only"] = columns_obs_only
        cached_adata.write(path_cache)
        cached_adata.X = cached_adata.X.astype("object")
        cached_adata = DataReader._decode_cached_adata(cached_adata, columns_obs_only)
        return cached_adata

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
                       index_columns = {"MyOtherFile":"MyOtherColumn1"}
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
                       index_columns = {"MyOtherFile":"MyOtherColumn1", "default": "MyColumn1"}
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
                       index_columns = {"MyFile":"MyColumn1"}
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
        if _index_column and _columns_obs_only and _index_column in _columns_obs_only:
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


class IndexNotFoundError(Exception):
    pass


class ColumnNotFoundError(Exception):
    pass


class BackupURLNotProvidedError(Exception):
    pass


class MudataCachingNotSupportedError(Exception):
    pass

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fhiry.parallel as fp
import numpy as np
import pandas as pd
from ehrdata._logger import logger
from rich import print

from ehrapy import ehrapy_settings, settings
from ehrapy._compat import function_future_warning
from ehrapy.anndata.anndata_ext import df_to_anndata
from ehrapy.data._dataloader import download, remove_archive_extension
from ehrapy.preprocessing._encoding import encode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anndata import AnnData


@function_future_warning("ep.io.read_csv", "ehrdata.io.read_csv")
def read_csv(
    dataset_path: Path | str,
    sep: str = ",",
    index_column: dict[str, str | int] | str | int | None = None,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
    columns_x_only: dict[str, list[str]] | list[str] | None = None,
    return_dfs: bool = False,
    cache: bool = False,
    download_dataset_name: str | None = None,
    backup_url: str | None = None,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
    **kwargs,
) -> AnnData | dict[str, AnnData]:
    """Reads or downloads a desired directory of csv/tsv files or a single csv/tsv file.

    Args:
        dataset_path: Path to the file or directory to read.
        sep: Separator in the file. Delegates to pandas.read_csv().
        index_column: The index column of obs. Usually the patient visit ID or the patient ID.
        columns_obs_only: These columns will be added to obs only and not X.
        columns_x_only: These columns will be added to X only and all remaining columns to obs.
                        Note that datetime columns will always be added to .obs though.
        return_dfs: Whether to return one or several Pandas DataFrames.
        cache: Whether to write to cache when reading or not.
        download_dataset_name: Name of the file or directory after download.
        backup_url: URL to download the data file(s) from, if the dataset is not yet on disk.
        archive_format: Whether the downloaded file is an archive.
        **kwargs: Passed to :func:`pandas.read_csv`

    Returns:
        An :class:`~anndata.AnnData` object or a dict with an identifier (the filename, without extension)
        for each :class:`~anndata.AnnData` object in the dict

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.io.read_csv("myfile.csv")
    """
    function_future_warning("ep.io.read_csv", "ehrdata.io.read_csv")
    _check_columns_only_params(columns_obs_only, columns_x_only)
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        dataset_path = _get_non_existing_files(dataset_path, download_dataset_name, backup_url, archive_format)

    adata = _read_csv(
        file_path=dataset_path,
        sep=sep,
        index_column=index_column,
        columns_obs_only=columns_obs_only,
        columns_x_only=columns_x_only,
        return_dfs=return_dfs,
        cache=cache,
        **kwargs,
    )
    return adata


def _read_csv(
    file_path: Path,
    sep: str,
    index_column: dict[str, str | int] | str | int | None,
    columns_obs_only: dict[str, list[str]] | list[str] | None,
    columns_x_only: dict[str, list[str]] | list[str] | None,
    return_dfs: bool = False,
    cache: bool = False,
    **kwargs,
) -> AnnData | dict[str, AnnData]:
    """Internal interface of the read_csv method."""
    if cache and return_dfs:
        raise CachingNotSupported("Caching is currently not supported for Pandas DataFrame objects.")
    if return_dfs and (columns_x_only or columns_obs_only):
        raise Warning(
            "Parameters columns_x_only and columns_obs_only are not supported when returning Pandas DataFrames."
        )

    path_cache = settings.cachedir / file_path
    # reading from (cache) file is separated in the read_h5ad function
    if cache and (path_cache.is_dir() or path_cache.is_file()):
        raise CacheExistsException(
            f"{path_cache} already exists. Use the read_h5ad function instead to read from cache!"
        )

    # If the the file path is a directory, assume it is a dataset with multiple files
    elif file_path.is_dir():
        return _read_from_directory(
            file_path,
            cache,
            path_cache,
            extension=sep,
            index_column=index_column,
            columns_obs_only=columns_obs_only,
            columns_x_only=columns_x_only,
            return_dfs=return_dfs,
        )
    # input is a single file
    else:
        adata, columns_obs_only = _do_read_csv(
            file_path,
            sep,
            index_column,  # type: ignore
            columns_obs_only,  # type: ignore
            columns_x_only,  # type: ignore
            cache,
            **kwargs,
        )
        # cache results if desired
        if cache:
            if not path_cache.parent.is_dir():
                path_cache.parent.mkdir(parents=True)
            return _write_cache(adata, path_cache, columns_obs_only)  # type: ignore
        return adata


@function_future_warning("ep.io.read_h5ad", "ehrdata.io.read_h5ad")
def read_h5ad(
    dataset_path: Path | str,
    backup_url: str | None = None,
    download_dataset_name: str | None = None,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
) -> AnnData | dict[str, AnnData]:
    """Reads or downloads a desired directory of h5ad files or a single h5ad file.

    Args:
        dataset_path: Path to the file or directory to read.
        backup_url: URL to download the data file(s) from if not yet existing.
        download_dataset_name: Name of the file or directory in case the dataset is downloaded.
        archive_format: Whether the downloaded file is an archive.

    Returns:
        An :class:`~anndata.AnnData` object or a dict with an identifier (the filename, without extension)
        for each :class:`~anndata.AnnData` object in the dict.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ed.dt.mimic_2()
        >>> ep.io.write("mimic_2.h5ad", adata)
        >>> adata_2 = ep.io.read_h5ad("mimic_2.h5ad")
    """
    function_future_warning("ep.io.read_h5ad", "ehrdata.io.read_h5ad")
    file_path: Path = Path(dataset_path)
    if not file_path.exists():
        file_path = _get_non_existing_files(file_path, download_dataset_name, backup_url, archive_format=archive_format)

    if file_path.is_dir():
        adata = _read_from_directory(file_path, False, None, "h5ad")
    else:
        adata = _do_read_h5ad(file_path)

    return adata


def _read_from_directory(
    file_path: Path,
    cache: bool,
    path_cache_dir: Path | None,
    extension: str,
    index_column: dict[str, str | int] | str | int | None = None,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
    columns_x_only: dict[str, list[str]] | list[str] | None = None,
    return_dfs: bool = False,
) -> dict[str, AnnData] | dict[str, pd.DataFrame]:
    """Parse AnnData objects or Pandas DataFrames from a directory containing the data files."""
    if return_dfs:
        dfs = _read_multiple_csv(file_path, sep=extension, return_dfs=True)
        return dfs  # type: ignore
    if extension in {",", "\t"}:
        adata_objects, columns_obs_only = _read_multiple_csv(  # type: ignore
            file_path,
            sep=extension,
            index_column=index_column,
            columns_obs_only=columns_obs_only,
            columns_x_only=columns_x_only,
            return_dfs=False,
        )
        # cache results
        if cache:
            if not path_cache_dir.parent.is_dir():
                path_cache_dir.parent.mkdir(parents=True)
            path_cache_dir.mkdir()
            return _write_cache_dir(adata_objects, path_cache_dir, columns_obs_only, index_column)  # type: ignore
        return adata_objects  # type: ignore
    elif extension == "h5ad":
        return _read_multiple_h5ad(file_path)
    else:
        raise NotImplementedError(f"Reading from directory with .{extension} files is not implemented yet!")


def _read_multiple_csv(
    file_path: Path,
    sep: str,
    index_column: dict[str, str | int] | str | int | None = None,
    columns_obs_only: dict[str, list[str]] | list[str] | None = None,
    columns_x_only: dict[str, list[str]] | list[str] | None = None,
    return_dfs: bool = False,
    cache: bool = False,
    **kwargs,
) -> tuple[dict[str, AnnData], dict[str, list[str] | None]] | dict[str, pd.DataFrame]:
    """Read a dataset containing multiple .csv/.tsv files.

    Args:
        file_path: File path to the directory containing multiple .csv/.tsv files.
        sep: Separator in the file. Delegates to pandas.read_csv().
        index_column: Column names of the index columns for obs
        columns_obs_only: List of columns per file (AnnData object) which should only be stored in .obs, but not in X.
                          Useful for free text annotations.
        columns_x_only: List of columns per file (AnnData object) which should only be stored in .X, but not in obs.
                        Datetime columns will be added to .obs regardless.
        return_dfs: When set to True, return a dictionary of Pandas DataFrames.
        cache: Whether to cache results or not.
        kwargs: Keyword arguments for Pandas `read_csv`.

    Returns:
        A Dict mapping the filename (object name) to the corresponding :class:`~anndata.AnnData` object and the columns
        that are obs only for each object.
    """
    obs_only_all = {}
    if return_dfs:
        df_dict: dict[str, pd.DataFrame] = {}
    else:
        anndata_dict = {}

    for file in file_path.iterdir():
        if file.is_file() and file.suffix in {".csv", ".tsv"}:
            # slice off the file suffix .csv or .tsv for a clean file name
            file_identifier = file.name[:-4]
            if return_dfs:
                df = pd.read_csv(file, sep=sep, **kwargs)
                df_dict[file_identifier] = df
                continue

            index_col, col_obs_only, col_x_only = _extract_index_and_columns_obs_only(
                file_identifier, index_column, columns_obs_only, columns_x_only
            )
            adata, single_adata_obs_only = _do_read_csv(file, sep, index_col, col_obs_only, col_x_only, cache=cache)
            obs_only_all[file_identifier] = single_adata_obs_only
            # obs indices have to be unique otherwise updating and working with the object will fail
            if index_col:
                adata.obs_names_make_unique()

            anndata_dict[file_identifier] = adata
    if return_dfs:
        return df_dict
    else:
        return anndata_dict, obs_only_all


def _do_read_csv(
    file_path: Path | Iterator[str],
    sep: str | None = ",",
    index_column: str | int | None = None,
    columns_obs_only: list[str] | None = None,
    columns_x_only: list[str] | None = None,
    cache: bool = False,
    **kwargs,
) -> tuple[AnnData, list[str] | None]:
    """Read `.csv` and `.tsv` file.

    Args:
        file_path: File path to the csv file.
        sep: Separator in the file. Delegates to pandas.read_csv().
        index_column: Index or column name of the index column (obs)
        columns_obs_only: List of columns which only be stored in .obs, but not in X. Useful for free text annotations.
        columns_x_only: List of columns which only be stored in X, but not in .obs.
        cache: Whether the data should be written to cache or not.
        **kwargs: Passed to :func:`pandas.read_csv`

    Returns:
        An :class:`~anndata.AnnData` object and the column obs only for the object.
    """
    try:
        if index_column and columns_obs_only and index_column in columns_obs_only:
            logger.warning(
                f"Index column '{index_column}' is also used as a column "
                f"for obs only. Using default indices instead and moving {index_column} to column_obs_only."
            )
            index_column = None
        initial_df = pd.read_csv(file_path, sep=sep, index_col=index_column, **kwargs)
    # in case the index column is misspelled or does not exist
    except ValueError:
        raise IndexNotFoundError(
            f"Could not create AnnData object while reading file {file_path} . Does index_column named {index_column} "
            f"exist in {file_path}?"
        ) from None

    initial_df, columns_obs_only = _prepare_dataframe(initial_df, columns_obs_only, columns_x_only, cache)

    return df_to_anndata(initial_df, columns_obs_only), columns_obs_only


def _read_multiple_h5ad(
    file_path: Path,
) -> dict[str, AnnData]:
    """Read a dataset containing multiple .h5ad files.

    Args:
        file_path: File path to the directory containing multiple .csv/.tsv files.

    Returns:
        A dict mapping the filename (object name) to the corresponding :class:`~anndata.AnnData` object
    """
    anndata_dict = {}
    for file in file_path.iterdir():
        if file.is_file() and file.suffix == ".h5ad":
            # slice off the file suffix .h5ad
            adata_identifier = file.name[:-5]
            adata = _do_read_h5ad(file)
            anndata_dict[adata_identifier] = adata
    return anndata_dict


def _do_read_h5ad(file_path: Path | Iterator[str]) -> AnnData:
    """Read from a h5ad file.

    Args:
        file_path: Path to the h5ad file.

    Returns:
        An AnnData object.
    """
    import anndata as ad

    adata = ad.read_h5ad(file_path)
    if "ehrapy_dummy_encoding" in adata.uns.keys():
        # if dummy encoding was needed, the original dtype of X could not be numerical, so cast it to object
        adata.X = adata.X.astype("object")
        decoded_adata = _decode_cached_adata(adata, list(adata.uns["columns_obs_only"]))
        return decoded_adata
    return adata


@function_future_warning("ep.io.read_fhir")
def read_fhir(
    dataset_path: str,
    format: Literal["json", "ndjson"] = "json",
    columns_obs_only: list[str] | None = None,
    columns_x_only: list[str] | None = None,
    return_df: bool = False,
    cache: bool = False,
    backup_url: str | None = None,
    index_column: str | int | None = None,
    download_dataset_name: str | None = None,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
) -> pd.DataFrame | AnnData:
    """Reads one or multiple FHIR files using fhiry.

    Uses https://github.com/dermatologist/fhiry to read the FHIR file into a Pandas DataFrame
    which is subsequently transformed into an AnnData object.

    Be aware that FHIR data can be nested and return lists or dictionaries as values.
    In such cases, one can either:
    1. Transform the data into an awkward array and flatten it when needed.
    2. Extract values from all lists and dictionaries to store single values in the fields.
    3. Remove all lists and dictionaries. Only do this if the information is not relevant to you.

    Args:
        dataset_path: Path to one or multiple FHIR files.
        format: The file format of the FHIR data. One of 'json' or 'ndjson'.
        columns_obs_only: These columns will be added to obs only and not X.
        columns_x_only: These columns will be added to X only and all remaining columns to obs.
                        Note that datetime columns will always be added to .obs though.
        return_df: Whether to return one or several Pandas DataFrames.
        cache: Whether to write to cache when reading or not.
        backup_url: URL to download the data file(s) from if not yet existing.
        index_column: The index column for the generated object. Usually the patient or visit ID.
        download_dataset_name: Name of the file or directory in case the dataset is downloaded.
        archive_format: Whether the downloaded file is an archive.

    Returns:
        A Pandas DataFrame or AnnData object of the read in FHIR file(s).

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.io.read_fhir("/path/to/fhir/resources")

        Be aware that most FHIR datasets have nested data that might need to be removed.
        In such cases consider working with DataFrames.

        >>> df = ep.io.read_fhir("/path/to/fhir/resources", return_df=True)
        >>> df.drop(
        ...     columns=[col for col in df.columns if any(isinstance(x, (list, dict)) for x in df[col].dropna())],
        ...     inplace=True,
        ... )
        >>> df.drop(columns=df.columns[df.isna().all()], inplace=True)

    """
    function_future_warning("ep.io.read_fhir")
    _check_columns_only_params(columns_obs_only, columns_x_only)
    file_path: Path = Path(dataset_path)
    if not file_path.exists():
        file_path = _get_non_existing_files(file_path, download_dataset_name, backup_url, archive_format)

    adata = _read_fhir(
        file_path=str(file_path.resolve()),
        format=format,
        index_column=index_column,
        columns_obs_only=columns_obs_only,
        columns_x_only=columns_x_only,
        return_df=return_df,
        cache=cache,
    )
    return adata


def _read_fhir(
    file_path: str,
    format: Literal["json", "ndjson"],
    index_column: dict[str, str | int] | str | int | None,
    columns_obs_only: list[str] | None,
    columns_x_only: list[str] | None,
    return_df: bool = False,
    cache: bool = False,
) -> AnnData | dict[str, AnnData]:
    """Internal interface of the read_fhir method."""
    if cache and return_df:
        raise CachingNotSupported("Caching is currently not supported for or Pandas DataFrame objects.")
    if return_df and (columns_x_only or columns_obs_only):
        raise Warning(
            "Parameters columns_x_only and columns_obs_only are not supported when returning Pandas DataFrames."
        )
    path_cache = settings.cachedir / file_path
    if cache and (path_cache.is_dir() or path_cache.is_file()):
        raise CacheExistsException(
            f"{path_cache} already exists. Use the read_h5ad function instead to read from cache!"
        )
    if format == "json":
        df = fp.process(file_path)
    elif format == "ndjson":
        df = fp.ndjson(file_path)
    else:
        raise ValueError("Only folders containing json and ndjson in FHIR format are supported.")

    df, columns_obs_only = _prepare_dataframe(df, columns_obs_only, columns_x_only, cache)
    if index_column:
        df.set_index(index_column)

    if return_df:
        return df
    else:
        adata = df_to_anndata(df, columns_obs_only)

    if cache:
        if not path_cache.parent.is_dir():
            path_cache.parent.mkdir(parents=True)
        return _write_cache(adata, path_cache, columns_obs_only)  # type: ignore

    return adata


def _get_non_existing_files(
    dataset_path: Path,
    download_dataset_name: str,
    backup_url: str,
    archive_format: Literal["zip", "tar", "tar.gz", "tgz"] = None,
) -> Path:
    """Handle non-existing files or directories by trying to download from a backup_url and moving them in the correct directory.

    Returns:
        The file or directory path of the downloaded content.
    """
    if backup_url is None and not dataset_path.exists():
        raise ValueError(
            f"File or directory {dataset_path} does not exist and no backup_url was provided.\n"
            f"Please provide a backup_url or check whether path is spelled correctly."
        )
    logger.info("Path or dataset does not yet exist. Attempting to download...")
    download(
        backup_url,
        output_file_name=download_dataset_name,
        output_path=ehrapy_settings.datasetdir,
        archive_format=archive_format,
    )

    if archive_format:
        dataset_path = remove_archive_extension(dataset_path)

    return dataset_path


def _read_from_cache_dir(cache_dir: Path) -> dict[str, AnnData]:
    """Read AnnData objects from the cache directory."""
    adata_objects = {}
    # read each cache file in the cache directory and store it into a dict
    for cache_file in cache_dir.iterdir():
        if cache_file.name.endswith(".h5ad"):
            adata_objects[cache_file.stem] = _read_from_cache(cache_file)
    return adata_objects


def _read_from_cache(path_cache: Path) -> AnnData:
    """Read AnnData object from cached file."""
    from anndata.io import read_h5ad

    cached_adata = read_h5ad(path_cache)
    # type cast required when dealing with non-numerical data; otherwise all values in X would be treated as strings
    if not np.issubdtype(cached_adata.X.dtype, np.number):
        cached_adata.X = cached_adata.X.astype("object")
    try:
        columns_obs_only = list(cached_adata.uns["columns_obs_only"])
        del cached_adata.uns["columns_obs_only"]
    # in case columns_obs_only has not been passed
    except KeyError:
        columns_obs_only = []
    # recreate the original AnnData object with the index column for obs and obs only columns
    cached_adata = _decode_cached_adata(cached_adata, columns_obs_only)

    return cached_adata


def _write_cache_dir(
    adata_objects: dict[str, AnnData],
    path_cache: Path,
    columns_obs_only,
    index_column: dict[str, str | int] | None,  # type ignore
) -> dict[str, AnnData]:
    """Write multiple AnnData objects into a common cache directory keeping index column and columns_obs_only.

    Args:
        adata_objects: A dictionary with an identifier as key for each of the AnnData objects.
        path_cache: Path to the cache directory.
        columns_obs_only: Columns for obs only.
        index_column: The index columns for each object (if any).

    Returns:
        A dict containing a unique identifier and an :class:`~anndata.AnnData` object for each file read.
    """
    for identifier in adata_objects:
        # for each identifier (for the AnnData object), we need the index column and obs_only cols (if any) for reuse when reading cache
        index_col, cols_obs_only, _ = _extract_index_and_columns_obs_only(identifier, index_column, columns_obs_only)
        adata_objects[identifier] = _write_cache(
            adata_objects[identifier], path_cache / (identifier + ".h5ad"), cols_obs_only
        )
    return adata_objects


def _write_cache(
    raw_anndata: AnnData,
    path_cache: Path,
    columns_obs_only: list[str] | None,
) -> AnnData:
    """Write AnnData object to cache."""
    original_x_dtype = raw_anndata.X.dtype
    if not np.issubdtype(original_x_dtype, np.number):
        cached_adata = encode(adata=raw_anndata, autodetect=True)
    else:
        cached_adata = raw_anndata
    # temporary key that stores all column names that are obs only for this AnnData object
    cached_adata.uns["columns_obs_only"] = columns_obs_only
    cached_adata.uns["ehrapy_dummy_encoding"] = True
    # append correct file suffix
    if not path_cache.suffix == ".h5ad":
        if path_cache.suffix in {".tsv", ".csv"}:
            path_cache = Path(str(path_cache)[:-4] + ".h5ad")
        else:
            path_cache = Path(str(path_cache) + ".h5ad")
    cached_adata.write(path_cache)
    # preserve original dtype of X (either numerical or object)
    cached_adata.X = cached_adata.X.astype(original_x_dtype)
    cached_adata = _decode_cached_adata(cached_adata, columns_obs_only)
    return cached_adata


def _prepare_dataframe(initial_df: pd.DataFrame, columns_obs_only, columns_x_only=None, cache=False):
    """Prepares the dataframe to be casted into an AnnData object.

    Datetime columns will be detected and added to columns_obs_only.

    Returns:
         The initially parsed dataframe and an updated list of columns_obs_only.
    """
    # when passing columns x only, simply handle the (asymmetric) difference to be obs only and everything else is kept in X
    if columns_x_only:
        columns_obs_only = list(set(initial_df.columns) - set(columns_x_only))
    # get all object dtype columns
    object_type_columns = [col_name for col_name in initial_df.columns if initial_df[col_name].dtype == "object"]
    # if columns_obs_only is None, initialize it as datetime columns need to be included here
    if not columns_obs_only:
        columns_obs_only = []
    no_datetime_object_col = []
    for col in object_type_columns:
        try:
            pd.to_datetime(initial_df[col], format="mixed", utc=True)
            # only add to column_obs_only if not present already to avoid duplicates
            if col not in columns_obs_only:
                columns_obs_only.append(col)
        except (ValueError, TypeError):
            # we only need to replace NANs on non datetime, non numerical columns since datetime are obs only by default
            no_datetime_object_col.append(col)
    # writing to hd5a files requires non string to be empty in non numerical columns
    if cache:
        # TODO remove this when anndata 0.8.0 is released
        initial_df[no_datetime_object_col] = initial_df[no_datetime_object_col].fillna("")
        # temporary workaround needed; see https://github.com/theislab/anndata/issues/504 and https://github.com/theislab/anndata/issues/662
        # converting booleans to strings is needed for caching as writing to .h5ad files currently does not support writing boolean values
        bool_columns = {
            column_name: "str" for column_name in initial_df.columns if initial_df.dtypes[column_name] == "bool"
        }
        initial_df = initial_df.astype(bool_columns)
    return initial_df, columns_obs_only


def _decode_cached_adata(adata: AnnData, column_obs_only: list[str]) -> AnnData:
    """Decode the label encoding of initial AnnData object.

    Args:
        adata: The label encoded AnnData object.
        column_obs_only: The columns, that should be kept in obs.

    Returns:
        The decoded, initial AnnData object.
    """
    var_names = list(adata.var_names)
    # for each encoded categorical, replace its encoded values with its original values in X
    for idx, var_name in enumerate(var_names):
        if not var_name.startswith("ehrapycat_"):
            break
        value_name = var_name[10:]
        if value_name not in adata.obs.keys():
            raise ValueError(f"Unencoded values for feature '{value_name}' not found in obs!")
        original_values = adata.obs[value_name]
        adata.X[:, idx] = original_values
        # update var name per categorical
        var_names[idx] = value_name
    # drop all columns, that are not obs only in obs
    if column_obs_only:
        adata.obs = adata.obs[column_obs_only]
    else:
        adata.obs = pd.DataFrame(index=adata.obs.index)
    # set the new var names (unencoded ones)
    adata.var.index = var_names
    adata.layers["original"] = adata.X.copy()
    # reset uns
    adata.uns = OrderedDict()

    return adata


def _extract_index_and_columns_obs_only(identifier: str, index_columns, columns_obs_only, columns_x_only=None):
    """Extract the index column (if any) and the columns, for obs only (if any) from the given user input.

    For each file, `index_columns` and `columns_obs_only` can provide three cases:
        1.) The filename (thus the identifier) is not present as a key and no default key is provided or one or both dicts are empty:
            --> No index column will be set and/or no columns are obs only (based on user input)

        2.) The filename (thus the identifier) is not present as a key, but default key is provided
            --> The index column will be set and/or columns will be obs only according to the default key

        3.) The filename is present as a key
            --> The index column will be set and/or columns are obs only according to its value

    Args:
        identifier: The name of the
        index_columns: Index columns
        columns_obs_only: Columns for obs only
        columns_x_only: Columns which are only in X.

    Returns:
        Index column (if any) and columns obs only (if any) for this specific AnnData object.
    """
    _index_column = None
    _columns_obs_only = None
    _columns_x_only = None
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

    # get columns x only (if any)
    if columns_x_only and identifier in columns_x_only.keys():
        _columns_x_only = columns_x_only[identifier]
    elif columns_x_only and "default" in columns_x_only.keys():
        _columns_x_only = columns_x_only["default"]

    # if index column is also found in column_obs_only or x_only, use default indices instead and only move it to obs/X, but warn the user
    if (_index_column and _columns_obs_only or _index_column and _columns_x_only) and (
        _index_column in _columns_obs_only or _index_column in _columns_x_only
    ):
        logger.warning(
            f"Index column '{_index_column}' for file '{identifier}' is also used as a column "
            f"for obs or X only. Using default indices instead and moving '{_index_column}' to obs/X!."
        )
        _index_column = None

    return _index_column, _columns_obs_only, _columns_x_only


def _check_columns_only_params(
    obs_only: dict[str, list[str]] | list[str] | None, x_only: dict[str, list[str]] | list[str] | None
) -> None:
    """Check whether columns_obs_only and columns_x_only are passed exclusively.

    For a single AnnData object (thus parameters being a list of strings) it's not desirable to pass both, obs_only and x_only.
    For multiple AnnData objects (thus the parameters being dicts of string keys with a list value), it is possible to pass both. But the keys
    (unique identifiers of the AnData objects, basically its names) should share no common identifier,
    thus a single AnnData object is either in x_only OR obs_only, but not in both.
    """
    if not obs_only or not x_only:
        return
    if obs_only and x_only and isinstance(obs_only, list):
        raise ValueError(
            "Can not use columns_obs_only together with columns_x_only with a single AnnData object. "
            "At least one has to be None!"
        )
    else:
        common_keys = obs_only.keys() & x_only.keys()  # type: ignore
        if common_keys:
            raise ValueError(
                "Can not use columns_obs_only together with columns_x_only for a single AnnData object. "
                "The following anndata identifiers where found"
                f"in both: {','.join(key for key in common_keys)}!"
            )


class IndexNotFoundError(Exception):
    pass


class CachingNotSupported(Exception):
    pass


class ExtensionMissingError(Exception):
    pass


class CacheExistsException(Exception):
    pass

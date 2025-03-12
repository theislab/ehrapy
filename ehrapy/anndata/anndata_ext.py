from __future__ import annotations

import random
from collections import OrderedDict
from string import ascii_letters
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from lamin_utils import logger
from scipy import sparse
from scipy.sparse import issparse

from ehrapy.anndata import check_feature_types
from ehrapy.anndata._constants import FEATURE_TYPE_KEY, NUMERIC_TAG

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


def df_to_anndata(
    df: pd.DataFrame, columns_obs_only: list[str] | None = None, index_column: str | None = None
) -> AnnData:
    """Transform a given Pandas DataFrame into an AnnData object.

    Note that columns containing boolean values (either 0/1 or T(t)rue/F(f)alse)
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical values.

    Args:
        df: The pandas dataframe to be transformed.
        columns_obs_only: An optional list of column names that should belong to obs only and not X.
        index_column: The index column of obs. This can be either a column name (or its numerical index in the DataFrame) or the index of the dataframe.

    Returns:
        An AnnData object created from the given Pandas DataFrame.

    Examples:
        >>> import ehrapy as ep
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "patient_id": ["0", "1", "2", "3", "4"],
        ...         "age": [65, 72, 58, 78, 82],
        ...         "sex": ["M", "F", "F", "M", "F"],
        ...     }
        ... )
        >>> adata = ep.ad.df_to_anndata(df, index_column="patient_id")
    """
    # Check and handle the overlap of index_column in columns_obs_only
    if index_column is not None:
        if isinstance(index_column, int):
            if index_column >= len(df.columns):
                raise IndexError("index_column integer index is out of bounds.")
            index_column = df.columns[index_column]
        if not df.index.name or df.index.name != index_column:
            if index_column in df.columns:
                df.set_index(index_column, inplace=True)
            else:
                raise ValueError(f"Column {index_column} not found in DataFrame.")

    # Now handle columns_obs_only with consideration of the new index
    if columns_obs_only:
        if index_column in columns_obs_only:
            columns_obs_only.remove(index_column)
        missing_cols = [col for col in columns_obs_only if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} specified in columns_obs_only are not in the DataFrame.")
        obs = df.loc[:, columns_obs_only].copy()
        df.drop(columns=columns_obs_only, inplace=True, errors="ignore")
    else:
        obs = pd.DataFrame(index=df.index)

    for col in obs.columns:
        if obs[col].dtype == "bool":
            obs[col] = obs[col].astype(bool)
        elif obs[col].dtype == "object":
            obs[col] = obs[col].astype("category")

    # Prepare the AnnData object
    X = df.to_numpy(copy=True)
    obs.index = obs.index.astype(str)
    var = pd.DataFrame(index=df.columns)
    var.index = var.index.astype(str)
    uns = OrderedDict()  # type: ignore

    # Handle dtype of X based on presence of numerical columns only
    all_numeric = df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    X = X.astype(np.float32 if all_numeric else object)

    adata = AnnData(X=X, obs=obs, var=var, uns=uns, layers={"original": X.copy()})
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    return adata


def anndata_to_df(
    adata: AnnData,
    layer: str = None,
    obs_cols: Iterable[str] | str | None = None,
    var_cols: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    """Transform an AnnData object to a Pandas DataFrame.

    Args:
        adata: The AnnData object to be transformed into a pandas DataFrame
        layer: The layer to access the values of. If not specified, it uses the `X` matrix.
        obs_cols: The columns of `obs` to add to the DataFrame.
        var_cols: The columns of `var` to fetch values from.

    Returns:
        The AnnData object as a pandas DataFrame

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> df = ep.ad.anndata_to_df(adata)
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if issparse(X):  # pragma: no cover
        X = X.toarray()

    df = pd.DataFrame(X, columns=list(adata.var_names))
    if obs_cols:
        if len(adata.obs.columns) == 0:
            raise ValueError("Cannot slice columns from empty obs!")
        if isinstance(obs_cols, str):
            obs_cols = list(obs_cols)
        if isinstance(obs_cols, list):  # pragma: no cover
            obs_slice = adata.obs[obs_cols]
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)
    if var_cols:
        if len(adata.var.columns) == 0:
            raise ValueError("Cannot slice columns from empty var!")
        if isinstance(var_cols, str):
            var_cols = list(var_cols)
        if isinstance(var_cols, list):
            var_slice = adata.var[var_cols]
        # reset index needed since we slice all or at least some columns from var DataFrame
        var_slice = var_slice.reset_index(drop=True)
        df = pd.concat([df, var_slice], axis=1)

    return df


def move_to_obs(adata: AnnData, to_obs: list[str] | str, copy_obs: bool = False) -> AnnData:
    """Move inplace or copy features from X to obs.

    Note that columns containing boolean values (either 0/1 or True(true)/False(false))
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical.

    Args:
        adata: The AnnData object
        to_obs: The columns to move to obs
        copy_obs: The values are copied to obs (and therefore kept in X) instead of moved completely

    Returns:
        The original AnnData object with moved or copied columns from X to obs

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.move_to_obs(adata, ["age"], copy_obs=False)
    """
    if isinstance(to_obs, str):  # pragma: no cover
        to_obs = [to_obs]

    # don't allow moving encoded columns as this could lead to inconsistent data in X and obs
    if any(column.startswith("ehrapycat") for column in to_obs):
        raise ValueError(
            "Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!"
        )

    if not all(elem in adata.var_names.values for elem in to_obs):
        raise ValueError(
            f"Columns `{[col for col in to_obs if col not in adata.var_names.values]}` are not in var_names."
        )

    cols_to_obs_indices = adata.var_names.isin(to_obs)

    num_set = _get_var_indices_for_type(adata, NUMERIC_TAG)
    var_num = list(set(to_obs) & set(num_set))

    if copy_obs:
        cols_to_obs = adata[:, cols_to_obs_indices].to_df()
        adata.obs = adata.obs.join(cols_to_obs)
        adata.obs[var_num] = adata.obs[var_num].apply(pd.to_numeric, downcast="float")

        adata.obs = _cast_obs_columns(adata.obs)
    else:
        df = adata[:, cols_to_obs_indices].to_df()
        adata._inplace_subset_var(~cols_to_obs_indices)
        adata.obs = adata.obs.join(df)
        adata.obs[var_num] = adata.obs[var_num].apply(pd.to_numeric, downcast="float")
        adata.obs = _cast_obs_columns(adata.obs)

    return adata


@check_feature_types
def _get_var_indices_for_type(adata: AnnData, tag: str) -> list[str]:
    """Get indices of columns in var for a given tag.

    Args:
        adata: The AnnData object
        tag: The tag to search for, should be one of 'CATEGORIGAL_TAG', 'NUMERIC_TAG', 'DATE_TAG'

    Returns:
        List of numeric columns
    """
    return adata.var_names[adata.var[FEATURE_TYPE_KEY] == tag].tolist()


def move_to_x(adata: AnnData, to_x: list[str] | str, copy_x: bool = False) -> AnnData:
    """Move features from obs to X inplace.

    Args:
        adata: The AnnData object
        to_x: The columns to move to X
        copy_x: The values are copied to X (and therefore kept in obs) instead of moved completely

    Returns:
        A new AnnData object with moved columns from obs to X. This should not be used for datetime columns currently.

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.ad.move_to_obs(adata, ["age"], copy_obs=False)
        >>> new_adata = ep.ad.move_to_x(adata, ["age"])
    """
    if isinstance(to_x, str):  # pragma: no cover
        to_x = [to_x]

    if not all(elem in adata.obs.columns.values for elem in to_x):
        raise ValueError(f"Columns `{[col for col in to_x if col not in adata.obs.columns.values]}` are not in obs.")

    cols_present_in_x = []
    cols_not_in_x = []

    for col in to_x:
        if col in set(adata.var_names):
            cols_present_in_x.append(col)
        else:
            cols_not_in_x.append(col)

    if cols_present_in_x:
        logger.warn(f"Columns `{cols_present_in_x}` are already in X. Skipped moving `{cols_present_in_x}` to X. ")

    if cols_not_in_x:
        new_adata = concat([adata, AnnData(adata.obs[cols_not_in_x])], axis=1)
        if copy_x:
            new_adata.obs = adata.obs
        else:
            new_adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.isin(cols_not_in_x)]]

        # AnnData's concat discards var if they don't match in their keys, so we need to create a new var
        created_var = pd.DataFrame(index=cols_not_in_x)
        new_adata.var = pd.concat([adata.var, created_var], axis=0)
    else:
        new_adata = adata

    return new_adata


def get_numerical_column_indices(
    adata: AnnData, layer: str | None = None, column_indices: Iterable[int] | None = None
) -> list[int]:
    mtx = adata.X if layer is None else adata[layer]
    indices = (
        list(range(mtx.shape[1])) if column_indices is None else [i for i in column_indices if i < mtx.shape[1] - 1]
    )
    non_numerical_indices = []
    for i in indices:
        # The astype("float64") call will throw only if the feature’s data type cannot be cast to float64, meaning in
        # practice it contains non-numeric values. Consequently, it won’t throw if the values are numeric but stored
        # as an "object" dtype, as astype("float64") can successfully convert them to floats.
        try:
            mtx[::, i].astype("float64")
        except ValueError:
            non_numerical_indices.append(i)

    return [idx for idx in indices if idx not in non_numerical_indices]


def get_fully_imputed_column_indices(
    adata: AnnData, layer: str | None = None, column_indices: Iterable[int] | None = None
) -> list[int]:
    mtx = adata.X if layer is None else adata.layers[layer]

    indices = range(mtx.shape[1]) if column_indices is None else [i for i in column_indices if i < mtx.shape[1]]
    mask = ~np.isnan(mtx[:, indices]).any(axis=0)

    return np.array(indices)[mask].tolist()


def get_column_indices(adata: AnnData, col_names: str | Iterable[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object.
        col_names: Column names to extract the indices for.

    Returns:
        List of column indices.
    """
    col_names = [col_names] if isinstance(col_names, str) else col_names
    mask = np.isin(adata.var_names, col_names)
    indices = np.where(mask)[0].tolist()

    return indices


def _assert_encoded(adata: AnnData):
    try:
        assert np.issubdtype(adata.X.dtype, np.number)
    except AssertionError:
        raise NotEncodedError("The AnnData object has not yet been encoded.") from AssertionError


@check_feature_types
def get_numeric_vars(adata: AnnData) -> list[str]:
    """Fetches the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object

    Returns:
        List of column numeric column names
    """
    _assert_encoded(adata)

    return _get_var_indices_for_type(adata, NUMERIC_TAG)


def assert_numeric_vars(adata: AnnData, vars: Sequence[str]):
    num_vars = get_numeric_vars(adata)

    try:
        assert set(vars) <= set(num_vars)
    except AssertionError:
        raise ValueError("Some selected vars are not numeric") from None


def set_numeric_vars(
    adata: AnnData, values: np.ndarray, vars: Sequence[str] | None = None, copy: bool = False
) -> AnnData | None:
    """Sets the numeric values in given column names in X.

    Args:
        adata: :class:`~anndata.AnnData` object
        values: Matrix containing the replacement values
        vars: List of names of the numeric variables to replace. If `None` they will be detected using :func:`~ehrapy.preprocessing.get_numeric_vars`.
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with updated X
    """
    _assert_encoded(adata)

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Values must be numeric (current dtype is {values.dtype})")

    n_values = values.shape[1]

    if n_values != len(vars):
        raise ValueError(f"Number of values ({n_values}) does not match number of vars ({len(vars)})")

    if copy:
        adata = adata.copy()

    vars_idx = get_column_indices(adata, vars)

    # if e.g. adata.X is of type int64, and values of dtype float64, the floats will be casted to int
    adata.X = adata.X.astype(values.dtype)

    adata.X[:, vars_idx] = values

    return adata


def _detect_binary_columns(df: pd.DataFrame, numerical_columns: list[str]) -> list[str]:
    """Detect all columns that contain only 0 and 1 (besides NaNs).

    Args:
        df: The dataframe to check.
        numerical_columns: All numerical columns of the dataframe.

    Returns:
            List of column names that are binary (containing only 0 and 1 (+NaNs))
    """
    binary_columns = []
    for column in numerical_columns:
        # checking for float and int as well as NaNs (this is safe since checked columns are numericals only)
        # only columns that contain at least one 0 and one 1 are counted as binary (or 0.0/1.0)
        if df[column].isin([0.0, 1.0, np.nan, 0, 1]).all() and df[column].nunique() == 2:
            binary_columns.append(column)

    return binary_columns


def _cast_obs_columns(obs: pd.DataFrame) -> pd.DataFrame:
    """Cast non numerical obs columns to either category or bool.
    Args:
        obs: Obs of an AnnData object

    Returns:
        The type casted obs.
    """
    # only cast non numerical columns
    object_columns = list(obs.select_dtypes(exclude=["number", "category", "bool"]).columns)
    # type cast each non-numerical column to either bool (if possible) or category else
    obs[object_columns] = obs[object_columns].apply(
        lambda obs_name: obs_name.astype("category")
        if not set(pd.unique(obs_name)).issubset({False, True, np.nan})
        else obs_name.astype("bool"),
        axis=0,
    )
    return obs


class NotEncodedError(AssertionError):
    pass


def _are_ndarrays_equal(arr1: np.ndarray, arr2: np.ndarray) -> np.bool_:
    """Check if two arrays are equal member-wise.

    Note: Two NaN are considered equal.

    Args:
        arr1: First array to compare
        arr2: Second array to compare

    Returns:
        True if the two arrays are equal member-wise
    """
    return np.all(np.equal(arr1, arr2, dtype=object) | ((arr1 != arr1) & (arr2 != arr2)))


def _is_val_missing(data: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Check if values in a AnnData matrix are missing.

    Args:
        data: The AnnData matrix to check

    Returns:
        An array of bool representing the missingness of the original data, with the same shape
    """
    return np.isin(data, [None, ""]) | (data != data)


def _to_dense_matrix(adata: AnnData, layer: str | None = None) -> np.ndarray:  # pragma: no cover
    """Extract a layer from an AnnData object and convert it to a dense matrix if required.

    Args:
        adata: The AnnData where to extract the layer from.
        layer: Name of the layer to extract. If omitted, X is considered.

    Returns:
        The layer as a dense matrix. If a conversion was required, this function returns a copy of the original layer,
        othersize this function returns a reference.
    """
    from scipy.sparse import issparse

    if layer is None:
        return adata.X.toarray() if issparse(adata.X) else adata.X
    else:
        return adata.layers[layer].toarray() if issparse(adata.layers[layer]) else adata.layers[layer]

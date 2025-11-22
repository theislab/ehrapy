from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData, concat
from ehrdata._logger import logger
from ehrdata.core.constants import FEATURE_TYPE_KEY, NUMERIC_TAG
from scipy.sparse import issparse

from ehrapy._compat import _cast_adata_to_match_data_type, function_2D_only, function_future_warning, use_ehrdata
from ehrapy.anndata import _check_feature_types

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ehrdata import EHRData


@function_future_warning("ep.ad.df_to_anndata", "ehrdata.from_pandas")
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
        >>> edata = ep.ad.df_to_anndata(df, index_column="patient_id")
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

    edata = AnnData(X=X, obs=obs, var=var, uns=uns, layers={"original": X.copy()})
    edata.obs_names = edata.obs_names.astype(str)
    edata.var_names = edata.var_names.astype(str)

    return edata


@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.anndata_to_df", "ehrdata.to_pandas")
@function_2D_only()
def anndata_to_df(
    edata: AnnData,
    layer: str | None = None,
    obs_cols: Iterable[str] | str | None = None,
    var_cols: Iterable[str] | str | None = None,
) -> pd.DataFrame:
    """Transform an AnnData object to a Pandas DataFrame.

    Args:
        edata: Central data object.
        layer: The layer to access the values of. If not specified, it uses the `X` matrix.
        obs_cols: The columns of `obs` to add to the DataFrame.
        var_cols: The columns of `var` to fetch values from.

    Returns:
        The AnnData object as a pandas DataFrame

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> df = ep.ad.anndata_to_df(edata)
    """
    if layer is not None:
        X = edata.layers[layer]
    else:
        X = edata.X
    if issparse(X):  # pragma: no cover
        X = X.toarray()

    df = pd.DataFrame(X, columns=list(edata.var_names))
    if obs_cols:
        if len(edata.obs.columns) == 0:
            raise ValueError("Cannot slice columns from empty obs!")
        if isinstance(obs_cols, str):
            obs_cols = list(obs_cols)
        if isinstance(obs_cols, list):  # pragma: no cover
            obs_slice = edata.obs[obs_cols]
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)
    if var_cols:
        if len(edata.var.columns) == 0:
            raise ValueError("Cannot slice columns from empty var!")
        if isinstance(var_cols, str):
            var_cols = list(var_cols)
        if isinstance(var_cols, list):
            var_slice = edata.var[var_cols]
        # reset index needed since we slice all or at least some columns from var DataFrame
        var_slice = var_slice.reset_index(drop=True)
        df = pd.concat([df, var_slice], axis=1)

    return df


@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.move_to_obs")
@function_2D_only()
def move_to_obs(edata: EHRData | AnnData, to_obs: list[str] | str, copy_obs: bool = False) -> EHRData | AnnData:
    """Move inplace or copy features from X to obs.

    Note that columns containing boolean values (either 0/1 or True(true)/False(false))
    will be stored as boolean columns whereas the other non-numerical columns will be stored as categorical.

    Args:
        edata: Central data object.
        to_obs: The columns to move to obs.
        copy_obs: The values are copied to obs (and therefore kept in X) instead of moved completely.

    Returns:
        The original data object with moved or copied columns from X to obs

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.move_to_obs(edata, ["age"], copy_obs=False)
    """
    if isinstance(to_obs, str):  # pragma: no cover
        to_obs = [to_obs]

    # don't allow moving encoded columns as this could lead to inconsistent data in X and obs
    if any(column.startswith("ehrapycat") for column in to_obs):
        raise ValueError(
            "Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!"
        )

    if not all(elem in edata.var_names.values for elem in to_obs):
        raise ValueError(
            f"Columns `{[col for col in to_obs if col not in edata.var_names.values]}` are not in var_names."
        )

    cols_to_obs_indices = edata.var_names.isin(to_obs)

    num_set = _get_var_indices_for_type(edata, NUMERIC_TAG)
    var_num = list(set(to_obs) & set(num_set))

    if copy_obs:
        cols_to_obs = edata[:, cols_to_obs_indices].to_df()
        edata.obs = edata.obs.join(cols_to_obs)
        edata.obs[var_num] = edata.obs[var_num].apply(pd.to_numeric, downcast="float")

        edata.obs = _cast_to_cat_or_bool(edata.obs)
    else:
        df = edata[:, cols_to_obs_indices].to_df()
        edata._inplace_subset_var(~cols_to_obs_indices)
        edata.obs = edata.obs.join(df)
        edata.obs[var_num] = edata.obs[var_num].apply(pd.to_numeric, downcast="float")
        edata.obs = _cast_to_cat_or_bool(edata.obs)

    return edata


@_check_feature_types
def _get_var_indices_for_type(edata: EHRData | AnnData, tag: str) -> list[str]:
    """Get indices of columns in var for a given tag.

    Args:
        edata: Central data object.
        tag: The tag to search for, should be one of 'CATEGORIGAL_TAG', 'NUMERIC_TAG', 'DATE_TAG'

    Returns:
        List of numeric columns
    """
    return edata.var_names[edata.var[FEATURE_TYPE_KEY] == tag].tolist()


@use_ehrdata(deprecated_after="1.0.0")
@function_future_warning("ep.ad.move_to_x")
@function_2D_only()
def move_to_x(edata: EHRData | AnnData, to_x: list[str] | str, copy_x: bool = False) -> EHRData | AnnData:
    """Move features from obs to X inplace.

    Args:
        edata: Central data object
        to_x: The columns to move to X
        copy_x: The values are copied to X (and therefore kept in obs) instead of moved completely

    Returns:
        A new data object with moved columns from obs to X. This should not be used for datetime columns currently.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.ad.move_to_obs(edata, ["age"], copy_obs=False)
        >>> new_edata = ep.ad.move_to_x(edata, ["age"])
    """
    if isinstance(to_x, str):  # pragma: no cover
        to_x = [to_x]

    if not all(elem in edata.obs.columns.values for elem in to_x):
        raise ValueError(f"Columns `{[col for col in to_x if col not in edata.obs.columns.values]}` are not in obs.")

    cols_present_in_x = []
    cols_not_in_x = []

    for col in to_x:
        if col in set(edata.var_names):
            cols_present_in_x.append(col)
        else:
            cols_not_in_x.append(col)

    if cols_present_in_x:
        logger.warn(f"Columns `{cols_present_in_x}` are already in X. Skipped moving `{cols_present_in_x}` to X. ")

    if cols_not_in_x:
        data_from_df = _cast_adata_to_match_data_type(AnnData(edata.obs[cols_not_in_x]), edata)
        new_edata = concat([edata, data_from_df], axis=1)

        if copy_x:
            new_edata.obs = edata.obs
        else:
            new_edata.obs = edata.obs[edata.obs.columns[~edata.obs.columns.isin(cols_not_in_x)]]

        # AnnData's concat discards var if they don't match in their keys, so we need to create a new var
        created_var = pd.DataFrame(index=cols_not_in_x)
        new_edata.var = pd.concat([edata.var, created_var], axis=0)
    else:
        new_edata = edata

    return new_edata


def _get_var_indices_numeric_or_encoded(
    edata: EHRData | AnnData,
    # layer: str | None = None,  # column_indices: Iterable[int] | None = None
) -> np.ndarray:
    return np.arange(0, edata.n_vars)[
        (edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG) | (edata.var["feature_type"].isin(["one-hot", "multi-hot"]))
    ]


def _get_var_indices(edata: EHRData | AnnData, col_names: str | Iterable[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names.

    Args:
        edata: Central data object.
        col_names: Column names to extract the indices for.

    Returns:
        List of column indices.
    """
    col_names = [col_names] if isinstance(col_names, str) else col_names
    mask = np.isin(edata.var_names, col_names)
    indices = np.where(mask)[0].tolist()

    return indices


def _assert_numeric_vars(edata: EHRData | AnnData, vars: Sequence[str]):
    """Ensures that variables are numerics and raises an error if not."""
    num_vars = _get_var_indices_for_type(edata, NUMERIC_TAG)

    try:
        assert set(vars) <= set(num_vars)
    except AssertionError:
        raise ValueError("Some selected vars are not numeric") from None


def _cast_to_cat_or_bool(obs: pd.DataFrame) -> pd.DataFrame:
    """Cast non numerical obs columns to either category or bool.

    Args:
        obs: obs DataFrame

    Returns:
        The type casted DataFrame.
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

from __future__ import annotations

from collections import OrderedDict
from typing import NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData, concat


class BaseDataframes(NamedTuple):
    obs: pd.DataFrame
    df: pd.DataFrame


def df_to_anndata(
    df: pd.DataFrame, columns_obs_only: list[str] | None = None, index_column: str | None = None
) -> AnnData:
    """Transform a given pandas dataframe into an AnnData object.

    Args:
        df: The pandas dataframe to be transformed
        columns_obs_only: An optional list of column names that should belong to obs only and not X
        index_column: The (optional) index column of obs

    Returns:
        An AnnData object created from the given pandas dataframe
    """
    if index_column:
        df = df.set_index(index_column)
    # move columns from the input dataframe to later obs
    dataframes = _move_columns_to_obs(df, columns_obs_only)
    # if data is numerical only, short-circuit AnnData creation to have float dtype instead of object
    all_num = all(np.issubdtype(column_dtype, np.number) for column_dtype in dataframes.df.dtypes)
    X = dataframes.df.to_numpy(copy=True)
    # initializing an OrderedDict with a non-empty dict might not be intended,
    # see: https://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain/25480206
    uns = OrderedDict()
    # store all numerical/non-numerical columns that are not obs only
    uns["numerical_columns"] = list(dataframes.df.select_dtypes("number").columns)
    uns["non_numerical_columns"] = list(set(dataframes.df.columns) ^ set(uns["numerical_columns"]))
    return AnnData(
        X=X,
        obs=dataframes.obs,
        var=pd.DataFrame(index=list(dataframes.df.columns)),
        dtype="float32" if all_num else "object",
        layers={"original": X.copy()},
        uns=uns,
    )


def _move_columns_to_obs(df: pd.DataFrame, columns_obs_only: list[str] | None) -> BaseDataframes:
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


def anndata_to_df(adata: AnnData, add_from_obs: list[str] | str | None = None) -> pd.DataFrame:
    """Transform an AnnData object to a pandas dataframe.

    Args:
        adata: The AnnData object to be transformed into a pandas Dataframe
        add_from_obs: Either "all" or a list of obs names or None, if no columns should be kept from obs

    Returns:
        The AnnData object as a pandas Dataframe
    """
    df = pd.DataFrame(adata.X, columns=list(adata.var_names))
    if add_from_obs:
        if len(adata.obs.columns) == 0:
            raise ObsEmptyError("Cannot slice columns from empty obs!")
        if isinstance(add_from_obs, list):
            obs_slice = adata.obs[add_from_obs]
        else:
            obs_slice = adata.obs
        # reset index needed since we slice all or at least some columns from obs DataFrame
        obs_slice = obs_slice.reset_index(drop=True)
        df = pd.concat([df, obs_slice], axis=1)

    return df


def move_to_obs(adata: AnnData, to_obs: list[str] | str) -> None:
    """Move some columns from X to obs inplace
    Args:
        adata: The AnnData object
        to_obs: The columns to move to obs

    Returns:
        The original AnnData object with moved columns from X to obs
    """
    if isinstance(to_obs, str):
        to_obs = [to_obs]
    # don't allow moving encoded columns as this could lead to inconsistent data in X and obs
    if any(column.startswith("ehrapycat") for column in to_obs):
        raise ObsMoveError(
            "Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!"
        )
    indices = adata.var_names.isin(to_obs)
    df = adata[:, indices].to_df()
    adata._inplace_subset_var(~indices)
    adata.obs = adata.obs.join(df)
    updated_num_uns, updated_non_num_uns, num_var = _update_uns(adata, to_obs)
    adata.obs[num_var] = adata.obs[num_var].apply(pd.to_numeric, errors="ignore", downcast="float")
    adata.uns["numerical_columns"] = updated_num_uns
    adata.uns["non_numerical_columns"] = updated_non_num_uns


def move_to_x(adata: AnnData, to_x: list[str] | str) -> AnnData:
    """Move some columns from obs to X inplace
    Args:
        adata: The AnnData object
        to_x: The columns to move to X

    Returns:
        A new AnnData object with moved columns from obs to X. This should not be used for datetime columns currently.
    """
    if isinstance(to_x, str):
        to_x = [to_x]
    new_adata = concat([adata, AnnData(adata.obs[to_x], dtype="object")], axis=1)
    new_adata.obs = adata.obs[adata.obs.columns[~adata.obs.columns.isin(to_x)]]
    # update uns (copy maybe: could be a costly operation but reduces reference cycles)
    # users might save those as separate AnnData object and this could be unexpected behaviour if we dont copy
    num_columns_moved, non_num_columns_moved, _ = _update_uns(adata, to_x, True)
    new_adata.uns["numerical_columns"] = adata.uns["numerical_columns"] + num_columns_moved
    new_adata.uns["non_numerical_columns"] = adata.uns["non_numerical_columns"] + non_num_columns_moved

    return new_adata


def get_column_indices(adata: AnnData, col_names: str | list[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object
        col_names: Column names to extract the indices for

    Returns:
        Set of column indices
    """
    if isinstance(col_names, str):  # pragma: no cover
        col_names = [col_names]

    indices = list()
    for idx, col in enumerate(adata.var_names):
        if col in col_names:
            indices.append(idx)

    return indices


def get_column_values(adata: AnnData, indices: int | list[int]) -> np.ndarray:
    """Fetches the column values for a specific index from X

    Args:
        adata: :class:`~anndata.AnnData` object
        indices: The index to extract the values for

    Returns:
        :class:`~numpy.ndarray` object containing the column values
    """
    return np.take(adata.X, indices, axis=1)


def assert_encoded(adata: AnnData):
    try:
        assert np.issubdtype(adata.X.dtype, np.number)
    except AssertionError:
        raise NotEncodedError("The AnnData object has not yet been encoded.") from AssertionError


def get_numeric_vars(adata: AnnData) -> list[str]:
    """Fetches the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object

    Returns:
        List of column numeric column names
    """
    assert_encoded(adata)

    return adata.uns["numerical_columns"]


def assert_numeric_vars(adata: AnnData, vars: list[str]):
    num_vars = get_numeric_vars(adata)

    try:
        assert set(vars) <= set(num_vars)
    except AssertionError:
        raise ValueError("Some selected vars are not numeric")


def set_numeric_vars(
    adata: AnnData, values: np.ndarray, vars: list[str] | None = None, copy: bool = False
) -> AnnData | None:
    """Sets the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object
        values: Matrix containing the replacement values
        vars: List of names of the numeric variables to replace. If `None` they will be detected using :func:`~ehrapy.api.preprocessing.get_numeric_vars`.
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with updated X
    """
    assert_encoded(adata)

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"values must be numeric (current dtype is {values.dtype})")

    n_values = values.shape[1]

    if n_values != len(vars):
        raise ValueError(f"Number of values ({n_values}) does not match number of vars ({len(vars)})")

    if copy:
        adata = adata.copy()

    vars_idx = get_column_indices(adata, vars)

    for i in range(n_values):
        adata.X[:, vars_idx[i]] = values[:, i]

    return adata


def _update_uns(
    adata: AnnData, moved_columns: list[str], to_x: bool = False
) -> tuple[list[str], list[str], list[str] | None]:
    """Update .uns of adata to reflect the changes made on the object by moving columns from X to obs or vice versa.

    1.) Moving `col1` from `X` to `obs`: `col1` is either numerical or non_numerical, so delete it from the corresponding entry in `uns`
    2.) Moving `col1` from `obs` to `X`: `col1` is either numerical or non_numerical, so add it to the corresponding entry in `uns`
    """
    moved_columns_set = set(moved_columns)
    num_set = set(adata.uns["numerical_columns"])
    non_num_set = set(adata.uns["non_numerical_columns"])
    if not to_x:
        var_num = []
        for var in moved_columns:
            if var in num_set:
                var_num.append(var)
                num_set -= {var}
            elif var in non_num_set:
                non_num_set -= {var}
        return list(num_set), list(non_num_set), var_num
    else:
        all_moved_non_num_columns = moved_columns_set ^ set(adata.obs.select_dtypes("number").columns)
        all_moved_num_columns = list(moved_columns_set ^ all_moved_non_num_columns)
        return all_moved_num_columns, list(all_moved_non_num_columns), None


class NotEncodedError(AssertionError):
    pass


class ColumnNotFoundError(Exception):
    pass


class ObsEmptyError(Exception):
    pass


class ObsMoveError(Exception):
    pass

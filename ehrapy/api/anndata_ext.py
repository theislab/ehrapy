from __future__ import annotations

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

    return AnnData(
        X=X,
        obs=dataframes.obs,
        var=pd.DataFrame(index=list(dataframes.df.columns)),
        dtype="float32" if all_num else "object",
        layers={"original": X.copy()},
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
        raise ObsMoveError("Cannot move encoded columns from X to obs. Either undo encoding or remove them from the list!")
    indices = adata.var_names.isin(to_obs)
    df = adata[:, indices].to_df()
    adata._inplace_subset_var(~indices)
    adata.obs = adata.obs.join(df)


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

    return new_adata


def get_column_indices(adata: AnnData, col_names: str | list[str]) -> list[int]:
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object
        col_names: Column names to extract the indices for

    Returns:
        Set of column indices
    """
    if isinstance(col_names, str):
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


class ColumnNotFoundError(Exception):
    pass


class ObsEmptyError(Exception):
    pass


class ObsMoveError(Exception):
    pass

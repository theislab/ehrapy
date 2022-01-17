from __future__ import annotations

import numpy as np
from anndata import AnnData


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


def assert_encoded(adata: AnnData):
    try:
        assert "categoricals" in adata.uns_keys()
    except AssertionError:
        raise NotEncodedError("The AnnData object has not yet been encoded.") from AssertionError


def get_numeric_vars(adata: AnnData) -> list[str]:
    """Fetches the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object

    Returns:
        Set of column numeric column names
    """

    assert_encoded(adata)

    return adata.uns["categoricals"]["not_categorical"]


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
        vars: List of names of the numeric variables to replace. If `None` they will be detected using ~ehrapy.pp.get_numeric_vars.
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


class NotEncodedError(AssertionError):
    def __init__(self, message):
        super().__init__(message)

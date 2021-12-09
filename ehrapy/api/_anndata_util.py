from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from scipy.sparse import spmatrix


def get_column_indices(adata: AnnData, col_names=Union[str, List]) -> List[int]:
    """Fetches the column indices in X for a given list of column names.

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


def get_column_values(adata: AnnData, indices: Union[int, List[int]]) -> np.ndarray:
    """Fetches the column values for a specific index from X.

    Args:
        adata: :class:`~anndata.AnnData` object
        indices: The index to extract the values for

    Returns:
        :class:`~numpy.ndarray` object containing the column values
    """
    return np.take(adata.X, indices, axis=1)


def get_numeric_vars(adata: AnnData) -> List[str]:
    """Fetches the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object

    Returns:
        Set of column numeric column names
    """

    numeric_vars: List[str] = []

    return numeric_vars


def set_numeric_vars(
    adata: AnnData, values: Union[np.ndarray, spmatrix], vars: Optional[List[str]] = None, copy: bool = False
) -> Optional[AnnData]:
    """Sets the column names for numeric variables in X.

    Args:
        adata: :class:`~anndata.AnnData` object
        values: Matrix containing the replacement values
        vars: List of names of the numeric variables to replace. If `None` they will be detected using ~ehrapy.pp.get_numeric_vars.
        copy: Whether to return a copy with the normalized data.

    Returns:
        :class:`~anndata.AnnData` object with updated X
    """

    adata_to_set = adata
    if copy:
        adata_copy = adata.copy()
        adata_to_set = adata_copy

    return adata_to_set

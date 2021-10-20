import importlib
from typing import List, Union

import numpy as np
from anndata import AnnData


def get_column_indices(adata: AnnData, col_names=Union[str, List]) -> List[int]:
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


def get_column_values(adata: AnnData, indices: Union[int, List[int]]) -> np.ndarray:
    """Fetches the column values for a specific index from X

    Args:
        adata: :class:`~anndata.AnnData` object
        indices: The index to extract the values for

    Returns:
        :class:`~numpy.ndarray` object containing the column values
    """
    return np.take(adata.X, indices, axis=1)


def check_module_importable(package: str) -> bool:
    """Checks whether a module is installed and can be loaded.

    Args:
        package: The package to check.

    Returns:
        True if the package is installed, false elsewise
    """
    spam_spec = importlib.util.find_spec(package)

    return spam_spec is not None

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


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
    return np.isin(data, [None, "", np.nan]) | (data != data)


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

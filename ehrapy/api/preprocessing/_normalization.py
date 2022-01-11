from __future__ import annotations

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, minmax_scale, robust_scale, scale

from ehrapy.api._anndata_util import assert_encoded, get_column_indices, get_column_values, get_numeric_vars

available_normalization_methods = {"scale", "minmax", "maxabs", "robust_scale", "identity"}


def normalize(adata: AnnData, methods: dict[str, list[str]] | str, copy: bool = False) -> AnnData | None:
    """Normalize numeric variable.

    This function normalizes the numeric variables in an AnnData object.

    Available normalization methods are:

    1. scale (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)
    2. minmax (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
    3. maxabs (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale)
    4. robust_scale (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale)
    5. identity (return the un-normalized values)

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encode using ~ehrapy.preprocessing.encode.encode.
        methods: Methods to use for normalization. Either:

            str: Name of the method to use for all numeric variable

            Dict: A dictionary specifying the method for each numeric variable where keys are methods and values are lists of variables
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.normalize(adata, method="minmax", copy=True)
    """
    assert_encoded(adata)

    num_vars = get_numeric_vars(adata)
    if isinstance(methods, str):
        methods = {methods: num_vars}
    else:
        if not set(methods.keys()) <= available_normalization_methods:
            raise ValueError(
                "Some keys of methods are not available normalization methods. Available methods are:"
                f"{available_normalization_methods}"
            )
        for vars_list in methods.values():
            if not set(vars_list) <= set(num_vars):
                raise ValueError("Some values of methods contain items which are not numeric variables")

    if copy:
        adata = adata.copy()

    adata.layers["raw"] = adata.X.copy()

    for method, vars_list in methods.items():
        var_idx = get_column_indices(adata, vars_list)
        var_values = get_column_values(adata, var_idx)

        if method == "scale":
            adata.X[:, var_idx] = _norm_scale(var_values)
        elif method == "minmax":
            adata.X[:, var_idx] = _norm_minmax(var_values)
        elif method == "maxabs":
            adata.X[:, var_idx] = _norm_maxabs(var_values)
        elif method == "robust_scale":
            adata.X[:, var_idx] = _norm_robust_scale(var_values)
        elif method == "identity":
            adata.X[:, var_idx] = _norm_identity(var_values)

    return adata


def _norm_scale(values: np.ndarray) -> np.ndarray:
    """Apply standard scaling normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with scaled values
    """
    return scale(values)


def _norm_minmax(values: np.ndarray) -> np.ndarray:
    """Apply minmax normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with minmax scaled values
    """
    return minmax_scale(values)


def _norm_maxabs(values: np.ndarray) -> np.ndarray:
    """Apply maxabs normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with maxabs scaled values
    """
    return maxabs_scale(values)


def _norm_robust_scale(values: np.ndarray) -> np.ndarray:
    """Apply robust_scale normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with robust scaled values
    """
    return robust_scale(values)


def _norm_identity(values: np.ndarray) -> np.ndarray:
    """Apply identity normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with normalized values
    """
    return values

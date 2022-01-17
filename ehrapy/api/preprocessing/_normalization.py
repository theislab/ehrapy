from __future__ import annotations

import numpy as np
from anndata import AnnData
from sklearn.preprocessing import maxabs_scale, minmax_scale, power_transform, quantile_transform, robust_scale, scale

from ehrapy.api._anndata_util import (
    assert_encoded,
    assert_numeric_vars,
    get_column_indices,
    get_column_values,
    get_numeric_vars,
    set_numeric_vars,
)

available_normalization_methods = {
    "scale",
    "minmax",
    "maxabs",
    "robust_scale",
    "quantile_uniform",
    "quantile_normal",
    "power_yeo_johnson",
    "power_box_cox",
    "log1p",
    "sqrt",
    "identity",
}


def normalize(
    adata: AnnData, methods: dict[str, list[str]] | str, base: int | float | None = None, copy: bool = False
) -> AnnData | None:
    """Normalize numeric variables.

    This function normalizes the numeric variables in an AnnData object.

    Available normalization methods are:

    1. scale (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale)
    2. minmax (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler)
    3. maxabs (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html#sklearn.preprocessing.maxabs_scale)
    4. robust_scale (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html#sklearn.preprocessing.robust_scale)
    5. quantile_uniform (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform)
    6. quantile_normal (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html#sklearn.preprocessing.quantile_transform)
    7. power_yeo_johnson (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform)
    8. power_box_cox (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html#sklearn.preprocessing.power_transform)
    9. log1p Computes :math:`x = \\log(x + 1)`, where :math:`log` denotes the natural logarithm unless a different base is given.
    10. sqrt Computes the square root of the values.
    11. identity (return the un-normalized values)

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encode using ~ehrapy.preprocessing.encode.encode.
        methods: Methods to use for normalization. Either:

            str: Name of the method to use for all numeric variable

            Dict: A dictionary specifying the method for each numeric variable where keys are methods and values are lists of variables
        base: Numeric base for logarithm in the log1p method. If None the natural logarithm is used.
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.normalize(adata, method="minmax", copy=True)
    """
    assert_encoded(adata)

    if isinstance(methods, str):
        methods = {methods: get_numeric_vars(adata)}
    else:
        if not set(methods.keys()) <= available_normalization_methods:
            raise ValueError(
                "Some keys of methods are not available normalization methods. Available methods are:"
                f"{available_normalization_methods}"
            )
        for vars_list in methods.values():
            assert_numeric_vars(adata, vars_list)

    adata = _prep_adata_norm(adata, copy)

    for method, vars_list in methods.items():
        var_idx = get_column_indices(adata, vars_list)
        var_values = get_column_values(adata, var_idx)

        if method == "minmax":
            adata.X[:, var_idx] = _norm_minmax(var_values)
        elif method == "maxabs":
            adata.X[:, var_idx] = _norm_maxabs(var_values)
        elif method == "robust_scale":
            adata.X[:, var_idx] = _norm_robust_scale(var_values)
        elif method == "quantile_uniform":
            adata.X[:, var_idx] = _norm_quantile_uniform(var_values)
        elif method == "quantile_normal":
            adata.X[:, var_idx] = _norm_quantile_normal(var_values)
        elif method == "power_yeo_johnson":
            adata.X[:, var_idx] = _norm_power_yeo_johnson(var_values)
        elif method == "power_box_cox":
            adata.X[:, var_idx] = _norm_power_box_cox(var_values)
        elif method == "log1p":
            adata.X[:, var_idx] = _norm_log1p(var_values, base)
        elif method == "sqrt":
            adata.X[:, var_idx] = _norm_sqrt(var_values)

        _record_norm(adata, vars_list, method)

    return adata


def norm_scale(adata: AnnData, vars: list[str] | None = None, copy: bool = False, **kwargs) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by ~sklearn.preprocessing.scale, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html#sklearn.preprocessing.scale for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place
        **kwargs: Additional arguments passed to ~sklearn.preprocessing.scale

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norma_scale(adata, copy=True)
    """

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    var_values = scale(var_values, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "scale")

    return adata


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


def _norm_quantile_uniform(values: np.ndarray) -> np.ndarray:
    """Apply uniform quantile normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with uniform quantile transformed values
    """

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        return np.squeeze(quantile_transform(values, output_distribution="uniform"))
    else:
        return quantile_transform(values, output_distribution="uniform")


def _norm_quantile_normal(values: np.ndarray) -> np.ndarray:
    """Apply normal quantile normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with normal quantile transformed values
    """

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        return np.squeeze(quantile_transform(values, output_distribution="normal"))
    else:
        return quantile_transform(values, output_distribution="normal")


def _norm_power_yeo_johnson(values: np.ndarray) -> np.ndarray:
    """Apply Yeo-Johnson power normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with Yeo-Johnson transformed values
    """

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        return np.squeeze(power_transform(values, method="yeo-johnson"))
    else:
        return power_transform(values, method="yeo-johnson")


def _norm_power_box_cox(values: np.ndarray) -> np.ndarray:
    """Apply Box-Cox power normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with Box-Cox transformed values
    """

    if values.ndim == 1:
        values = values.reshape(-1, 1)
        return np.squeeze(power_transform(values, method="box-cox"))
    else:
        return power_transform(values, method="box-cox")


def _norm_log1p(values: np.ndarray, base: int | float | None) -> np.ndarray:
    """Apply log1p normalization.

    Args:
        values: A single column numpy array
        base: Numeric base for the logarithm

    Returns:
        Single column numpy array with log1p transformed values
    """

    np.log1p(values, out=values)

    if base is not None:
        np.divide(values, np.log(base), out=values)

    return values


def _norm_sqrt(values: np.ndarray) -> np.ndarray:
    """Apply square root normalization.

    Args:
        values: A single column numpy array

    Returns:
        Single column numpy array with square root transformed values
    """

    return np.sqrt(values)


def norm_identity(adata: AnnData, vars: list[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply identity normalization.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize. If None (default) all numeric variables will be normalized.
        copy: Whether to return a copy or act in place

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Example:
        .. code-block:: python

            import ehrapy.api as ep
            adata = ep.data.mimic_2(encode=True)
            adata_norm = ep.pp.norma_identity(adata, copy=True)
    """

    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = get_column_values(adata, var_idx)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "identity")

    return adata


def _prep_adata_norm(adata: AnnData, copy: bool = False) -> AnnData | None:

    if copy:
        adata = adata.copy()

    if "raw_norm" not in adata.layers.keys():
        adata.layers["raw_norm"] = adata.X.copy()

    return adata


def _record_norm(adata: AnnData, vars_list: list[str], method: str) -> None:

    if "normalization" in adata.uns_keys():
        norm_record = adata.uns["normalization"]
    else:
        norm_record = {}

    for var in vars_list:
        if var in norm_record.keys():
            norm_record[var].append(method)
        else:
            norm_record[var] = [method]

    adata.uns["normalization"] = norm_record

    return None

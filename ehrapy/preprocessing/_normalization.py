from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lamin_utils import logger
from sklearn.preprocessing import maxabs_scale, minmax_scale, power_transform, quantile_transform, robust_scale, scale

from ehrapy.anndata.anndata_ext import (
    _get_column_indices,
    assert_numeric_vars,
    get_numeric_vars,
    set_numeric_vars,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData


def _scale_func_batch(scale_func, var_values, adata, batch_key, **kwargs):
    if batch_key is None:
        var_values = scale_func(var_values, **kwargs)

    if batch_key is not None:
        for batch in adata.obs[batch_key].unique():
            batch_idx = adata.obs[batch_key] == batch
            var_values[batch_idx] = scale_func(var_values[batch_idx], **kwargs)

    return var_values


def scale_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.scale`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.scale_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(scale, var_values, adata, batch_key, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "scale")

    return adata


def minmax_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.minmax_scale`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.minmax_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to False.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.minmax_scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.minmax_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(minmax_scale, var_values, adata, batch_key, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "minmax")

    return adata


def maxabs_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False
) -> AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.maxabs_scale`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.maxabs_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.maxabs_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(maxabs_scale, var_values, adata, batch_key)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "maxabs")

    return adata


def robust_scale_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.robust_scale`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.robust_scale.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.robust_scale`

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.robust_scale_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(robust_scale, var_values, adata, batch_key, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "robust_scale")


    return adata


def quantile_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.quantile_transform`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.quantile_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using ~ehrapy.preprocessing.encode.encode.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.quantile_transform`

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.quantile_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(quantile_transform, var_values, adata, batch_key, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "quantile")


    return adata


def power_norm(
    adata: AnnData, vars: str | Sequence[str] | None = None, batch_key: str | None = None, copy: bool = False, **kwargs
) -> AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.power_transform`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.power_transform.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        batch_key: Key in adata.obs that contains batch information. If provided, scaling is applied per batch.
        copy: Whether to return a copy or act in place. Defaults to False.
        **kwargs: Additional arguments passed to :func:`~sklearn.preprocessing.power_transform`

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.power_norm(adata, copy=True)
    """
    if batch_key is not None and batch_key not in adata.obs_keys():
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs.")
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = _scale_func_batch(power_transform, var_values, adata, batch_key, **kwargs)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "power")

    logg.debug("Power transformation normalization was applied on AnnData's `X`.")

    return adata


def log_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    copy: bool = False,
) -> AnnData | None:
    """Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm. Defaults to 1.
        copy: Whether to return a copy or act in place. Defaults to False.

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.log_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    adata_to_check_for_negatives = adata[:, vars] if vars else adata
    offset_tmp_applied = adata_to_check_for_negatives.X + offset
    if np.any(offset_tmp_applied < 0):
        raise ValueError(
            "Matrix X contains negative values. "
            "Undefined behavior for log normalization. "
            "Please specifiy a higher offset to this function "
            "or offset negative values with ep.pp.offset_negative_values()."
        )

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    if offset == 1:
        np.log1p(var_values, out=var_values)
    else:
        var_values = var_values + offset
        np.log(var_values, out=var_values)

    if base is not None:
        np.divide(var_values, np.log(base), out=var_values)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "log")


    return adata


def sqrt_norm(adata: AnnData, vars: str | Sequence[str] | None = None, copy: bool = False) -> AnnData | None:
    """Apply square root normalization.

    Take the square root of all values.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized. Defaults to None.
        copy: Whether to return a copy or act in place. Defaults to False.

    Returns:
        :class:`~anndata.AnnData` object with normalized X.
        Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.sqrt_norm(adata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = _get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    var_values = np.sqrt(var_values)

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, "sqrt")

    logg.debug("Square root normalization was applied on AnnData's `X`.")

    return adata


def _prep_adata_norm(adata: AnnData, copy: bool = False) -> AnnData | None:  # pragma: no cover
    if copy:
        adata = adata.copy()

    if "raw_norm" not in adata.layers.keys():
        adata.layers["raw_norm"] = adata.X.copy()

    return adata


def _record_norm(adata: AnnData, vars: Sequence[str], method: str) -> None:
    if "normalization" in adata.uns_keys():
        norm_record = adata.uns["normalization"]
    else:
        norm_record = {}

    for var in vars:
        if var in norm_record.keys():
            norm_record[var].append(method)
        else:
            norm_record[var] = [method]

    adata.uns["normalization"] = norm_record

    return None


def offset_negative_values(adata: AnnData, layer: str = None, copy: bool = False) -> AnnData:
    """Offsets negative values into positive ones with the lowest negative value becoming 0.

    This is primarily used to enable the usage of functions such as log_norm that
    do not allow negative values for mathematical or technical reasons.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
        layer: The layer to offset.
        copy: Whether to return a modified copy of the AnnData object.

    Returns:
        Copy of AnnData object if copy is True.
    """
    if copy:
        adata = adata.copy()

    if layer:
        minimum = np.min(adata[layer])
        if minimum < 0:
            adata[layer] = adata[layer] + np.abs(minimum)
    else:
        minimum = np.min(adata.X)
        if minimum < 0:
            adata.X = adata.X + np.abs(minimum)

    if copy:
        return adata

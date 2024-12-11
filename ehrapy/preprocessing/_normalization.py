from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import sklearn.preprocessing as sklearn_pp

from ehrapy._compat import is_dask_array

try:
    import dask_ml.preprocessing as daskml_pp
except ImportError:
    daskml_pp = None

from ehrapy.anndata.anndata_ext import (
    assert_numeric_vars,
    get_column_indices,
    get_numeric_vars,
    set_numeric_vars,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from anndata import AnnData


def _scale_func_group(
    adata: AnnData,
    scale_func: Callable[[np.ndarray | pd.DataFrame], np.ndarray],
    vars: str | Sequence[str] | None,
    group_key: str | None,
    copy: bool,
    norm_name: str,
) -> AnnData | None:
    """apply scaling function to selected columns of adata, either globally or per group."""

    if group_key is not None and group_key not in adata.obs_keys():
        raise KeyError(f"group key '{group_key}' not found in adata.obs.")

    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = get_numeric_vars(adata)
    else:
        assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    var_idx = get_column_indices(adata, vars)
    var_values = np.take(adata.X, var_idx, axis=1)

    if group_key is None:
        var_values = scale_func(var_values)

    else:
        for group in adata.obs[group_key].unique():
            group_idx = adata.obs[group_key] == group
            var_values[group_idx] = scale_func(var_values[group_idx])

    set_numeric_vars(adata, var_values, vars)

    _record_norm(adata, vars, norm_name)

    if copy:
        return adata
    else:
        return None


def scale_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.StandardScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :func:`~dask_ml.preprocessing.StandardScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.StandardScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the StandardScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.scale_norm(adata, copy=True)
    """

    if is_dask_array(adata.X):
        scale_func = daskml_pp.StandardScaler(**kwargs).fit_transform
    else:
        scale_func = sklearn_pp.StandardScaler(**kwargs).fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="scale",
    )


def minmax_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MinMaxScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :func:`~dask_ml.preprocessing.MinMaxScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.MinMaxScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the MinMaxScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.minmax_norm(adata, copy=True)
    """

    if is_dask_array(adata.X):
        scale_func = daskml_pp.MinMaxScaler(**kwargs).fit_transform
    else:
        scale_func = sklearn_pp.MinMaxScaler(**kwargs).fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="minmax",
    )


def maxabs_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MaxAbsScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.maxabs_norm(adata, copy=True)
    """
    if is_dask_array(adata.X):
        raise NotImplementedError("MaxAbsScaler is not implemented in dask_ml.")
    else:
        scale_func = sklearn_pp.MaxAbsScaler().fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="maxabs",
    )


def robust_scale_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.RobustScaler`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :func:`~dask_ml.preprocessing.RobustScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.RobustScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the RobustScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.robust_scale_norm(adata, copy=True)
    """
    if is_dask_array(adata.X):
        scale_func = daskml_pp.RobustScaler(**kwargs).fit_transform
    else:
        scale_func = sklearn_pp.RobustScaler(**kwargs).fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="robust_scale",
    )


def quantile_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.QuantileTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :func:`~dask_ml.preprocessing.QuantileTransformer`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.QuantileTransformer.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the QuantileTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.quantile_norm(adata, copy=True)
    """
    if is_dask_array(adata.X):
        scale_func = daskml_pp.QuantileTransformer(**kwargs).fit_transform
    else:
        scale_func = sklearn_pp.QuantileTransformer(**kwargs).fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="quantile",
    )


def power_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :func:`~sklearn.preprocessing.PowerTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the PowerTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.power_norm(adata, copy=True)
    """
    if is_dask_array(adata.X):
        raise NotImplementedError("dask-ml has no PowerTransformer, this is only available in scikit-learn")
    else:
        scale_func = sklearn_pp.PowerTransformer(**kwargs).fit_transform

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="power",
    )


def log_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    copy: bool = False,
) -> AnnData | None:
    """Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

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

    var_idx = get_column_indices(adata, vars)
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

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import sklearn.preprocessing as sklearn_pp

from ehrapy._compat import _raise_array_type_not_implemented

try:
    import dask.array as da
    import dask_ml.preprocessing as daskml_pp

    DASK_AVAILABLE = True
except ImportError:
    daskml_pp = None
    DASK_AVAILABLE = False

from ehrapy.anndata._constants import NUMERIC_TAG
from ehrapy.anndata.anndata_ext import (
    _assert_numeric_vars,
    _get_var_indices_for_type,
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
    axis: int | None,
    copy: bool,
    norm_name: str,
) -> AnnData | None:
    """Apply scaling function to selected columns of adata, either globally or per group."""
    if group_key is not None and group_key not in adata.obs_keys():
        raise KeyError(f"group key '{group_key}' not found in adata.obs.")

    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(adata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    # Handle 3D R tensor normalization (axis=2)
    if axis == 2:
        if group_key is not None:
            raise ValueError("group_key not supported with axis=2 (time normalization)")

        R_data = adata.R.copy()
        var_indices = [adata.var.index.get_loc(v) for v in vars]

        for i in range(R_data.shape[0]):  # each patient
            for j in var_indices:  # selected variables
                time_series = R_data[i, j, :]
                valid_mask = ~np.isnan(time_series)

                if valid_mask.sum() > 1:  # Need at least 2 values
                    valid_values = time_series[valid_mask].reshape(-1, 1)
                    normalized_values = scale_func(valid_values).flatten()
                    time_series[valid_mask] = normalized_values
                    R_data[i, j, :] = time_series

        adata.R = R_data
        _record_norm(adata, vars, norm_name)

        if copy:
            return adata
        else:
            return None

    # Original X matrix normalization (axis=0 or axis=1 or None)
    var_values = adata[:, vars].X.copy()

    if axis == 0:  # Normalize across observations
        if group_key is not None:
            raise ValueError("group_key not supported with axis=0")
        var_values = scale_func(var_values.T).T
    elif axis == 1:  # Normalize across variables
        if group_key is not None:
            raise ValueError("group_key not supported with axis=1")
        var_values = scale_func(var_values)
    else:  # Original behavior (axis=None)
        if group_key is None:
            var_values = scale_func(var_values)
        else:
            for group in adata.obs[group_key].unique():
                group_idx = adata.obs[group_key] == group
                var_values[group_idx] = scale_func(var_values[group_idx])

    if len(vars) == adata.n_vars:
        adata.X = var_values
    else:
        var_indices = [adata.var.index.get_loc(v) for v in vars]
        adata.X = adata.X.astype(var_values.dtype)
        adata.X[:, var_indices] = var_values

    _record_norm(adata, vars, norm_name)

    if copy:
        return adata
    else:
        return None


@singledispatch
def _scale_norm_function(arr):
    _raise_array_type_not_implemented(_scale_norm_function, type(arr))


@_scale_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.StandardScaler(**kwargs).fit_transform


if DASK_AVAILABLE:

    @_scale_norm_function.register
    def _(arr: da.Array, **kwargs):
        return daskml_pp.StandardScaler(**kwargs).fit_transform


def scale_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.StandardScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.StandardScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.StandardScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the StandardScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.scale_norm(adata, copy=True)
        >>> # Normalize across time for EHRData
        >>> adata_time_norm = ep.pp.scale_norm(adata, axis=2, copy=True)
    """
    scale_func = _scale_norm_function(adata.X, **kwargs)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="scale",
    )


@singledispatch
def _minmax_norm_function(arr):
    _raise_array_type_not_implemented(_minmax_norm_function, type(arr))


@_minmax_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.MinMaxScaler(**kwargs).fit_transform


if DASK_AVAILABLE:

    @_minmax_norm_function.register
    def _(arr: da.Array, **kwargs):
        return daskml_pp.MinMaxScaler(**kwargs).fit_transform


def minmax_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MinMaxScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.MinMaxScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.MinMaxScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the MinMaxScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.minmax_norm(adata, copy=True)
    """
    scale_func = _minmax_norm_function(adata.X, **kwargs)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="minmax",
    )


@singledispatch
def _maxabs_norm_function(arr):
    _raise_array_type_not_implemented(_scale_norm_function, type(arr))


@_maxabs_norm_function.register
def _(arr: np.ndarray):
    return sklearn_pp.MaxAbsScaler().fit_transform


def maxabs_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
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
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.maxabs_norm(adata, copy=True)
    """
    scale_func = _maxabs_norm_function(adata.X)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="maxabs",
    )


@singledispatch
def _robust_scale_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_robust_scale_norm_function, type(arr))


@_robust_scale_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.RobustScaler(**kwargs).fit_transform


if DASK_AVAILABLE:

    @_robust_scale_norm_function.register
    def _(arr: da.Array, **kwargs):
        return daskml_pp.RobustScaler(**kwargs).fit_transform


def robust_scale_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.RobustScaler`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.RobustScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.RobustScaler.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the RobustScaler.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.robust_scale_norm(adata, copy=True)
    """
    scale_func = _robust_scale_norm_function(adata.X, **kwargs)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="robust_scale",
    )


@singledispatch
def _quantile_norm_function(arr):
    _raise_array_type_not_implemented(_quantile_norm_function, type(arr))


@_quantile_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.QuantileTransformer(**kwargs).fit_transform


if DASK_AVAILABLE:

    @_quantile_norm_function.register
    def _(arr: da.Array, **kwargs):
        return daskml_pp.QuantileTransformer(**kwargs).fit_transform


def quantile_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.QuantileTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html for details.
    If `adata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.QuantileTransformer`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.QuantileTransformer.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the QuantileTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.quantile_norm(adata, copy=True)
    """
    scale_func = _quantile_norm_function(adata.X, **kwargs)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="quantile",
    )


@singledispatch
def _power_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_power_norm_function, type(arr))


@_power_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.PowerTransformer(**kwargs).fit_transform


def power_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    axis: int | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.PowerTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html for details.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in adata.obs that contains group information. If provided, scaling is applied per group.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the PowerTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed adata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in adata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> adata_norm = ep.pp.power_norm(adata, copy=True)
    """
    scale_func = _power_norm_function(adata.X, **kwargs)

    return _scale_func_group(
        adata=adata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        axis=axis,
        copy=copy,
        norm_name="power",
    )


def log_norm(
    adata: AnnData,
    vars: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    axis: int | None = None,
    copy: bool = False,
) -> AnnData | None:
    r"""Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`.

    Args:
        adata: :class:`~anndata.AnnData` object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm.
        axis: Axis along which to normalize. 0=observations, 1=variables, 2=time (R tensor), None=original behavior.
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
        vars = _get_var_indices_for_type(adata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(adata, vars)

    adata = _prep_adata_norm(adata, copy)

    # Handle 3D R tensor normalization (axis=2)
    if axis == 2 and hasattr(adata, "R") and adata.R is not None:
        R_data = adata.R.copy()
        var_indices = [adata.var.index.get_loc(v) for v in vars]

        for i in range(R_data.shape[0]):  # each patient
            for j in var_indices:  # selected variables
                time_series = R_data[i, j, :]
                valid_mask = ~np.isnan(time_series)

                if valid_mask.sum() > 0:
                    valid_values = time_series[valid_mask]
                    offset_applied = valid_values + offset

                    if np.any(offset_applied < 0):
                        raise ValueError(
                            f"R tensor contains negative values for patient {i}, variable {j}. "
                            "Undefined behavior for log normalization. "
                            "Please specify a higher offset."
                        )

                    if offset == 1:
                        normalized_values = np.log1p(valid_values)
                    else:
                        normalized_values = np.log(valid_values + offset)

                    if base is not None:
                        normalized_values = normalized_values / np.log(base)

                    time_series[valid_mask] = normalized_values
                    R_data[i, j, :] = time_series

        adata.R = R_data
        _record_norm(adata, vars, "log")

        if copy:
            return adata
        else:
            return None

    # Original X matrix normalization
    if axis == 0:  # Across observations
        data_to_check = adata[:, vars].X.T if vars else adata.X.T
        var_values = adata[:, vars].X.copy().T
    elif axis == 1:  # Across variables
        data_to_check = adata[:, vars].X if vars else adata.X
        var_values = adata[:, vars].X.copy()
    else:  # Original behavior (axis=None)
        data_to_check = adata[:, vars].X if vars else adata.X
        var_values = adata[:, vars].X.copy()

    offset_tmp_applied = data_to_check + offset
    if np.any(offset_tmp_applied < 0):
        raise ValueError(
            "Matrix X contains negative values. "
            "Undefined behavior for log normalization. "
            "Please specify a higher offset to this function "
            "or offset negative values with ep.pp.offset_negative_values()."
        )

    if offset == 1:
        np.log1p(var_values, out=var_values)
    else:
        var_values = var_values + offset
        np.log(var_values, out=var_values)

    if base is not None:
        np.divide(var_values, np.log(base), out=var_values)

    if axis == 0:  # Transpose back
        var_values = var_values.T

    adata.X = adata.X.astype(var_values.dtype)
    adata[:, vars].X = var_values

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

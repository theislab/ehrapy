from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import sklearn.preprocessing as sklearn_pp

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented
from ehrapy.anndata._constants import NUMERIC_TAG
from ehrapy.anndata.anndata_ext import (
    _assert_numeric_vars,
    _get_var_indices_for_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from anndata import AnnData
    from ehrdata import EHRData


def _scale_func_group(
    edata: EHRData | AnnData,
    scale_func: Callable[[np.ndarray | pd.DataFrame], np.ndarray],
    vars: str | Sequence[str] | None,
    group_key: str | None,
    copy: bool,
    norm_name: str,
) -> EHRData | AnnData | None:
    """Apply scaling function to selected columns of edata, either globally or per group."""
    if group_key is not None and group_key not in edata.obs_keys():
        raise KeyError(f"group key '{group_key}' not found in edata.obs.")

    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(edata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(edata, vars)

    edata = _prep_edata_norm(edata, copy)

    var_values = edata[:, vars].X.copy()

    if group_key is None:
        var_values = scale_func(var_values)

    else:
        for group in edata.obs[group_key].unique():
            group_idx = edata.obs[group_key] == group
            var_values[group_idx] = scale_func(var_values[group_idx])

    edata.X = edata.X.astype(var_values.dtype)
    edata[:, vars].X = var_values

    _record_norm(edata, vars, norm_name)

    if copy:
        return edata
    else:
        return None


@singledispatch
def _scale_norm_function(arr):
    _raise_array_type_not_implemented(_scale_norm_function, type(arr))


@_scale_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.StandardScaler(**kwargs).fit_transform


@_scale_norm_function.register
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.StandardScaler(**kwargs).fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def scale_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.StandardScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.StandardScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.StandardScaler.html for details.

    Args:
        edata: Data object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the StandardScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.scale_norm(edata, copy=True)
    """
    scale_func = _scale_norm_function(edata.X, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="scale",
    )


@singledispatch
def _minmax_norm_function(arr):
    _raise_array_type_not_implemented(_minmax_norm_function, type(arr))


@_minmax_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.MinMaxScaler(**kwargs).fit_transform


@_minmax_norm_function.register
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.MinMaxScaler(**kwargs).fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def minmax_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MinMaxScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.MinMaxScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.MinMaxScaler.html for details.

    Args:
        edata: Data object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the MinMaxScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.minmax_norm(edata, copy=True)
    """
    scale_func = _minmax_norm_function(edata.X, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="minmax",
    )


@singledispatch
def _maxabs_norm_function(arr):
    _raise_array_type_not_implemented(_scale_norm_function, type(arr))


@_maxabs_norm_function.register
def _(arr: np.ndarray):
    return sklearn_pp.MaxAbsScaler().fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def maxabs_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MaxAbsScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html for details.

    Args:
        edata: Data object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.maxabs_norm(edata, copy=True)
    """
    scale_func = _maxabs_norm_function(edata.X)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="maxabs",
    )


@singledispatch
def _robust_scale_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_robust_scale_norm_function, type(arr))


@_robust_scale_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.RobustScaler(**kwargs).fit_transform


@_robust_scale_norm_function.register
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.RobustScaler(**kwargs).fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def robust_scale_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.RobustScaler`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.RobustScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.RobustScaler.html for details.

    Args:
        edata: Data object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the RobustScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated AnnData object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.robust_scale_norm(edata, copy=True)
    """
    scale_func = _robust_scale_norm_function(edata.X, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="robust_scale",
    )


@singledispatch
def _quantile_norm_function(arr):
    _raise_array_type_not_implemented(_quantile_norm_function, type(arr))


@_quantile_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.QuantileTransformer(**kwargs).fit_transform


@_quantile_norm_function.register
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.QuantileTransformer(**kwargs).fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def quantile_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.QuantileTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.QuantileTransformer`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.QuantileTransformer.html for details.

    Args:
        edata: Data object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the QuantileTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated data object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.quantile_norm(edata, copy=True)
    """
    scale_func = _quantile_norm_function(edata.X, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="quantile",
    )


@singledispatch
def _power_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_power_norm_function, type(arr))


@_power_norm_function.register
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.PowerTransformer(**kwargs).fit_transform


@use_ehrdata(deprecated_after="1.0.0")
def power_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    group_key: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.PowerTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html for details.

    Args:
        edata: Data object containing X to normalize values in.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the PowerTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated data object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.power_norm(edata, copy=True)
    """
    scale_func = _power_norm_function(edata.X, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        copy=copy,
        norm_name="power",
    )


@use_ehrdata(deprecated_after="1.0.0")
def log_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    copy: bool = False,
) -> EHRData | AnnData | None:
    r"""Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`.

    Args:
        edata: Data object containing X to normalize values in. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated data object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> edata_norm = ep.pp.log_norm(edata, copy=True)
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(edata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(edata, vars)

    edata = _prep_edata_norm(edata, copy)

    edata_to_check_for_negatives = edata[:, vars] if vars else edata
    offset_tmp_applied = edata_to_check_for_negatives.X + offset
    if np.any(offset_tmp_applied < 0):
        raise ValueError(
            "Matrix X contains negative values. "
            "Undefined behavior for log normalization. "
            "Please specifiy a higher offset to this function "
            "or offset negative values with ep.pp.offset_negative_values()."
        )

    var_values = edata[:, vars].X.copy()

    if offset == 1:
        np.log1p(var_values, out=var_values)
    else:
        var_values = var_values + offset
        np.log(var_values, out=var_values)

    if base is not None:
        np.divide(var_values, np.log(base), out=var_values)

    edata.X = edata.X.astype(var_values.dtype)
    edata[:, vars].X = var_values

    _record_norm(edata, vars, "log")

    return edata


def _prep_edata_norm(edata: EHRData | AnnData, copy: bool = False) -> EHRData | AnnData | None:  # pragma: no cover
    if copy:
        edata = edata.copy()

    if "raw_norm" not in edata.layers.keys():
        edata.layers["raw_norm"] = edata.X.copy()

    return edata


def _record_norm(edata: EHRData | AnnData, vars: Sequence[str], method: str) -> None:
    if "normalization" in edata.uns_keys():
        norm_record = edata.uns["normalization"]
    else:
        norm_record = {}

    for var in vars:
        if var in norm_record.keys():
            norm_record[var].append(method)
        else:
            norm_record[var] = [method]

    edata.uns["normalization"] = norm_record

    return None


@use_ehrdata(deprecated_after="1.0.0")
def offset_negative_values(edata: EHRData | AnnData, layer: str = None, copy: bool = False) -> EHRData | AnnData | None:
    """Offsets negative values into positive ones with the lowest negative value becoming 0.

    This is primarily used to enable the usage of functions such as log_norm that
    do not allow negative values for mathematical or technical reasons.

    Args:
        edata: Data object containing X to normalize values in.
        layer: The layer to offset.
        copy: Whether to return a modified copy of the data object.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated data object.
    """
    if copy:
        edata = edata.copy()

    if layer:
        minimum = np.min(edata[layer])
        if minimum < 0:
            edata[layer] = edata[layer] + np.abs(minimum)
    else:
        minimum = np.min(edata.X)
        if minimum < 0:
            edata.X = edata.X + np.abs(minimum)

    return edata if copy else None

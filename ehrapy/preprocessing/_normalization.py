from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import sklearn.preprocessing as sklearn_pp
from ehrdata.core.constants import NUMERIC_TAG

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented, function_2D_only, use_ehrdata
from ehrapy.anndata.anndata_ext import (
    _assert_numeric_vars,
    _get_var_indices_for_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd
    from anndata import AnnData
    from ehrdata import EHRData


def _get_target_layer(edata: EHRData | AnnData, layer: str | None) -> tuple[np.ndarray, str]:
    """Get the target data layer and its effective name.

    Returns:
        tuple: (data_array, effective_layer_name)
    """
    if layer is None:
        if hasattr(edata, "R") and edata.R is not None:
            return edata.R, "R"
        else:
            return edata.X, "X"
    else:
        return edata.layers[layer], layer


def _set_target_layer(
    edata: EHRData | AnnData, data: np.ndarray, layer_name: str, var_indices: Sequence[int] | Sequence[str]
) -> None:
    """Write normalized data back to the target layer."""
    if layer_name == "R":
        edata.R[:, var_indices, :] = data.astype(edata.R.dtype)
    elif layer_name == "X":
        edata.X = edata.X.astype(data.dtype)
        edata[:, var_indices].X = data
    else:
        if data.ndim == 3:
            edata.layers[layer_name][:, var_indices, :] = data.astype(edata.layers[layer_name].dtype)
        else:
            edata.layers[layer_name] = edata.layers[layer_name].astype(data.dtype)
            edata[:, var_indices].layers[layer_name] = data


def _normalize_3d_data(
    data: np.ndarray,
    var_indices: list[int],
    scale_func: Callable[[np.ndarray], np.ndarray],
    group_key: str | None,
    edata: EHRData | AnnData,
) -> np.ndarray:
    """Apply normalization to 3D data (n_obs x n_var x n_timestamps)."""
    var_values = data[:, var_indices, :]
    n_obs, n_var_selected, n_timestamps = var_values.shape

    if group_key is None:
        for var_idx in range(n_var_selected):
            var_data = var_values[:, var_idx, :].reshape(-1, 1)
            var_data = scale_func(var_data)
            var_values[:, var_idx, :] = var_data.reshape(n_obs, n_timestamps)
    else:
        for group in edata.obs[group_key].unique():
            group_idx = edata.obs[group_key] == group
            group_data = var_values[group_idx]
            n_obs_group = group_data.shape[0]
            for var_idx in range(n_var_selected):
                var_data = group_data[:, var_idx, :].reshape(-1, 1)
                var_data = scale_func(var_data)
                var_values[group_idx, var_idx, :] = var_data.reshape(n_obs_group, n_timestamps)

    return var_values


def _normalize_2d_data(
    edata: EHRData | AnnData, vars: Sequence[str], scale_func: Callable[[np.ndarray], np.ndarray], group_key: str | None
) -> np.ndarray:
    """Apply normalization to 2D data (n_obs × n_var)."""
    var_values = edata[:, vars].X.copy()

    if group_key is None:
        var_values = scale_func(var_values)
    else:
        for group in edata.obs[group_key].unique():
            group_idx = edata.obs[group_key] == group
            var_values[group_idx] = scale_func(var_values[group_idx])

    return var_values


def _scale_func_group(
    edata: EHRData | AnnData,
    scale_func: Callable[[np.ndarray | pd.DataFrame], np.ndarray],
    vars: str | Sequence[str] | None,
    group_key: str | None,
    layer: str | None,
    copy: bool,
    norm_name: str,
) -> EHRData | AnnData | None:
    """Apply scaling function to selected columns of edata, either globally or per group.

    Supports both 2D and 3D data with unified layer handling.
    """
    if group_key is not None and group_key not in edata.obs_keys():
        raise KeyError(f"group key '{group_key}' not found in edata.obs.")

    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(edata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(edata, vars)

    edata = _prep_edata_norm(edata, copy)

    target_data, layer_name = _get_target_layer(edata, layer)

    if target_data.ndim == 3:
        from ehrapy.anndata.anndata_ext import _get_var_indices

        var_indices = _get_var_indices(edata, vars)
        normalized_data = _normalize_3d_data(target_data, var_indices, scale_func, group_key, edata)
        _set_target_layer(edata, normalized_data, layer_name, var_indices)

    elif target_data.ndim == 2:
        normalized_data = _normalize_2d_data(edata, vars, scale_func, group_key)
        _set_target_layer(edata, normalized_data, layer_name, vars)

    else:
        raise ValueError(f"Unsupported data dimensionality: {target_data.ndim}D. Expected 2D or 3D data.")

    _record_norm(edata, vars, norm_name)

    return edata if copy else None


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
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.StandardScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.StandardScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.StandardScaler.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the StandardScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmean(edata.R)
        120.142281
        >>> ep.pp.scale_norm(edata)
        >>> np.nanmean(edata.R)
        0

    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _scale_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
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
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply min-max normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MinMaxScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.MinMaxScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.MinMaxScaler.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the MinMaxScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmin(edata.R), np.nanmax(edata.R)
        (8, 4695)
        >>> ep.pp.minmax_norm(edata)
        >>> np.nanmin(edata.R), np.nanmax(edata.R)
        (0, 1)
    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _minmax_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
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
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Apply max-abs normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.MaxAbsScaler`, see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmax(np.abs(edata.R))
        4695
        >>> ep.pp.maxabs_norm(edata)
        >>> np.nanmax(np.abs(edata.R))
        1
    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _maxabs_norm_function(arr)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
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
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply robust scaling normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.RobustScaler`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.RobustScaler`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.RobustScaler.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the RobustScaler.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmedian(edata.R)
        82
        >>> ep.pp.robust_scale_norm(edata)
        >>> np.nanmedian(edata.R)
        0
    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _robust_scale_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
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
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply quantile normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.QuantileTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html for details.
    If `edata.X` is a Dask Array, functionality is provided by :class:`~dask_ml.preprocessing.QuantileTransformer`, see https://ml.dask.org/modules/generated/dask_ml.preprocessing.QuantileTransformer.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the QuantileTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmin(edata.R), np.nanmax(edata.R)
        (8, 4695)
        >>> ep.pp.quantile_norm(edata)
        >>> np.nanmin(edata.R), np.nanmax(edata.R)
        (0, 1)
    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _quantile_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
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
    layer: str | None = None,
    copy: bool = False,
    **kwargs,
) -> EHRData | AnnData | None:
    """Apply power transformation normalization.

    Functionality is provided by :class:`~sklearn.preprocessing.PowerTransformer`,
    see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html for details.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Per-variable normalization across samples and timestamps

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        group_key: Key in edata.obs that contains group information. If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.
        **kwargs: Additional arguments passed to the PowerTransformer.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> from scipy import stats
        >>> edata = ed.dt.physionet2012()
        >>> stats.skew(edata.R.flatten())
        13.528100
        >>> ep.pp.power_norm(edata)
        >>> stats.skew(edata.R.flatten())
        -0.041263
    """
    arr, _ = _get_target_layer(edata, layer)
    scale_func = _power_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
        copy=copy,
        norm_name="power",
    )


@use_ehrdata(deprecated_after="1.0.0")
def log_norm(
    edata: EHRData | AnnData,
    vars: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    r"""Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`.

    Supports both 2D and 3D data:

    - 2D data: Standard normalization across observations
    - 3D data: Applied to all elements across samples and timestamps

    Args:
        edata: Central data object.
        vars: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        base: Numeric base for logarithm. If None the natural logarithm is used.
        offset: Offset added to values before computing the logarithm.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmax(edata.R)
        4695
        >>> ep.pp.log_norm(edata)
        >>> np.nanmax(edata.R)
        8.454679
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(edata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(edata, vars)

    edata = _prep_edata_norm(edata, copy)

    arr, layer_name = _get_target_layer(edata, layer)
    is_3d = arr.ndim == 3

    if vars:
        if is_3d:
            var_indices = [edata.var_names.get_loc(v) for v in vars]
            check_data = arr[:, var_indices, :]
        else:
            edata_to_check_for_negatives = edata[:, vars]
            check_data = edata_to_check_for_negatives.X
    else:
        check_data = arr

    offset_tmp_applied = check_data + offset
    if np.any(offset_tmp_applied < 0):
        data_type = "Matrix R" if layer_name == "R" else "Layer" if layer else "Matrix X"
        raise ValueError(
            f"{data_type} contains negative values. "
            "Undefined behavior for log normalization. "
            "Please specify a higher offset to this function "
            "or offset negative values with ep.pp.offset_negative_values()."
        )

    if is_3d or layer:
        var_values = arr.copy()
    else:
        if vars:
            var_values = edata[:, vars].X.copy()
        else:
            var_values = arr.copy()

    if offset == 1:
        np.log1p(var_values, out=var_values)
    else:
        var_values = var_values + offset
        np.log(var_values, out=var_values)

    if base is not None:
        np.divide(var_values, np.log(base), out=var_values)

    if layer_name == "R":
        edata.R = edata.R.astype(var_values.dtype)
        if vars:
            var_indices = [edata.var_names.get_loc(v) for v in vars]
            edata.R[:, var_indices, :] = var_values
        else:
            edata.R[:, :, :] = var_values
    elif layer:
        edata.layers[layer] = edata.layers[layer].astype(var_values.dtype)
        if is_3d:
            if vars:
                var_indices = [edata.var_names.get_loc(v) for v in vars]
                edata.layers[layer][:, var_indices, :] = var_values
            else:
                edata.layers[layer][:, :, :] = var_values
        else:
            if vars:
                edata[:, vars].layers[layer] = var_values
            else:
                edata.layers[layer] = var_values
    else:
        edata.X = edata.X.astype(var_values.dtype)
        if vars:
            edata[:, vars].X = var_values
        else:
            edata.X = var_values

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

    Supports both 2D and 3D data:

    - 2D data: Standard offset across observations
    - 3D data: Applied to all elements across samples and timestamps

    Args:
        edata: Central data object.
        layer: The layer to offset.
        copy: Whether to return a modified copy of the data object.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> edata_shifted = edata.copy()
        >>> edata_shifted.R = edata_shifted.R - 2.0
        >>> np.nanmean(edata_shifted.R)
        118.142281
        >>> ep.pp.offset_negative_values(edata_shifted)
        >>> np.nanmean(edata_shifted.R)
        137.942281
    """
    if copy:
        edata = edata.copy()

    arr, layer_name = _get_target_layer(edata, layer)
    minimum = np.nanmin(arr)
    if minimum < 0:
        offset_arr = arr + np.abs(minimum)
        if layer_name == "R":
            edata.R = offset_arr
        elif layer:
            edata.layers[layer] = offset_arr
        else:
            edata.X = offset_arr

    return edata if copy else None

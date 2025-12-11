from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import sklearn.preprocessing as sklearn_pp
from ehrdata.core.constants import NUMERIC_TAG

from ehrapy._compat import (
    DaskArray,
    _apply_over_time_axis,
    _raise_array_type_not_implemented,
    use_ehrdata,
)
from ehrapy.anndata.anndata_ext import (
    _assert_numeric_vars,
    _get_var_indices,
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

    var_indices = _get_var_indices(edata, vars)
    X = edata.X if layer is None else edata.layers[layer]

    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32)

    if group_key is None:
        X[:, var_indices] = scale_func(X[:, var_indices])

    else:
        # Group-wise normalization is not supported for Dask arrays
        if isinstance(X, DaskArray):
            raise NotImplementedError(
                f"Group-wise normalization with group_key='{group_key}' does not support array type {type(X)}. "
                "Please convert to numpy array first or use normalization without group_key."
            )

        for group in edata.obs[group_key].unique():
            group_mask = np.where(edata.obs[group_key] == group)[0]
            X[np.ix_(group_mask, var_indices)] = scale_func(X[np.ix_(group_mask, var_indices)])

    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    _record_norm(edata, vars, norm_name)

    return edata if copy else None


@singledispatch
def _scale_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_scale_norm_function, type(arr))


@_scale_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.StandardScaler(**kwargs).fit_transform(arr)


@_scale_norm_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.StandardScaler(**kwargs).fit_transform(arr)


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmean(edata.layers["tem_data"])
        74.213570
        >>> ep.pp.scale_norm(edata, layer="tem_data")
        >>> np.nanmean(edata.layers["tem_data"])
        0.0

    """
    scale_func = lambda arr: _scale_norm_function(arr, **kwargs)

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
def _minmax_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_minmax_norm_function, type(arr))


@_minmax_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.MinMaxScaler(**kwargs).fit_transform(arr)


@_minmax_norm_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.MinMaxScaler(**kwargs).fit_transform(arr)


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"]), np.nanmax(edata.layers["tem_data"])
        (-17.8, 36400.0)
        >>> ep.pp.minmax_norm(edata, layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"]), np.nanmax(edata.layers["tem_data"])
        (0.0, 1.0)
    """
    scale_func = lambda arr: _minmax_norm_function(arr, **kwargs)

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
    _raise_array_type_not_implemented(_maxabs_norm_function, type(arr))


@_maxabs_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray):
    return sklearn_pp.MaxAbsScaler().fit_transform(arr)


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
    Note: Dask arrays are not supported for this function. Please convert to numpy array first.

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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmax(np.abs(edata.layers["tem_data"]))
        36400.0
        >>> ep.pp.maxabs_norm(edata, layer="tem_data")
        >>> np.nanmax(np.abs(edata.layers["tem_data"]))
        1.0
    """
    X = edata.X if layer is None else edata.layers[layer]
    if isinstance(X, DaskArray):
        _raise_array_type_not_implemented(_maxabs_norm_function, type(X))
    scale_func = _maxabs_norm_function

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


@_robust_scale_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.RobustScaler(**kwargs).fit_transform(arr)


@_robust_scale_norm_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.RobustScaler(**kwargs).fit_transform(arr)


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmedian(edata.layers["tem_data"])
        69.0
        >>> ep.pp.robust_scale_norm(edata, layer="tem_data")
        >>> np.nanmedian(edata.layers["tem_data"])
        0.0
    """
    scale_func = lambda arr: _robust_scale_norm_function(arr, **kwargs)

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
def _quantile_norm_function(arr, **kwargs):
    _raise_array_type_not_implemented(_quantile_norm_function, type(arr))


@_quantile_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.QuantileTransformer(**kwargs).fit_transform(arr)


@_quantile_norm_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, **kwargs):
    import dask_ml.preprocessing as daskml_pp

    return daskml_pp.QuantileTransformer(**kwargs).fit_transform(arr)


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"]), np.nanmax(edata.layers["tem_data"])
        (-17.8, 36400.0)
        >>> ep.pp.quantile_norm(edata, layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"]), np.nanmax(edata.layers["tem_data"])
        (0.0, 1.0)
    """
    scale_func = lambda arr: _quantile_norm_function(arr, **kwargs)

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


@_power_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, **kwargs):
    return sklearn_pp.PowerTransformer(**kwargs).fit_transform(arr)


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
    Note: Dask arrays are not supported for this function. Please convert to numpy array first.

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
        >>> import numpy as np
        >>> from scipy import stats
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> ep.pp.offset_negative_values(edata, layer="tem_data")
        >>> skewed_data = np.power(edata.layers["tem_data"], 2)
        >>> edata.layers["tem_data"] = skewed_data
        >>> stats.skew(edata.layers["tem_data"].flatten())
        504.250727
        >>> ep.pp.power_norm(edata, layer="tem_data")
        >>> stats.skew(edata.layers["tem_data"].flatten())
        0.144324
    """
    X = edata.X if layer is None else edata.layers[layer]
    if isinstance(X, DaskArray):
        _raise_array_type_not_implemented(_power_norm_function, type(X))
    scale_func = lambda arr: _power_norm_function(arr, **kwargs)

    return _scale_func_group(
        edata=edata,
        scale_func=scale_func,
        vars=vars,
        group_key=group_key,
        layer=layer,
        copy=copy,
        norm_name="power",
    )


@singledispatch
def _log_norm_function(arr, offset: int | float = 1, base: int | float | None = None):
    _raise_array_type_not_implemented(_log_norm_function, type(arr))


@_log_norm_function.register(np.ndarray)
@_apply_over_time_axis
def _(arr: np.ndarray, offset: int | float = 1, base: int | float | None = None) -> np.ndarray:
    if offset == 1:
        np.log1p(arr, out=arr)
    else:
        np.add(arr, offset, out=arr)
        np.log(arr, out=arr)

    if base is not None:
        np.divide(arr, np.log(base), out=arr)

    return arr


@_log_norm_function.register(DaskArray)
@_apply_over_time_axis
def _(arr: DaskArray, offset: int | float = 1, base: int | float | None = None) -> DaskArray:
    import dask.array as da

    if offset == 1:
        result = da.log1p(arr)
    else:
        result = da.log(arr + offset)

    if base is not None:
        result = result / np.log(base)

    return result


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> ep.pp.offset_negative_values(edata, layer="tem_data")
        >>> np.nanmax(edata.layers["tem_data"])
        36400.0
        >>> ep.pp.log_norm(edata, layer="tem_data")
        >>> np.nanmax(edata.layers["tem_data"])
        10.502379
    """
    if isinstance(vars, str):
        vars = [vars]
    if vars is None:
        vars = _get_var_indices_for_type(edata, NUMERIC_TAG)
    else:
        _assert_numeric_vars(edata, vars)

    edata = _prep_edata_norm(edata, copy)

    X = edata.X if layer is None else edata.layers[layer]

    if vars:
        var_indices = _get_var_indices(edata, vars)
        check_data = X[:, var_indices] if X.ndim == 2 else X[:, var_indices, :]
    else:
        check_data = X

    offset_tmp_applied = check_data + offset
    if np.any(offset_tmp_applied < 0):
        data_type = f"Layer '{layer}'" if layer else "Matrix X"
        raise ValueError(
            f"{data_type} contains negative values. "
            "Undefined behavior for log normalization. "
            "Please specify a higher offset to this function "
            "or offset negative values with ep.pp.offset_negative_values()."
        )

    if vars:
        var_indices = _get_var_indices(edata, vars)
        var_values = X[:, var_indices] if X.ndim == 2 else X[:, var_indices, :]
        transformed_values = _log_norm_function(var_values, offset=offset, base=base)
        if layer is None:
            edata.X[:, var_indices] = transformed_values
        else:
            if X.ndim == 3:
                edata.layers[layer][:, var_indices, :] = transformed_values
            else:
                edata.layers[layer][:, var_indices] = transformed_values
    else:
        transformed_values = _log_norm_function(X, offset=offset, base=base)
        if layer is None:
            edata.X = transformed_values
        else:
            edata.layers[layer] = transformed_values

    _record_norm(edata, vars, "log")

    return edata if copy else None


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
        >>> edata = ed.dt.physionet2012(layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"])
        -17.8
        >>> ep.pp.offset_negative_values(edata, layer="tem_data")
        >>> np.nanmin(edata.layers["tem_data"])
        0.0
    """
    edata = _prep_edata_norm(edata, copy)

    X = edata.X if layer is None else edata.layers[layer]
    minimum = np.nanmin(X)
    if minimum < 0:
        np.add(X, np.abs(minimum), out=X)

    return edata if copy else None

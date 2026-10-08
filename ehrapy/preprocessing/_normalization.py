from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import array_api_extra as xpx
import ehrdata as ed
import numpy as np
import pandas as pd
import sklearn.preprocessing as sklearn_pp
from array_api_compat import array_namespace, is_lazy_array
from ehrdata.core.constants import FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils.types import CSBase, DaskArray
from scipy.special import ndtri

from ehrapy._compat import (
    _by_group,
    _columnwise,
    _obs_axes,
    _raise_densifying,
    _raise_if_dask_with_sparse_chunks,
    _set_columns,
    _sparse_columns,
    _sparse_rows,
    nanquantile,
    nanstd,
    sparse_nan_min_max,
    sparse_nan_moments,
    sparse_nanquantile,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ehrdata import EHRData

    type Array = np.ndarray | DaskArray
    type Params = tuple[Array | None, Array | None]


def _scale_func_group(
    edata: EHRData,
    transform: Callable[[Array | CSBase, np.ndarray | None], Array | CSBase],
    var_names: str | Sequence[str] | None,
    groupby: str | None,
    layer: str | None,
    copy: bool,
    norm_name: str,
) -> EHRData | None:
    """Apply a per-variable transformation to the selected numeric variables, either globally or per group."""
    if groupby is not None:
        if groupby not in edata.obs:
            raise KeyError(f"groupby key '{groupby}' not found in edata.obs.")
        if edata.obs[groupby].isna().any():
            raise ValueError(f"groupby key '{groupby}' contains missing values.")
    if copy:
        edata = edata.copy()
    X = edata.X if layer is None else edata.layers[layer]
    _raise_if_dask_with_sparse_chunks(X, norm_name)
    if FEATURE_TYPE_KEY not in edata.var.columns:
        if is_lazy_array(X):
            raise ValueError(
                f"{norm_name} needs feature types in `edata.var`. "
                "Infer them first with `ed.infer_feature_types(edata)`, which reads every value once."
            )
        ed.infer_feature_types(edata, layer=layer, output=None)

    var_names = _numeric_var_names(edata, var_names)
    var_indices = edata.var_names.get_indexer(var_names)
    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32)

    groups = None if groupby is None else pd.factorize(edata.obs[groupby])[0]
    X = _set_columns(X, var_indices, transform(X[:, var_indices], groups))

    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    _record_norm(edata, var_names, norm_name)

    return edata if copy else None


def _numeric_var_names(edata: EHRData, var_names: str | Sequence[str] | None) -> list[str]:
    numeric_vars = edata.var_names[edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG].tolist()
    if var_names is None:
        return numeric_vars
    var_names = [var_names] if isinstance(var_names, str) else list(var_names)
    if not set(var_names) <= set(numeric_vars):
        raise ValueError("Some selected vars are not numeric")
    return var_names


def _nonzero(scale: Array) -> Array:
    """Replace zero scales by one so that constant variables stay unchanged, as scikit-learn does."""
    xp = array_namespace(scale)
    return xp.where(scale == 0, xp.ones_like(scale), scale)


def _affine(X: Array, groups: np.ndarray | None, params: Callable[[Array], Params]) -> Array:
    """Compute `(X - shift) / scale` with per-variable parameters, estimated per group if `groups` is given."""
    shift, scale = _by_group(X, groups, params)
    if shift is not None:
        X = X - shift
    if scale is not None:
        X = X / scale
    return X


def _sparse_scale(X: CSBase, groups: np.ndarray | None, scale: Callable[[CSBase], np.ndarray]) -> CSBase:
    """Divide every variable of a sparse matrix by a per-variable scale, estimated per group if `groups` is given."""
    X = X.astype(np.result_type(X.dtype, np.float32))
    columns = _sparse_columns(X)
    if groups is None:
        X.data /= scale(X)[columns]
    else:
        scales = np.stack([scale(X[groups == group]) for group in range(groups.max() + 1)])
        X.data /= scales[groups[_sparse_rows(X)], columns]
    return X


@singledispatch
def _scale(X: Array, groups: np.ndarray | None, *, with_mean: bool, with_std: bool) -> Array:
    def params(x: Array) -> Params:
        axes = _obs_axes(x)
        return (
            xpx.nanmean(x, axis=axes) if with_mean else None,
            _nonzero(nanstd(x, axis=axes)) if with_std else None,
        )

    return _affine(X, groups, params)


@_scale.register(CSBase)
def _(X: CSBase, groups: np.ndarray | None, *, with_mean: bool, with_std: bool) -> CSBase:
    if with_mean:
        _raise_densifying("scale_norm with `with_mean=True`", "centering shifts implicit zeros")
    if not with_std:
        return X
    return _sparse_scale(X, groups, lambda x: _nonzero(np.sqrt(sparse_nan_moments(x)[2])))


def scale_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    with_mean: bool = True,
    with_std: bool = True,
    copy: bool = False,
) -> EHRData | None:
    """Apply scaling normalization.

    Standardizes every variable by subtracting its mean and dividing by its standard deviation, ignoring missing values, like :class:`~sklearn.preprocessing.StandardScaler`.
    For 3D data, the statistics of a variable are computed across observations and timepoints.

    Args:
        edata: Central data object. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        with_mean: Whether to center the variables.
        with_std: Whether to scale the variables to unit variance.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmean(edata.X)
        74.194793
        >>> ep.pp.scale_norm(edata)
        >>> np.nanmean(edata.X)
        0.0

    """
    return _scale_func_group(
        edata=edata,
        transform=lambda X, groups: _scale(X, groups, with_mean=with_mean, with_std=with_std),
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="scale",
    )


@singledispatch
def _minmax(X: Array, groups: np.ndarray | None, *, feature_range: tuple[float, float]) -> Array:
    low, high = feature_range

    def params(x: Array) -> Params:
        axes = _obs_axes(x)
        minimum = xpx.nanmin(x, axis=axes)
        scale = _nonzero(xpx.nanmax(x, axis=axes) - minimum) / (high - low)
        return minimum - low * scale, scale

    return _affine(X, groups, params)


@_minmax.register(CSBase)
def _(X: CSBase, groups: np.ndarray | None, *, feature_range: tuple[float, float]) -> CSBase:
    _raise_densifying("minmax_norm", "shifting by the minimum moves implicit zeros; use maxabs_norm instead")


def minmax_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    feature_range: tuple[float, float] = (0.0, 1.0),
    copy: bool = False,
) -> EHRData | None:
    """Apply min-max normalization.

    Rescales every variable to `feature_range`, ignoring missing values, like :class:`~sklearn.preprocessing.MinMaxScaler`.
    For 3D data, the statistics of a variable are computed across observations and timepoints.

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        feature_range: Desired range of the transformed data.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmin(edata.X), np.nanmax(edata.X)
        (-17.8, 36400.0)
        >>> ep.pp.minmax_norm(edata)
        >>> np.nanmin(edata.X), np.nanmax(edata.X)
        (0.0, 1.0)
    """
    return _scale_func_group(
        edata=edata,
        transform=lambda X, groups: _minmax(X, groups, feature_range=feature_range),
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="minmax",
    )


@singledispatch
def _maxabs(X: Array, groups: np.ndarray | None) -> Array:
    def params(x: Array) -> Params:
        return None, _nonzero(xpx.nanmax(array_namespace(x).abs(x), axis=_obs_axes(x)))

    return _affine(X, groups, params)


@_maxabs.register(CSBase)
def _(X: CSBase, groups: np.ndarray | None) -> CSBase:
    def scale(x: CSBase) -> np.ndarray:
        minimum, maximum = sparse_nan_min_max(x)
        return _nonzero(np.maximum(np.abs(minimum), np.abs(maximum)))

    return _sparse_scale(X, groups, scale)


def maxabs_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Apply max-abs normalization.

    Divides every variable by its maximum absolute value, ignoring missing values, like :class:`~sklearn.preprocessing.MaxAbsScaler`.
    For 3D data, the statistics of a variable are computed across observations and timepoints.

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmax(np.abs(edata.X))
        36400.0
        >>> ep.pp.maxabs_norm(edata)
        >>> np.nanmax(np.abs(edata.X))
        1.0
    """
    return _scale_func_group(
        edata=edata,
        transform=_maxabs,
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="maxabs",
    )


def _robust_scale_adjustment(quantile_range: tuple[float, float], unit_variance: bool) -> float:
    low, high = quantile_range
    return float(ndtri(high / 100) - ndtri(low / 100)) if unit_variance else 1.0


@singledispatch
def _robust_scale(
    X: Array,
    groups: np.ndarray | None,
    *,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: tuple[float, float],
    unit_variance: bool,
) -> Array:
    low, high = quantile_range
    adjustment = _robust_scale_adjustment(quantile_range, unit_variance)

    def params(x: Array) -> Params:
        quantiles = nanquantile(x, [low / 100, 0.5, high / 100], axis=_obs_axes(x))
        return (
            quantiles[1] if with_centering else None,
            _nonzero(quantiles[2] - quantiles[0]) / adjustment if with_scaling else None,
        )

    return _affine(X, groups, params)


@_robust_scale.register(CSBase)
def _(
    X: CSBase,
    groups: np.ndarray | None,
    *,
    with_centering: bool,
    with_scaling: bool,
    quantile_range: tuple[float, float],
    unit_variance: bool,
) -> CSBase:
    if with_centering:
        _raise_densifying("robust_scale_norm with `with_centering=True`", "centering shifts implicit zeros")
    if not with_scaling:
        return X
    low, high = quantile_range
    adjustment = _robust_scale_adjustment(quantile_range, unit_variance)

    def scale(x: CSBase) -> np.ndarray:
        quantiles = sparse_nanquantile(x, [low / 100, high / 100])
        return _nonzero(quantiles[1] - quantiles[0]) / adjustment

    return _sparse_scale(X, groups, scale)


def robust_scale_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    with_centering: bool = True,
    with_scaling: bool = True,
    quantile_range: tuple[float, float] = (25.0, 75.0),
    unit_variance: bool = False,
    copy: bool = False,
) -> EHRData | None:
    """Apply robust scaling normalization.

    Subtracts the median of every variable and divides by its interquantile range, ignoring missing values, like :class:`~sklearn.preprocessing.RobustScaler`.
    For 3D data, the statistics of a variable are computed across observations and timepoints.

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        with_centering: Whether to subtract the median.
        with_scaling: Whether to divide by the interquantile range.
        quantile_range: Lower and upper percentile of the interquantile range.
        unit_variance: Whether to scale normally distributed variables to unit variance.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmedian(edata.X)
        69.0
        >>> ep.pp.robust_scale_norm(edata)
        >>> np.nanmedian(edata.X)
        0.0
    """
    return _scale_func_group(
        edata=edata,
        transform=lambda X, groups: _robust_scale(
            X,
            groups,
            with_centering=with_centering,
            with_scaling=with_scaling,
            quantile_range=quantile_range,
            unit_variance=unit_variance,
        ),
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="robust_scale",
    )


def _dense_only(name: str, reason: str, transform: Callable[[Array, np.ndarray | None], Array]):
    def checked(X: Array | CSBase, groups: np.ndarray | None) -> Array:
        if isinstance(X, CSBase):
            _raise_densifying(name, reason)
        return transform(X, groups)

    return checked


def quantile_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    n_quantiles: int = 1000,
    output_distribution: Literal["uniform", "normal"] = "uniform",
    subsample: int = 10_000,
    random_state: int | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Apply quantile normalization.

    Maps every variable to a uniform or normal distribution with :class:`~sklearn.preprocessing.QuantileTransformer`, ignoring missing values.
    For 3D data, a variable's quantiles are computed across observations and timepoints.

    Args:
        edata: Central data object. Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        n_quantiles: Number of quantiles used to discretize the cumulative distribution function.
        output_distribution: Marginal distribution of the transformed data.
        subsample: Maximum number of samples used to estimate the quantiles.
        random_state: Seed for subsampling.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> edata = ed.dt.physionet2012()
        >>> np.nanmin(edata.X), np.nanmax(edata.X)
        (-17.8, 36400.0)
        >>> ep.pp.quantile_norm(edata)
        >>> np.nanmin(edata.X), np.nanmax(edata.X)
        (0.0, 1.0)
    """

    def kernel(x: np.ndarray) -> np.ndarray:
        return sklearn_pp.QuantileTransformer(
            n_quantiles=min(n_quantiles, x.shape[0]),
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=random_state,
        ).fit_transform(x)

    return _scale_func_group(
        edata=edata,
        transform=_dense_only(
            "quantile_norm", "it maps zeros to nonzero values", lambda X, groups: _columnwise(X, groups, kernel)
        ),
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="quantile",
    )


def power_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    groupby: str | None = None,
    layer: str | None = None,
    method: Literal["yeo-johnson", "box-cox"] = "yeo-johnson",
    standardize: bool = True,
    copy: bool = False,
) -> EHRData | None:
    """Apply power transformation normalization.

    Makes every variable more Gaussian-like with :class:`~sklearn.preprocessing.PowerTransformer`, ignoring missing values.
    For 3D data, a variable's transformation is fitted across observations and timepoints.

    Args:
        edata: Central data object.
               Must already be encoded using :func:`~ehrapy.preprocessing.encode`.
        var_names: List of the names of the numeric variables to normalize.
              If None all numeric variables will be normalized.
        groupby: Key in edata.obs that contains group information.
                 If provided, scaling is applied per group.
        layer: The layer to normalize.
        method: The power transform method.
            'box-cox' requires strictly positive data.
        standardize: Whether to center and scale the transformed data to zero mean and unit variance.
        copy: Whether to return a copy or act in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object. Also stores a record of applied normalizations as a dictionary in edata.uns["normalization"].

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> from scipy import stats
        >>> edata = ed.dt.physionet2012()
        >>> ep.pp.offset_negative_values(edata)
        >>> skewed_data = np.power(edata.X, 2)
        >>> edata.X = skewed_data
        >>> stats.skew(edata.X.flatten(), nan_policy="omit")
        503.071351
        >>> ep.pp.power_norm(edata)
        >>> stats.skew(edata.X.flatten(), nan_policy="omit")
        0.017135
    """

    def kernel(x: np.ndarray) -> np.ndarray:
        return sklearn_pp.PowerTransformer(method=method, standardize=standardize).fit_transform(x)

    return _scale_func_group(
        edata=edata,
        transform=_dense_only(
            "power_norm", "it maps zeros to nonzero values", lambda X, groups: _columnwise(X, groups, kernel)
        ),
        var_names=var_names,
        groupby=groupby,
        layer=layer,
        copy=copy,
        norm_name="power",
    )


def _raise_negative(edata_part: str) -> None:
    raise ValueError(
        f"{edata_part} contains negative values. "
        "Undefined behavior for log normalization. "
        "Please specify a higher offset to this function "
        "or offset negative values with ep.pp.offset_negative_values()."
    )


@singledispatch
def _log(X: Array, *, base: float | None, offset: float, edata_part: str) -> Array:
    xp = array_namespace(X)
    if not is_lazy_array(X) and bool(xp.any(X + offset < 0)):
        _raise_negative(edata_part)
    X = xp.log1p(X) if offset == 1 else xp.log(X + offset)
    return X if base is None else X / np.log(base)


@_log.register(CSBase)
def _(X: CSBase, *, base: float | None, offset: float, edata_part: str) -> CSBase:
    if offset != 1:
        _raise_densifying("log_norm with `offset != 1`", "log(0 + offset) is nonzero")
    if np.any(X.data < -1):
        _raise_negative(edata_part)
    X = X.astype(np.result_type(X.dtype, np.float32))
    np.log1p(X.data, out=X.data)
    if base is not None:
        X.data /= np.log(base)
    return X


def log_norm(
    edata: EHRData,
    *,
    var_names: str | Sequence[str] | None = None,
    base: int | float | None = None,
    offset: int | float = 1,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    r"""Apply log normalization.

    Computes :math:`x = \\log(x + offset)`, where :math:`log` denotes the natural logarithm
    unless a different base is given and the default :math:`offset` is :math:`1`.
    Applies elementwise, so 3D data is transformed at every timepoint.

    Args:
        edata: Central data object.
        var_names: List of the names of the numeric variables to normalize.
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
        >>> ep.pp.offset_negative_values(edata)
        >>> np.nanmax(edata.X)
        36417.8
        >>> ep.pp.log_norm(edata)
        >>> np.nanmax(edata.X)
        10.502840
    """
    edata_part = "Matrix X" if layer is None else f"Layer '{layer}'"
    return _scale_func_group(
        edata=edata,
        transform=lambda X, _groups: _log(X, base=base, offset=offset, edata_part=edata_part),
        var_names=var_names,
        groupby=None,
        layer=layer,
        copy=copy,
        norm_name="log",
    )


def _record_norm(edata: EHRData, var_names: Sequence[str], method: str) -> None:
    norm_record = edata.uns.get("normalization", {})
    for var in var_names:
        norm_record.setdefault(var, []).append(method)
    edata.uns["normalization"] = norm_record


@singledispatch
def _offset_negative(X: Array) -> Array:
    return X - array_namespace(X).minimum(xpx.nanmin(X), 0)


@_offset_negative.register(CSBase)
def _(X: CSBase) -> CSBase:
    if np.nanmin(sparse_nan_min_max(X)[0]) < 0:
        _raise_densifying("offset_negative_values on data with negative values", "the offset moves implicit zeros")
    return X


def offset_negative_values(edata: EHRData, *, layer: str | None = None, copy: bool = False) -> EHRData | None:
    """Offsets negative values into positive ones with the lowest negative value becoming 0.

    This is primarily used to enable the usage of functions such as log_norm that
    do not allow negative values for mathematical or technical reasons.
    The offset is the global minimum, so 3D data is offset across all observations, variables and timepoints.

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
        >>> np.nanmin(edata.X)
        -17.8
        >>> ep.pp.offset_negative_values(edata)
        >>> np.nanmin(edata.X)
        0.0
    """
    if copy:
        edata = edata.copy()

    X = edata.X if layer is None else edata.layers[layer]
    _raise_if_dask_with_sparse_chunks(X, "offset_negative_values")
    X = _offset_negative(X)
    if layer is None:
        edata.X = X
    else:
        edata.layers[layer] = X

    return edata if copy else None

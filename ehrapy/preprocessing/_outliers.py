from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
from array_api_compat import array_namespace
from fast_array_utils.types import CSBase, DaskArray

from ehrapy._compat import (
    _broadcast_var_stat,
    _ensure_feature_types,
    _has_sparse_chunks,
    _map_variable_blocks,
    _order_statistic,
    _raise_densifying,
    _set_columns,
    _sparse_columns,
)
from ehrapy.preprocessing._normalization import _numeric_var_names

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from ehrdata import EHRData

    type Array = np.ndarray | DaskArray
    type Bounds = tuple[Array | float, Array | float]


def winsorize(
    edata: EHRData,
    *,
    var_names: Collection[str] | None = None,
    obs_cols: Collection[str] | None = None,
    limits: tuple[float, float] = (0.01, 0.01),
    inclusive: tuple[bool, bool] = (True, True),
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Returns a Winsorized version of the input array.

    Replaces the `limits` fractions of the smallest and largest values of every feature by the smallest and largest remaining value, ignoring missing values, like :func:`scipy.stats.mstats.winsorize`.
    For 3D data, the limits of a variable are computed across observations and timepoints.

    Args:
        edata: Central data object.
        var_names: The features to winsorize.
            If `None` and no `obs_cols` are given, all numeric variables are winsorized.
        obs_cols: Columns in obs with features to winsorize.
        limits: Tuple of the percentages to cut on each side of the array as floats between 0. and 1.
        inclusive: Whether the number of values cut on each side is truncated (`True`) or rounded (`False`).
        layer: The layer to operate on.
        copy: Whether to return a copy.

    Returns:
        Winsorized data object if copy is True.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.winsorize(edata, var_names=["bmi"])
    """
    if not all(0 <= limit <= 1 for limit in limits):
        raise ValueError(f"limits must be between 0 and 1, got {limits}.")
    return _clip_features(
        edata, var_names, obs_cols, layer, copy, lambda X: _winsorize_bounds(X, limits, inclusive), "winsorize"
    )


def clip_quantile(
    edata: EHRData,
    limits: tuple[float, float],
    *,
    var_names: Collection[str] | None = None,
    obs_cols: Collection[str] | None = None,
    layer: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Clips (limits) features.

    Given an interval, values outside the interval are clipped to the interval edges.
    Applies elementwise, so 3D data is clipped at every timepoint.

    Args:
        edata: Central data object.
        limits: Values outside the interval are clipped to the interval edges.
        var_names: Columns in var with features to clip.
            If `None` and no `obs_cols` are given, all numeric variables are clipped.
        obs_cols: Columns in obs with features to clip
        layer: The layer to operate on.
        copy: Whether to return a copy of data or not

    Returns:
        A copy of original data object with clipped features.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.clip_quantile(edata, limits=(0, 75), var_names=["bmi"])
    """
    return _clip_features(edata, var_names, obs_cols, layer, copy, lambda _X: limits, "clip_quantile")


def _clip_features(
    edata: EHRData,
    var_names: Collection[str] | None,
    obs_cols: Collection[str] | None,
    layer: str | None,
    copy: bool,
    bounds: Callable[[Array | CSBase], Bounds],
    name: str,
) -> EHRData | None:
    """Clip the selected variables and obs columns to the lower and upper bounds that `bounds` computes from their values."""
    if copy:
        edata = edata.copy()

    if var_names is None and obs_cols is None:
        _ensure_feature_types(edata, layer, name)
        var_names = _numeric_var_names(edata, None)
    obs_cols, var_names = _validate_outlier_input(edata, obs_cols, var_names)  # type: ignore

    if var_names:
        X = edata.X if layer is None else edata.layers[layer]
        indices = edata.var_names.get_indexer(list(var_names))
        X = _set_columns(X, indices, _clip(X[:, indices], bounds, name))
        if layer is None:
            edata.X = X
        else:
            edata.layers[layer] = X

    if obs_cols:
        obs_cols = list(obs_cols)
        values = edata.obs[obs_cols].to_numpy(dtype=float)
        edata.obs[obs_cols] = np.clip(values, *bounds(values))

    return edata if copy else None


@singledispatch
def _clip(X: Array, bounds: Callable[[Array], Bounds], name: str) -> Array:
    xp = array_namespace(X)
    if not xp.isdtype(X.dtype, "numeric"):
        X = xp.astype(X, xp.float64)
    return xp.clip(X, *bounds(X))


@_clip.register(CSBase)
def _(X: CSBase, bounds: Callable[[CSBase], Bounds], name: str) -> CSBase:
    lower, upper = (np.broadcast_to(bound, X.shape[1]) for bound in bounds(X))
    columns = _sparse_columns(X)
    has_implicit_zeros = np.bincount(columns, minlength=X.shape[1]) < X.shape[0]
    if np.any(has_implicit_zeros & ((lower > 0) | (upper < 0))):
        _raise_densifying(name, "the bounds of a variable with implicit zeros exclude zero")
    X = X.copy()
    X.data[:] = np.clip(X.data, lower[columns], upper[columns])
    return X


@_clip.register(DaskArray)
def _(X: DaskArray, bounds: Callable[[Array], Bounds], name: str) -> DaskArray:
    if _has_sparse_chunks(X):
        return _map_variable_blocks(X, _clip, bounds, name, meta=X._meta)
    return _clip.dispatch(object)(X, bounds, name)


def _winsorize_ranks(n: Array, limits: tuple[float, float], inclusive: tuple[bool, bool]) -> tuple[Array, Array]:
    """0-based ranks of the smallest and largest value kept when cutting `limits` of `n` values, as scipy's winsorize does."""
    xp = array_namespace(n)
    low_cut, high_cut = (
        xp.floor(limit * n) if include else xp.round(limit * n)
        for limit, include in zip(limits, inclusive, strict=True)
    )
    low = xp.minimum(low_cut, xp.maximum(n - 1, 0))
    return low, xp.maximum(n - 1 - high_cut, low)


@singledispatch
def _winsorize_bounds(X: Array, limits: tuple[float, float], inclusive: tuple[bool, bool]) -> Bounds:
    """Per-variable winsorizing bounds ignoring NaNs, broadcastable against `X`."""
    xp = array_namespace(X)
    samples = X if X.ndim == 2 else xp.reshape(xp.permute_dims(X, (0, 2, 1)), (-1, X.shape[1]))
    ordered = xp.sort(samples, axis=0)
    n = xp.sum(xp.astype(~xp.isnan(samples), xp.float64), axis=0)
    positions = xp.reshape(xp.arange(samples.shape[0], dtype=xp.float64), (-1, 1))
    lower, upper = (
        _broadcast_var_stat(xp.sum(xp.where(positions == rank, ordered, 0), axis=0), X)
        for rank in _winsorize_ranks(n, limits, inclusive)
    )
    return lower, upper


@_winsorize_bounds.register(CSBase)
def _(X: CSBase, limits: tuple[float, float], inclusive: tuple[bool, bool]) -> Bounds:
    X = X.tocsc()
    bounds = np.full((2, X.shape[1]), np.nan)
    for j in range(X.shape[1]):
        stored = X.data[X.indptr[j] : X.indptr[j + 1]]
        n_implicit_zeros = X.shape[0] - len(stored)
        values = np.sort(stored[~np.isnan(stored)])
        n = len(values) + n_implicit_zeros
        if n == 0:
            continue
        ranks = np.stack(_winsorize_ranks(np.asarray(float(n)), limits, inclusive)).astype(int)
        bounds[:, j] = _order_statistic(values, int(np.searchsorted(values, 0)), n_implicit_zeros, ranks)
    return bounds[0], bounds[1]


def _validate_outlier_input(edata, obs_cols: Collection[str], vars: Collection[str]) -> tuple[set[str], set[str]]:
    """Validates the obs/var columns for outlier preprocessing."""
    vars = set(vars) if vars is not None else set()
    obs_cols = set(obs_cols) if obs_cols is not None else set()

    if vars is not None:
        diff = vars - set(edata.var_names)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in var_names.")
    if obs_cols is not None:
        diff = obs_cols - set(edata.obs.columns.values)
        if len(diff) != 0:
            raise ValueError(f"Columns {','.join(var for var in diff)} are not in obs.")

    return obs_cols, vars

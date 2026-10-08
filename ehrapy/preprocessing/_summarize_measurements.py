from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import array_api_extra as xpx
import numpy as np
import pandas as pd
from array_api_compat import array_namespace
from ehrdata import EHRData
from fast_array_utils.types import CSBase, DaskArray

from ehrapy._compat import nanquantile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    type Array = np.ndarray | DaskArray


def summarize_measurements(
    edata: EHRData,
    *,
    layer: str | None = None,
    var_names: Iterable[str] | None = None,
    statistics: Iterable[Literal["min", "max", "mean", "median", "first", "last"]] = ("min", "max", "mean"),
) -> EHRData:
    """Summarizes numerical measurements into statistics such as their minimum, maximum and average values.

    For 3D data, every variable is aggregated over the time axis of each observation, ignoring missing values.
    This is how longitudinal data reaches the functions that only support 2D data.
    The statistics `"first"` and `"last"` are the first and last non-missing value.
    For 2D data, rows that share an observation name are aggregated.

    Args:
        edata: Data object containing measurements.
        layer: Layer to calculate the expanded measurements for.
        var_names: For which measurements to determine the expanded measurements for. Defaults to None (all numerical measurements).
        statistics: Which expanded measurements to calculate.

    Returns:
        A new data object with the statistic `stat` of the variable `var` in the column `f"{var}_{stat}"` of `.X`.
        For 3D data, it keeps the observations and `.obs` of `edata`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_observations=100, base_timepoints=10)
        >>> edata.X.shape
        (100, 10, 10)
        >>> edata_summary = ep.pp.summarize_measurements(edata, statistics=["mean", "max", "last"])
        >>> edata_summary.shape
        (100, 30, 1)
        >>> ep.pp.pca(edata_summary)
    """
    X = edata.X if layer is None else edata.layers[layer]
    var_names = edata.var_names if var_names is None else list(var_names)
    if missing := set(var_names) - set(edata.var_names):
        raise KeyError(f"Variables not found: {missing}")
    statistics = list(statistics)
    values = X[:, edata.var_names.get_indexer(var_names)]
    var = pd.DataFrame(index=[f"{var}_{statistic}" for var in var_names for statistic in statistics])

    if X.ndim == 3:
        xp = array_namespace(values)
        summary = xp.stack([_aggregate_time(values, statistic) for statistic in statistics], axis=2)
        return EHRData(X=xp.reshape(summary, (summary.shape[0], -1)), obs=edata.obs.copy(), var=var)

    groups, observations = pd.factorize(edata.obs_names, sort=True)
    return EHRData(X=_summarize_groups(values, groups, statistics), obs=pd.DataFrame(index=observations), var=var)


@singledispatch
def _summarize_groups(X: np.ndarray, groups: np.ndarray, statistics: Sequence[str]) -> np.ndarray:
    """Aggregate every variable over the rows of each group, ignoring missing values, with the statistics of a variable in adjacent columns."""
    return pd.DataFrame(X).groupby(groups).agg(statistics).to_numpy(dtype=np.float64)


@_summarize_groups.register(DaskArray)
def _(X: DaskArray, groups: np.ndarray, statistics: Sequence[str]) -> DaskArray:
    # every block must hold all rows of its variables
    X = X.rechunk({0: -1})
    return X.map_blocks(
        _summarize_groups,
        groups=groups,
        statistics=statistics,
        chunks=((groups.max() + 1,), tuple(n_vars * len(statistics) for n_vars in X.chunks[1])),
        meta=X._meta.astype(np.float64),
    )


@_summarize_groups.register(CSBase)
def _(X: CSBase, groups: np.ndarray, statistics: Sequence[str]) -> CSBase:
    coo = X.tocoo()
    sizes = np.bincount(groups)
    rank = np.empty_like(groups)
    rank[np.argsort(groups, kind="stable")] = np.arange(len(groups))
    group = groups[coo.row]
    position = rank[coo.row] - (np.cumsum(sizes) - sizes)[group]
    order = np.lexsort((position, group, coo.col))
    group, column, position, values = group[order], coo.col[order], position[order], coo.data[order]
    starts = np.flatnonzero((np.diff(column, prepend=-1) != 0) | (np.diff(group, prepend=-1) != 0))
    summary = [
        _segment_statistic(values, position, starts, sizes[group[starts]], statistic) for statistic in statistics
    ]
    columns = [column[starts] * len(statistics) + i for i in range(len(statistics))]
    return type(X)(
        (np.concatenate(summary), (np.tile(group[starts], len(statistics)), np.concatenate(columns))),
        shape=(len(sizes), X.shape[1] * len(statistics)),
        dtype=np.float64,
    )


def _segment_statistic(
    values: np.ndarray, position: np.ndarray, starts: np.ndarray, sizes: np.ndarray, statistic: str
) -> np.ndarray:
    """`statistic` of every segment of `values` that starts at `starts`, sorted by `position` in a group of `sizes` rows, counting rows without a value as zeros."""
    n_stored = np.diff(starts, append=len(values))
    n_zeros = sizes - n_stored
    zero_or_nan = np.where(n_zeros > 0, 0.0, np.nan)
    valid = ~np.isnan(values)
    count = np.add.reduceat(valid, starts, dtype=np.intp) + n_zeros
    match statistic:
        case "min":
            return np.fmin(np.fmin.reduceat(values, starts), zero_or_nan)
        case "max":
            return np.fmax(np.fmax.reduceat(values, starts), zero_or_nan)
        case "mean":
            with np.errstate(invalid="ignore"):
                return np.add.reduceat(np.where(valid, values, 0), starts) / count
        case "median":
            ordered = values[np.lexsort((values, np.repeat(starts, n_stored)))]
            n_negative = np.add.reduceat(ordered < 0, starts, dtype=np.intp)

            def order_statistic(rank: np.ndarray) -> np.ndarray:
                index = starts + np.where(rank < n_negative, rank, rank - n_zeros)
                is_zero = (rank >= n_negative) & (rank < n_negative + n_zeros)
                return np.where(is_zero, 0, ordered[np.clip(index, 0, len(ordered) - 1)])

            return np.where(count > 0, (order_statistic((count - 1) // 2) + order_statistic(count // 2)) / 2, np.nan)
        case "first" | "last":
            last = statistic == "last"
            indices = np.arange(len(values))
            n_zeros_before = position - (indices - np.repeat(starts, n_stored))
            edge = (np.maximum if last else np.minimum).reduceat(
                np.where(valid, indices, -1 if last else len(values)), starts
            )
            has_value = (edge >= 0) & (edge < len(values))
            edge = np.where(has_value, edge, starts)
            no_zero_beyond = n_zeros_before[edge] == (n_zeros if last else 0)
            return np.where(has_value & no_zero_beyond, values[edge], zero_or_nan)
    raise ValueError(f"Unknown statistic: {statistic}")


def _aggregate_time(X: Array, statistic: str) -> Array:
    """Aggregate every variable of 3D data over the time axis, ignoring missing values."""
    xp = array_namespace(X)
    if not xp.isdtype(X.dtype, "real floating"):
        X = xp.astype(X, xp.float64)
    match statistic:
        case "min":
            return xpx.nanmin(X, axis=2)
        case "max":
            return xpx.nanmax(X, axis=2)
        case "mean":
            return xpx.nanmean(X, axis=2)
        case "median":
            return nanquantile(X, 0.5, axis=2)
        case "first" | "last":
            valid = ~xp.isnan(X)
            if statistic == "last":
                X, valid = xp.flip(X, axis=2), xp.flip(valid, axis=2)
            first_valid = xp.argmax(xp.astype(valid, xp.int8), axis=2, keepdims=True)
            return xp.sum(xp.where(xp.arange(X.shape[2]) == first_valid, X, 0), axis=2)
    raise ValueError(f"Unknown statistic: {statistic}")

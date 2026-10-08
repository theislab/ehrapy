from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import array_api_extra as xpx
import pandas as pd
from array_api_compat import array_namespace
from ehrdata import EHRData
from ehrdata.io import from_pandas, to_pandas

from ehrapy._compat import (
    _raise_if_not_numpy,
    nanquantile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np
    from fast_array_utils.types import DaskArray

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
    Numpy and dask arrays are supported for 3D data, and dask arrays stay lazy.
    For 2D data, rows that share an observation name are aggregated, and only numpy arrays are supported.

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
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_observations=100, base_timepoints=10, layer="tem_data")
        >>> edata.layers["tem_data"].shape
        (100, 10, 10)
        >>> edata_summary = ep.pp.summarize_measurements(edata, layer="tem_data", statistics=["mean", "max", "last"])
        >>> edata_summary.shape
        (100, 30, 1)
        >>> ep.pp.pca(edata_summary)
    """
    X = edata.X if layer is None else edata.layers[layer]
    var_names = edata.var_names if var_names is None else list(var_names)
    if missing := set(var_names) - set(edata.var_names):
        raise KeyError(f"Variables not found: {missing}")
    statistics = list(statistics)

    if X.ndim == 3:
        values = X[:, edata.var_names.get_indexer(var_names)]
        xp = array_namespace(values)
        summary = xp.stack([_aggregate_time(values, statistic) for statistic in statistics], axis=2)
        return EHRData(
            X=xp.reshape(summary, (summary.shape[0], -1)),
            obs=edata.obs.copy(),
            var=pd.DataFrame(index=[f"{var}_{statistic}" for var in var_names for statistic in statistics]),
        )

    _raise_if_not_numpy(
        X, "summarize_measurements on 2D data", "it groups the rows that share an observation name in memory"
    )
    aggregation_functions = dict.fromkeys(var_names, statistics)

    grouped = to_pandas(edata, layer=layer).groupby(edata.obs.index).agg(aggregation_functions)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]

    expanded_edata = from_pandas(grouped)

    return expanded_edata


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

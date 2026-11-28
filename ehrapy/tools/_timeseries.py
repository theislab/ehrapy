"""Helpers of the tools on longitudinal data.

The tools work on the positions of the timepoints along the last axis of a 3D layer, in their order: windows and gaps
are numbers of timepoints, and times in `obs` are positions, 0 for the first timepoint. The times in `tem` are not read.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import DaskArray
from scipy import sparse

from ehrapy._compat import _materialize
from ehrapy.preprocessing._missing_data import _previous_observed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ehrdata import EHRData


def _count(value: int, name: str, *, positive: bool = False) -> int:
    """A number of timepoints: a whole number of at least 0, or of more than 0 if `positive`."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a whole number of timepoints, not {value!r}.")
    if value < 0 or (positive and value == 0):
        bound = "more than" if positive else "at least"
        raise ValueError(f"{name} must be a number of {bound} 0 timepoints, not {value!r}.")
    return int(value)


def _obs_positions(edata: EHRData, key: str) -> np.ndarray:
    """The positions of the timepoints in `edata.obs[key]`, rounded down, NaN where missing.

    A position is a number, 0 for the first timepoint, and can lie before the first or after the last timepoint.
    """
    if key not in edata.obs:
        raise KeyError(f"edata.obs has no column {key!r}.")
    column = edata.obs[key]
    if column.isna().all():
        return np.full(edata.n_obs, np.nan)
    if pd.api.types.is_bool_dtype(column) or not pd.api.types.is_numeric_dtype(column):
        raise TypeError(
            f"edata.obs[{key!r}] must hold the positions of timepoints as numbers, 0 for the first timepoint, "
            f"not values of type {column.dtype}."
        )
    return np.floor(pd.to_numeric(column).to_numpy(np.float64, na_value=np.nan))


def _obsm_values(edata: EHRData, name: str):
    """`edata.obsm[name]`, which must hold a value per observation and timepoint, with a data frame or sparse matrix made dense."""
    values = edata.obsm[name]
    if np.shape(values) != (edata.n_obs, edata.n_t):
        raise ValueError(
            f"edata.obsm[{name!r}] must hold a value per observation and timepoint, not shape {np.shape(values)}."
        )
    if isinstance(values, pd.DataFrame):
        return values.to_numpy(dtype=np.float64, na_value=np.nan)
    if sparse.issparse(values):
        return values.toarray()
    return values


def _timeseries(edata: EHRData, names: Mapping[str, str], layer: str | None, purpose: str):
    """The inputs `names` ({input: name}) as one array (observations x inputs x timepoints), in the order of `names`.

    A name is a variable of `layer` (`.X` if `None`), or else a key of `edata.obsm` holding one value per observation and
    timepoint, such as the result of another tool. The array is a dask array if the layer is one and numpy otherwise.
    """
    selected = set(names.values())
    if missing := sorted(name for name in selected if name not in edata.var_names and name not in edata.obsm):
        raise KeyError(f"Not found in edata.var_names or edata.obsm: {missing}.")
    if both := sorted(name for name in selected if name in edata.var_names and name in edata.obsm):
        raise ValueError(f"{both} are both variables and keys of edata.obsm; rename one of them.")
    if duplicated := sorted(selected & set(edata.var_names[edata.var_names.duplicated()])):
        raise ValueError(
            f"edata.var_names holds {duplicated} more than once; make the names unique, for instance with "
            "`edata.var_names_make_unique()`."
        )
    from_layer = [name for name in names.values() if name in edata.var_names]
    X = edata.X if layer is None else edata.layers[layer]
    if from_layer and (X is None or X.ndim != 3):
        raise ValueError(f"{purpose} needs longitudinal 3D data; pass the 3D layer as `layer`.")
    obsm = {name: _obsm_values(edata, name) for name in selected if name not in edata.var_names}
    if not obsm:
        positions = [edata.var_names.get_loc(name) for name in names.values()]
        return X if positions == list(range(X.shape[1])) else X[:, positions, :]

    if from_layer and not isinstance(X, np.ndarray | DaskArray):
        raise NotImplementedError(
            f"{purpose} does not support array type {type(X).__name__} together with obsm inputs."
        )
    lazy = isinstance(X, DaskArray) and bool(from_layer)
    blocks = []
    for name in names.values():
        if name in edata.var_names:
            position = edata.var_names.get_loc(name)
            blocks.append(X[:, position : position + 1, :])
            continue
        values = obsm[name]
        if lazy:
            import dask.array as da

            values = (
                values if isinstance(values, DaskArray) else da.from_array(np.asarray(values), chunks=(X.chunks[0], -1))
            )
            blocks.append(values.astype(np.float64)[:, None, :])
        else:
            (values,) = _materialize(values)
            blocks.append(values.astype(np.float64)[:, None, :])
    if lazy:
        import dask.array as da

        return da.concatenate(blocks, axis=1)
    return np.concatenate([np.asarray(_materialize(block)[0], dtype=np.float64) for block in blocks], axis=1)


# The helpers of the scores

# The default margin of the comparisons with thresholds, far below the precision of clinical records: a rise from 1.1 to
# 1.4 mg/dL is 0.29999999999999982 in floating point and reaches 0.3.
TOLERANCE = 1e-9
CLOSED = ("right", "both")


def _tolerance(value: float) -> float:
    """A margin of the comparisons with thresholds, a number of at least 0."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"tolerance must be a number of at least 0, not {value!r}.")
    return float(value)


def _closed(value: str) -> Literal["right", "both"]:
    if value not in CLOSED:
        raise ValueError(f"closed must be 'right' or 'both', not {value!r}.")
    return value  # type: ignore[return-value]


def _rolling(
    x: np.ndarray,
    window: int,
    how: Literal["max", "min", "sum", "count"],
    *,
    closed: Literal["right", "both"] = "right",
) -> np.ndarray:
    """The statistic `how` of the non-missing values of `x` (observations x timepoints) over the trailing windows.

    The window of a timepoint holds it and the `window` - 1 timepoints before it, or with `closed="both"` the `window`
    timepoints before it.
    """
    size = window + 1 if closed == "both" else window
    return getattr(pd.DataFrame(x.T).rolling(size, min_periods=1), how)().to_numpy().T


def _locf(x: np.ndarray, within: int) -> np.ndarray:
    """The last value of `x` (observations x timepoints) at most `within` timepoints before every timepoint."""
    observed = ~np.isnan(x)
    positions = np.arange(x.shape[1])
    last = np.where(observed, positions, _previous_observed(observed))
    values = np.take_along_axis(x, np.maximum(last, 0), axis=1)
    return np.where((last >= 0) & (positions[None, :] - last <= within), values, np.nan)


def _period(seen: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Whether every timepoint lies between the first and the last one `seen`, widened to the positions `first` and `last`."""
    n_t = seen.shape[1]
    any_seen = seen.any(axis=1)
    lo = np.fmin(np.where(any_seen, seen.argmax(axis=1), n_t), first)
    hi = np.fmax(np.where(any_seen, n_t - 1 - seen[:, ::-1].argmax(axis=1), -1), last)
    positions = np.arange(n_t)[None, :]
    return (positions >= lo[:, None]) & (positions <= hi[:, None])

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import DaskArray

from ehrapy._compat import _like_obs, _map_observation_blocks, _materialize, _raise_array_type_not_implemented
from ehrapy.tools._timeseries import _count, _obs_positions, _timeseries

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ehrdata import EHRData

MARKERS = ("any", "count", "sum", "max")


def _is_marked(x: np.ndarray) -> np.ndarray:
    """Whether every value marks its timepoint: not missing and not 0."""
    return ~np.isnan(x) & (x != 0)


def _store_marked_positions(edata: EHRData, marked: np.ndarray, key_added: str) -> None:
    """Store the position of the first and the last `marked` timepoint of every observation in `obs`, NaN for none."""
    found = marked.any(axis=1)
    last = marked.shape[1] - 1 - marked[:, ::-1].argmax(axis=1) if marked.shape[1] else np.zeros(len(marked))
    first = marked.argmax(axis=1) if marked.shape[1] else np.zeros(len(marked))
    for which, position in (("first", first), ("last", last)):
        edata.obs[f"{key_added}_{which}"] = np.where(found, position, np.nan)


def _is_boolean(edata: EHRData, name: str, layer: str | None) -> bool:
    """Whether the variable or obsm key `name` holds booleans."""
    if name in edata.var_names:
        values = edata.X if layer is None else edata.layers[layer]
    else:
        values = edata.obsm[name]
        if isinstance(values, pd.DataFrame):
            return all(pd.api.types.is_bool_dtype(dtype) for dtype in values.dtypes)
    return np.dtype(getattr(values, "dtype", np.float64)) == np.bool_


@singledispatch
def _markers(X, *, how: str, zero_is_none: np.ndarray):
    """The marker (observations x timepoints) of the variables of `X` (observations x variables x timepoints).

    A value marks its timepoint if it is not missing, and not 0 for the variables of `zero_is_none`.
    """
    _raise_array_type_not_implemented(_markers, type(X))


@_markers.register(np.ndarray)
def _(X: np.ndarray, *, how: str, zero_is_none: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    seen = ~np.isnan(values) & ~(zero_is_none[None, :, None] & (values == 0))
    if how == "any":
        return seen.any(axis=1)
    if how == "count":
        return seen.sum(axis=1, dtype=np.int64)
    if how == "sum":
        result = np.where(seen, values, 0.0).sum(axis=1)
    else:
        result = np.where(seen, values, -np.inf).max(axis=1)
    return np.where(seen.any(axis=1), result, np.nan)


@_markers.register(DaskArray)
def _(X: DaskArray, *, how: str, zero_is_none: np.ndarray) -> DaskArray:
    dtype = {"any": bool, "count": np.int64}.get(how, np.float64)
    return X.rechunk({1: -1}).map_blocks(
        _markers, how=how, zero_is_none=zero_is_none, drop_axis=1, meta=np.empty((0, 0), dtype=dtype)
    )


def mark_events(
    edata: EHRData,
    var_names: str | Iterable[str] | None = None,
    *,
    how: Literal["any", "count", "sum", "max"] = "any",
    ignore_zero: bool = False,
    layer: str | None = None,
    key_added: str = "events",
    copy: bool = False,
) -> EHRData | None:
    """Mark the timepoints at which any of a set of variables has a value, such as the variables of the drugs of a class.

    A variable marks a timepoint where it has a value: a boolean variable where it is `True`, and with `ignore_zero` any variable where its value is not 0.
    A value of 0 marks its timepoint otherwise, such as a dose recorded as 0 hours long, and so does the 0 that a numeric sparse matrix in `obsm` holds where it stores nothing.
    The marker of a timepoint is whether a variable marks it (`"any"`), the number of variables that mark it (`"count"`), which is 0 where none does, or the sum (`"sum"`) or the largest (`"max"`) of their values, which are missing where none does.
    A timepoint is marked where its marker is neither missing nor 0, as the other event tools read markers.
    The position of the first and the last marked timepoint of every observation, 0 for the first timepoint, is stored in `obs`.

    Every variable is a variable of `layer`, or else a key of `obsm` with a value per observation and timepoint, such as the result of another tool.

    Args:
        edata: Central data object with longitudinal data.
        var_names: The variables that mark the timepoints.
            If `None`, all variables of `layer`.
        how: What the marker of a timepoint is: whether a variable marks it (`"any"`), the number of variables that mark it (`"count"`), or the sum (`"sum"`) or the largest (`"max"`) of their values.
        ignore_zero: Whether a value of 0 marks no timepoint, as in counts that are 0 where nothing was recorded.
        layer: The 3D layer to read the variables from.
            If `None`, `edata.X` is used.
        key_added: The marker of every observation and timepoint is stored in `edata.obsm[key_added]`, and the position of the first and the last marked timepoint in `edata.obs[f"{key_added}_first"]` and `edata.obs[f"{key_added}_last"]`, missing for observations without one.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> X = np.array([[[np.nan, 2.0, np.nan, np.nan], [np.nan, 1.0, 0.0, np.nan]]])
        >>> var = pd.DataFrame(index=["drug_a", "drug_b"])
        >>> edata = ed.EHRData(X=X, var=var)
        >>> ep.tl.mark_events(edata, ["drug_a", "drug_b"], how="count")
        >>> edata.obsm["events"]
        array([[0, 2, 1, 0]])
        >>> edata.obs[["events_first", "events_last"]]
           events_first  events_last
        0           1.0          2.0
    """
    if how not in MARKERS:
        raise ValueError(f"how must be one of {', '.join(map(repr, MARKERS))}, not {how!r}.")
    if var_names is None:
        names = list(edata.var_names)
    else:
        names = [var_names] if isinstance(var_names, str) else list(var_names)
    if not names:
        raise ValueError("Name at least one variable in var_names.")
    selected = _timeseries(edata, dict(zip(names, names, strict=True)), layer, "Marking events")
    zero_is_none = np.array([ignore_zero or _is_boolean(edata, name, layer) for name in names])

    (markers,) = _materialize(_markers(selected, how=how, zero_is_none=zero_is_none))
    edata = edata.copy() if copy else edata
    edata.obsm[key_added] = markers
    _store_marked_positions(edata, _is_marked(np.asarray(markers, dtype=np.float64)), key_added)
    return edata if copy else None


@singledispatch
def _sustained(X, end: np.ndarray, *, duration: int, max_gap: int):
    """Whether every marked timepoint of `X` (observations x events, lengths x timepoints) starts sustained events, as observations x 1 x timepoints."""
    _raise_array_type_not_implemented(_sustained, type(X))


@_sustained.register(np.ndarray)
def _(X: np.ndarray, end: np.ndarray, *, duration: int, max_gap: int) -> np.ndarray:
    # `end` (observations x 1 x 1) holds the position of the end of every observation, NaN for none
    values = np.asarray(X, dtype=np.float64)
    times = np.arange(values.shape[2], dtype=np.float64)
    end = end[:, 0, 0]
    marked = _is_marked(values[:, 0, :]) & ~(times[None, :] > end[:, None])
    obs, index = np.nonzero(marked)
    start = times[index]
    lengths = np.nan_to_num(values[obs, 1, index]) if values.shape[1] > 1 else np.zeros(len(start))
    window_end = np.fmin(start + duration, end[obs])

    # every mark steps through the later marks of its window, which follow it in time order within its observation:
    # `reach` is the latest time covered so far and `gap` the longest time from it to the next mark
    reach = start + lengths
    gap = np.full(len(start), -np.inf)
    following = np.arange(len(start))
    step = 1
    while following.size:
        following = following[following + step < len(start)]
        nxt = following + step
        in_window = (obs[nxt] == obs[following]) & (start[nxt] <= start[following] + duration)
        following, nxt = following[in_window], nxt[in_window]
        gap[following] = np.maximum(gap[following], start[nxt] - reach[following])
        reach[following] = np.maximum(reach[following], start[nxt] + lengths[nxt])
        step += 1
    gap = np.maximum(gap, window_end - reach)

    qualified = np.zeros(marked.shape, dtype=bool)
    qualified[obs[gap <= max_gap], index[gap <= max_gap]] = True
    return qualified[:, None, :]


@_sustained.register(DaskArray)
def _(X: DaskArray, end: np.ndarray, **settings: int) -> DaskArray:
    return _map_observation_blocks(
        X,
        _sustained,
        _like_obs(X, end),
        **settings,
        chunks=(X.chunks[0], (1,), (X.shape[2],)),
        meta=np.empty((0, 0, 0), dtype=bool),
    )


def mark_sustained_events(
    edata: EHRData,
    events: str,
    *,
    duration: int,
    max_gap: int,
    lengths: str | None = None,
    end_key: str | None = None,
    layer: str | None = None,
    key_added: str = "sustained_events",
    copy: bool = False,
) -> EHRData | None:
    """Find the timepoints from which events go on for a given number of timepoints without a long gap.

    A timepoint is marked where `events` is neither missing nor 0, such as the marker of :func:`~ehrapy.tools.mark_events`.
    A marked timepoint at position t covers the positions from t to t plus its length in `lengths`, which can be 0 or negative.
    A marked timepoint t qualifies if, in the window from t to t + `duration` or to the end of the observation if that is earlier, every marked timepoint and the end of the window lie at most `max_gap` timepoints after the latest position covered by the marked timepoints of the window before them.
    Marked timepoints after the end of the observation are left out.
    Durations and lengths count timepoints, which are hours on hourly data.

    Example use: On hourly data, with `duration=72` and `max_gap=24`, the starts of antibiotic administrations as `events`, the longest of their durations at every timepoint in hours rounded down as `lengths`, and the hour of death as `end_key`,
    the qualifying timepoints are the continuous antibiotic treatment as part of the sepsis label as used in an ICU benchmark "YAIB", specifically `abx_cont` of its extensions of the `ricu` R package :cite:p:`vandeWater2024,Bennett2023`.

    Args:
        edata: Central data object with longitudinal data.
        events: Variable that marks the timepoints of the events.
        duration: Timepoints after every marked timepoint that the events must go on for.
        max_gap: Most timepoints from the latest position covered by the events to the next event, or to the end of the window.
        lengths: Variable with the length of the event of every marked timepoint in timepoints, missing counting as 0.
            This and `events` are variables of `layer`, or else keys of `obsm` with a value per observation and timepoint, such as the result of another tool.
        end_key: Column of `obs` with the position of the end of every observation, 0 for the first timepoint, rounded down and missing for observations without one.
        layer: The 3D layer to read the variables from.
            If `None`, `edata.X` is used.
        key_added: Whether every timepoint qualifies is stored in `edata.obsm[key_added]`, and the position of the first and the last qualifying timepoint in `edata.obs[f"{key_added}_first"]` and `edata.obs[f"{key_added}_last"]`, missing for observations without one.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> drug = np.array([[1, 0, 1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]], dtype=float)
        >>> obs = pd.DataFrame({"end": [np.nan, 6]}, index=["a", "b"])
        >>> edata = ed.EHRData(X=drug[:, None, :], obs=obs, var=pd.DataFrame(index=["drug"]))
        >>> ep.tl.mark_sustained_events(edata, "drug", duration=4, max_gap=2, end_key="end")
        >>> edata.obsm["sustained_events"].astype(int)
        array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0]])
    """
    duration, max_gap = _count(duration, "duration", positive=True), _count(max_gap, "max_gap")
    names = {"events": events} if lengths is None else {"events": events, "lengths": lengths}
    selected = _timeseries(edata, names, layer, "Finding sustained events")
    end = np.full(edata.n_obs, np.nan) if end_key is None else _obs_positions(edata, end_key)

    (qualified,) = _materialize(_sustained(selected, end[:, None, None], duration=duration, max_gap=max_gap))
    edata = edata.copy() if copy else edata
    edata.obsm[key_added] = qualified[:, 0, :]
    _store_marked_positions(edata, qualified[:, 0, :], key_added)
    return edata if copy else None


@singledispatch
def _paired(X, *, a_to_b: int, b_to_a: int):
    """Whether every marked timepoint of `X` (observations x events a, b x timepoints) is paired, as observations x 1 x timepoints."""
    _raise_array_type_not_implemented(_paired, type(X))


@_paired.register(np.ndarray)
def _(X: np.ndarray, *, a_to_b: int, b_to_a: int) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    a, b = _is_marked(values[:, 0, :]), _is_marked(values[:, 1, :])
    paired = np.zeros(a.shape, dtype=bool)
    n_t = values.shape[2]
    for marked, other, within in ((a, b, a_to_b), (b, a, b_to_a)):
        # the number of marks of `other` before every timepoint, and the position after the last one within reach
        before = np.concatenate([np.zeros((len(other), 1), dtype=np.int64), np.cumsum(other, axis=1)], axis=1)
        stop = np.minimum(np.arange(n_t) + min(within, n_t) + 1, n_t)
        paired |= marked & (before[:, stop] > before[:, :-1])
    return paired[:, None, :]


@_paired.register(DaskArray)
def _(X: DaskArray, **settings: int) -> DaskArray:
    return _map_observation_blocks(
        X,
        _paired,
        **settings,
        chunks=(X.chunks[0], (1,), (X.shape[2],)),
        meta=np.empty((0, 0, 0), dtype=bool),
    )


def mark_paired_events(
    edata: EHRData,
    a: str,
    b: str,
    *,
    a_to_b: int,
    b_to_a: int,
    layer: str | None = None,
    key_added: str = "paired_events",
    copy: bool = False,
) -> EHRData | None:
    """Find the timepoints of two kinds of events at which one is followed by the other.

    A timepoint is marked where `a` or `b` is neither missing nor 0, such as the marker of :func:`~ehrapy.tools.mark_events`.
    A marked timepoint of `a` qualifies if `b` marks the same timepoint or one at most `a_to_b` timepoints after it, and a marked timepoint of `b` qualifies if `a` marks the same timepoint or one at most `b_to_a` timepoints after it.

    Example use: On hourly data, with antibiotics as `a`, body fluid samples as `b`, `a_to_b=24` and `b_to_a=72`, the qualifying timepoints are the suspected infections as the `ricu` R package infers them :cite:p:`Bennett2023`.

    Args:
        edata: Central data object with longitudinal data.
        a: Variable that marks the timepoints of the first kind of events.
        b: Variable that marks the timepoints of the second kind of events.
            This and `a` are variables of `layer`, or else keys of `obsm` with a value per observation and timepoint, such as the result of another tool.
        a_to_b: Most timepoints from a marked timepoint of `a` to one of `b`.
        b_to_a: Most timepoints from a marked timepoint of `b` to one of `a`.
        layer: The 3D layer to read the variables from.
            If `None`, `edata.X` is used.
        key_added: Whether every timepoint qualifies is stored in `edata.obsm[key_added]`, and the position of the first and the last qualifying timepoint in `edata.obs[f"{key_added}_first"]` and `edata.obs[f"{key_added}_last"]`, missing for observations without one.
        copy: Whether to return a copy of `edata` rather than modifying it in place.

    Returns:
        A copy of `edata` with the results if `copy` is `True`, else `None`.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> import numpy as np
        >>> import pandas as pd
        >>> X = np.array([[[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]]], dtype=float)
        >>> edata = ed.EHRData(X=X, var=pd.DataFrame(index=["drug", "sample"]))
        >>> ep.tl.mark_paired_events(edata, "drug", "sample", a_to_b=2, b_to_a=3)
        >>> edata.obsm["paired_events"].astype(int)
        array([[1, 0, 1, 0, 0, 0]])
        >>> edata.obs["paired_events_first"]
        0    0.0
        Name: paired_events_first, dtype: float64
    """
    a_to_b, b_to_a = _count(a_to_b, "a_to_b"), _count(b_to_a, "b_to_a")
    selected = _timeseries(edata, {"a": a, "b": b}, layer, "Finding paired events")

    (paired,) = _materialize(_paired(selected, a_to_b=a_to_b, b_to_a=b_to_a))
    edata = edata.copy() if copy else edata
    edata.obsm[key_added] = paired[:, 0, :]
    _store_marked_positions(edata, paired[:, 0, :], key_added)
    return edata if copy else None

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import array_api_extra as xpx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from array_api_compat import array_namespace, is_lazy_array
from ehrdata import EHRData
from ehrdata._logger import logger
from ehrdata.core.constants import CATEGORICAL_TAG, FEATURE_TYPE_KEY, NUMERIC_TAG
from fast_array_utils import stats
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase, DaskArray
from scipy.stats import chi2, rankdata, ttest_ind_from_stats

from ehrapy._compat import (
    _broadcast_var_stat,
    _by_group,
    _columnwise,
    _has_sparse_chunks,
    _like_obs,
    _map_observation_blocks,
    _map_reduction,
    _map_variable_blocks,
    _materialize,
    _obs_axes,
    _sparse_columns,
    _sparse_rows,
    _var_axes,
    nanquantile,
    nanstd,
    sparse_nan_min_max,
    sparse_nan_moments,
    sparse_nanquantile,
)
from ehrapy.get._get import _resolve_axis
from ehrapy.preprocessing._encoding import _get_encoded_features
from ehrapy.preprocessing._missing_data import _missing_mask, _previous_observed
from ehrapy.preprocessing._summarize_measurements import _aggregate_time, _tem_times

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from ehrapy.preprocessing._summarize_measurements import Statistic

    type Array = np.ndarray | DaskArray


def qc_metrics(
    edata: EHRData,
    *,
    qc_vars: Collection[str] = (),
    layer: str | None = None,
    time_key: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Calculates various quality control metrics.

    Uses the original values to calculate the metrics and not the encoded ones.
    Look at the return type for a more in depth description of the default and extended metrics.
    If :func:`~ehrdata.infer_feature_types` is run first, then extended metrics that require feature type information are calculated in addition to default metrics.
    Numeric statistics ignore non-numeric values such as unencoded categories.
    For 3D data, variable metrics are computed across observations and timepoints, and observation metrics across variables and timepoints.

    Args:
        edata: Central data object.
        qc_vars: Optional List of vars to calculate additional metrics for.
        layer: Layer to use to calculate the metrics.
        time_key: Column of `tem` with the time of every timepoint, as numbers, time differences or dates, in which the longitudinal metrics of 3D data are measured.
            If `None`, `time_value` is used if `tem` has it, so that times are in the unit of the intervals, else `interval_start_offset`.
            Time differences are measured in seconds and dates in seconds since the first timepoint.
            If `tem` has no such column, the timepoints are evenly spaced and times are their positions.
        copy: Whether to return a copy of `edata` or modify it in place.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated object.
        The calculated QC metrics are added to `obs` and `var` respectively.

        Default observation level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.
        - `entropy_of_missingness`: Entropy of the missingness pattern for each observation. Higher values indicate a more heterogeneous (less structured) missingness pattern.

        Extended observation level metrics include (only computed if :func:`~ehrdata.infer_feature_types` is run first):
        - `unique_values_abs`: Absolute amount of unique values. Returned as ``NaN`` for numeric features.
        - `unique_values_ratio`: Relative amount of unique values in percent. Returned as ``NaN`` for numeric features.

        Longitudinal observation level metrics include (only computed for 3D data):

        - `measured_timepoints_abs`: Number of timepoints with a value of any variable.
        - `measured_timepoints_pct`: Relative amount of timepoints with a value of any variable in percent.
        - `first_measured_time`: Time of the first timepoint with a value of any variable.
        - `last_measured_time`: Time of the last timepoint with a value of any variable.

        Default feature level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.
        - `entropy_of_missingness`: Entropy of the missingness pattern for each feature. Higher values indicate a more heterogeneous (less structured) missingness pattern.
        - `mean`: Mean value of the features.
        - `median`: Median value of the features.
        - `standard_deviation`: Standard deviation of the features.
        - `min`: Minimum value of the features.
        - `max`: Maximum value of the features.
        - `iqr_outliers`: Whether the feature contains outliers based on the interquartile range (IQR) method.


        Extended feature level metrics include (only computed if :func:`~ehrdata.infer_feature_types` is run first):

        - `unique_values_abs`: Absolute amount of unique values. Returned as ``NaN`` for numeric features
        - `unique_values_ratio`: Relative amount of unique values in percent. Returned as ``NaN`` for numeric features
        - `coefficient_of_variation`: Coefficient of variation of the features.
        - `is_constant`: Whether the feature is constant (with near zero variance).
        - `constant_variable_ratio`: Relative amount of constant features in percent.
        - `range_ratio`: Relative dispersion of features values respective to their mean.

        Longitudinal feature level metrics include (only computed for 3D data, ``NaN`` for encoded features):

        - `measured_obs_pct`: Relative amount of observations with at least one value of the feature in percent.
        - `median_interval`: Median time between consecutive values of the feature within an observation.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.qc_metrics(edata)
        >>> edata.obs.head()
        >>> edata.var.head()
    """
    if not isinstance(edata, EHRData):
        raise ValueError(f"Central data object should be an EHRData object, but received {type(edata).__name__}")

    if copy:
        edata = edata.copy()

    mtx = edata.X if layer is None else edata.layers[layer]
    if mtx.dtype == object and not is_lazy_array(mtx):
        _raise_error_when_heterogeneous(mtx)

    var_metrics, obs_metrics = _compute_qc_metrics(
        mtx, edata, qc_vars=qc_vars, extended=FEATURE_TYPE_KEY in edata.var, time_key=time_key
    )

    edata.var[var_metrics.columns] = var_metrics
    edata.obs[obs_metrics.columns] = obs_metrics

    return edata if copy else None


def _kept_index(X: CSBase, axis: int | tuple[int, ...]) -> tuple[np.ndarray, int]:
    """Position along the kept axis of every stored element when reducing `X` over `axis`, and that axis' length."""
    if axis in (0, (0,)):
        return _sparse_columns(X), X.shape[1]
    return _sparse_rows(X), X.shape[0]


@singledispatch
def _compute_missing_values(mtx: Array, axis: int | tuple[int, ...]) -> Array:
    """Number of missing values along `axis`."""
    xp = array_namespace(mtx)
    return xp.sum(_missing_mask(mtx), axis=axis, dtype=xp.int64)


@_compute_missing_values.register(CSBase)
def _(mtx: CSBase, axis: int | tuple[int, ...]) -> np.ndarray:
    index, n = _kept_index(mtx, axis)
    return np.bincount(index[np.isnan(mtx.data)], minlength=n)


@_compute_missing_values.register(DaskArray)
def _(mtx: DaskArray, axis: int | tuple[int, ...]) -> DaskArray:
    if _has_sparse_chunks(mtx):
        return _map_reduction(mtx, _compute_missing_values, (axis,) if isinstance(axis, int) else axis, np.int64)
    return _compute_missing_values.dispatch(object)(mtx, axis)


def _count_distinct(index: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Number of distinct non-missing `values` at every position of `index` in `range(n)`."""
    codes, uniques = pd.factorize(values)
    present = codes >= 0
    n_codes = max(len(uniques), 1)
    keys = np.unique(index[present].astype(np.int64) * n_codes + codes[present])
    return np.bincount(keys // n_codes, minlength=n)


@singledispatch
def _nunique(mtx: np.ndarray, axis: tuple[int, ...]) -> np.ndarray:
    """Number of distinct non-missing values along `axis`."""
    (kept,) = (i for i in range(mtx.ndim) if i not in axis)
    values = np.moveaxis(mtx, kept, 0).reshape(mtx.shape[kept], -1)
    index = np.repeat(np.arange(values.shape[0]), values.shape[1])
    return _count_distinct(index, values.ravel(), values.shape[0])


@_nunique.register(DaskArray)
def _(mtx: DaskArray, axis: tuple[int, ...]) -> DaskArray:
    return _map_reduction(mtx, _nunique, axis, np.int64)


@_nunique.register(CSBase)
def _(mtx: CSBase, axis: tuple[int, ...]) -> np.ndarray:
    index, n = _kept_index(mtx, axis)
    implicit_zeros = np.flatnonzero(np.bincount(index, minlength=n) < mtx.shape[axis[0]])
    values = np.concatenate([mtx.data, np.zeros(len(implicit_zeros), dtype=mtx.dtype)])
    return _count_distinct(np.concatenate([index, implicit_zeros]), values, n)


@singledispatch
def _intervals(observed: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Time since the previous observed timepoint at every observed timepoint, NaN elsewhere."""
    previous = _previous_observed(observed)
    return np.where(observed & (previous >= 0), times - times[np.maximum(previous, 0)], np.nan)


@_intervals.register(DaskArray)
def _(observed: DaskArray, times: np.ndarray) -> DaskArray:
    return _map_observation_blocks(observed, _intervals, times, dtype=np.float64)


@singledispatch
def _as_float(mtx: np.ndarray) -> np.ndarray:
    """Values as float64, with non-numeric values of object arrays, such as unencoded categories, as NaN."""
    if mtx.dtype != object:
        return mtx.astype(np.float64)
    return pd.to_numeric(mtx.ravel(), errors="coerce").astype(np.float64).reshape(mtx.shape)


@_as_float.register(DaskArray)
def _(mtx: DaskArray) -> DaskArray:
    return mtx.map_blocks(_as_float, dtype=np.float64)


def _tukey_fences(q1: Array, q3: Array) -> tuple[Array, Array]:
    """Limits beyond which values are outliers by the interquartile range method."""
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


_VAR_STATS = ("mean", "median", "standard_deviation", "min", "max", "iqr_outliers")


@singledispatch
def _var_stats(mtx: Array) -> dict[str, Array]:
    """Mean, median, standard deviation, minimum and maximum of every variable, and whether it has IQR outliers."""
    mtx = _as_float(mtx)
    xp = array_namespace(mtx)
    axes = _obs_axes(mtx)
    quartiles = nanquantile(mtx, [0.25, 0.5, 0.75], axis=axes)
    lower, upper = (_broadcast_var_stat(fence, mtx) for fence in _tukey_fences(quartiles[0], quartiles[2]))
    return {
        "mean": xpx.nanmean(mtx, axis=axes),
        "median": quartiles[1],
        "standard_deviation": nanstd(mtx, axis=axes),
        "min": xpx.nanmin(mtx, axis=axes),
        "max": xpx.nanmax(mtx, axis=axes),
        "iqr_outliers": xp.any((mtx < lower) | (mtx > upper), axis=axes),
    }


@_var_stats.register(DaskArray)
def _(mtx: DaskArray) -> dict[str, DaskArray]:
    if not _has_sparse_chunks(mtx):
        return _var_stats.dispatch(object)(mtx)
    stacked = _map_variable_blocks(
        mtx,
        lambda block: np.stack(list(_var_stats(block).values())).astype(np.float64),
        chunks=((len(_VAR_STATS),), mtx.chunks[1]),
        meta=np.array((), dtype=np.float64),
    )
    stats = dict(zip(_VAR_STATS, stacked, strict=True))
    stats["iqr_outliers"] = stats["iqr_outliers"].astype(bool)
    return stats


@_var_stats.register(CSBase)
def _(mtx: CSBase) -> dict[str, np.ndarray]:
    _, mean, var = sparse_nan_moments(mtx)
    minimum, maximum = sparse_nan_min_max(mtx)
    q1, median, q3 = sparse_nanquantile(mtx, [0.25, 0.5, 0.75])
    lower, upper = _tukey_fences(q1, q3)
    columns = _sparse_columns(mtx)
    outside = (mtx.data < lower[columns]) | (mtx.data > upper[columns])
    implicit_zeros = np.bincount(columns, minlength=mtx.shape[1]) < mtx.shape[0]
    return {
        "mean": mean,
        "median": median,
        "standard_deviation": np.sqrt(var),
        "min": minimum,
        "max": maximum,
        "iqr_outliers": (np.bincount(columns[outside], minlength=mtx.shape[1]) > 0)
        | (implicit_zeros & ((lower > 0) | (upper < 0))),
    }


@singledispatch
def _total(mtx: Array) -> Array:
    """Sum of the values of every observation across variables and timepoints."""
    return array_namespace(mtx).sum(_as_float(mtx), axis=_var_axes(mtx))


@_total.register(CSBase)
def _(mtx: CSBase) -> np.ndarray:
    return stats.sum(mtx, axis=1)


@_total.register(DaskArray)
def _(mtx: DaskArray) -> DaskArray:
    if _has_sparse_chunks(mtx):
        return stats.sum(mtx, axis=1)
    return _total.dispatch(object)(mtx)


@singledispatch
def _with_original_values(mtx: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Variables of `mtx` followed by the original values of encoded features."""
    return np.concatenate([mtx.astype(object), original], axis=1)


@_with_original_values.register(DaskArray)
def _(mtx: DaskArray, original: np.ndarray) -> DaskArray:
    import dask.array as da

    if _has_sparse_chunks(mtx):
        return _map_observation_blocks(
            mtx,
            _with_original_values,
            _like_obs(mtx, original),
            chunks=(mtx.chunks[0], (mtx.shape[1] + original.shape[1],)),
            meta=mtx._meta.astype(np.float64),
        )
    return da.concatenate([mtx.astype(object), _like_obs(mtx, original)], axis=1)


@_with_original_values.register(CSBase)
def _(mtx: CSBase, original: np.ndarray) -> CSBase:
    # sparse matrices cannot hold the original objects, so all values become codes that keep zero at zero
    codes = pd.factorize(np.concatenate([[0.0], mtx.data, original.ravel()]).astype(object))[0].astype(np.float64)
    codes[codes < 0] = np.nan
    coded = mtx.astype(np.float64)
    coded.data = codes[1 : 1 + mtx.nnz]
    return sp.hstack([coded, codes[1 + mtx.nnz :].reshape(original.shape)], format=mtx.format)


def _original_values(edata: EHRData, mtx: Array | CSBase) -> tuple[np.ndarray, np.ndarray]:
    """Index of every variable's encoded feature (-1 if not encoded), and the features' original values broadcast over timepoints."""
    features = sorted(_get_encoded_features(edata)) if "encoding_mode" in edata.var else []
    if missing := [feature for feature in features if feature not in edata.obs]:
        raise KeyError(f"Original values for {missing} not found in edata.obs.")
    if features:
        encoded = edata.var["encoding_mode"].notna()
        feature_index = pd.Index(features).get_indexer(edata.var["unencoded_var_names"].where(encoded))
    else:
        feature_index = np.full(edata.n_vars, -1)
    original = edata.obs[features].to_numpy(dtype=object)
    original = np.where(original == "nan", np.nan, original)
    if mtx.ndim == 3:
        original = np.broadcast_to(original[:, :, None], (*original.shape, mtx.shape[2]))
    return feature_index, original


def _entropy(p_missing: np.ndarray) -> np.ndarray:
    """Binary entropy of the missingness of every variable or observation."""
    p = np.clip(p_missing, 1e-10, 1 - 1e-10)  # avoid log(0)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _percentage(part: np.ndarray, total: np.ndarray | int) -> np.ndarray:
    return part / np.where(total > 0, total, np.nan) * 100


def _compute_qc_metrics(
    mtx: Array | CSBase, edata: EHRData, *, qc_vars: Collection[str], extended: bool, time_key: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate the variable and observation metrics of :func:`qc_metrics`, computing dask arrays once.

    Encoded variables are described by the original values of their feature in `edata.obs`.
    """
    obs_axes, var_axes = _obs_axes(mtx), _var_axes(mtx)
    feature_index, original = _original_values(edata, mtx)
    encoded = feature_index >= 0
    obs_mtx = mtx[:, ~encoded] if encoded.any() else mtx
    if extended:
        categorical = (edata.var[FEATURE_TYPE_KEY] == CATEGORICAL_TAG).to_numpy()
    else:
        categorical = np.zeros(edata.n_vars, dtype=bool)
    plain_categorical = categorical & ~encoded

    lazy = {
        "var_missing": _compute_missing_values(mtx, axis=obs_axes),
        "obs_missing": _compute_missing_values(obs_mtx, axis=var_axes),
        **_var_stats(mtx),
    }
    if plain_categorical.any():
        lazy["var_unique"] = _nunique(mtx[:, plain_categorical], axis=obs_axes)
    if categorical.any():
        obs_categorical = mtx[:, plain_categorical]
        if encoded.any():
            obs_categorical = _with_original_values(obs_categorical, original)
        lazy["obs_unique"] = _nunique(obs_categorical, axis=var_axes)
        lazy["obs_categorical_missing"] = _compute_missing_values(obs_categorical, axis=var_axes)
    if qc_vars:
        lazy["total_features"] = _total(mtx)
        for qc_var in qc_vars:
            lazy[f"total_features_{qc_var}"] = _total(mtx[:, edata.var[qc_var].to_numpy(dtype=bool)])
    if mtx.ndim == 3:
        observed = ~_missing_mask(mtx)
        times = _tem_times(edata, time_key)
        xp = array_namespace(observed)
        measured = xp.any(~_missing_mask(obs_mtx), axis=1)
        if encoded.any():
            measured = measured | np.any(~_missing_mask(original), axis=1)
        lazy["var_measured_obs"] = xp.sum(xp.any(observed, axis=2), axis=0)
        lazy["median_interval"] = nanquantile(_intervals(observed, times), 0.5, axis=obs_axes)
        lazy["measured_timepoints"] = xp.sum(measured, axis=1)
        lazy["first_measured_time"] = xp.min(xp.where(measured, times, xp.inf), axis=1)
        lazy["last_measured_time"] = xp.max(xp.where(measured, times, -xp.inf), axis=1)
    metrics = dict(zip(lazy, _materialize(*lazy.values()), strict=True))

    n_var_values = math.prod(mtx.shape[axis] for axis in obs_axes)
    var_missing = metrics["var_missing"]
    if encoded.any():
        var_missing[encoded] = _compute_missing_values(original, axis=obs_axes)[feature_index[encoded]]
    stats = {
        name: np.where(encoded, np.nan, metrics[name])
        for name in ("mean", "median", "standard_deviation", "min", "max")
    }

    var_metrics = pd.DataFrame(index=edata.var_names)
    var_metrics["missing_values_abs"] = var_missing
    var_metrics["missing_values_pct"] = var_missing / n_var_values * 100
    var_metrics["entropy_of_missingness"] = _entropy(var_missing / n_var_values)
    if extended:
        unique = np.full(edata.n_vars, np.nan)
        if plain_categorical.any():
            unique[plain_categorical] = metrics["var_unique"]
        if (encoded_categorical := categorical & encoded).any():
            original_unique = _nunique(original, axis=obs_axes)
            unique[encoded_categorical] = original_unique[feature_index[encoded_categorical]]
        var_metrics["unique_values_abs"] = unique
        var_metrics["unique_values_ratio"] = _percentage(unique, n_var_values - var_missing)

        numeric = (edata.var[FEATURE_TYPE_KEY] == NUMERIC_TAG).to_numpy()
        mean, std, minimum, maximum = stats["mean"], stats["standard_deviation"], stats["min"], stats["max"]
        constant = (std == 0) | (maximum == minimum)
        with np.errstate(divide="ignore", invalid="ignore"):
            coefficient_of_variation = std / mean
            range_ratio = (maximum - minimum) / mean * 100
        var_metrics["coefficient_of_variation"] = np.where(
            numeric & np.isfinite(coefficient_of_variation), coefficient_of_variation, np.nan
        )
        var_metrics["is_constant"] = np.where(numeric, constant, np.nan)
        var_metrics["constant_variable_ratio"] = constant[numeric].mean() * 100 if numeric.any() else np.nan
        var_metrics["range_ratio"] = np.where(numeric & np.isfinite(range_ratio), range_ratio, np.nan)
    for name, values in stats.items():
        var_metrics[name] = values
    var_metrics["iqr_outliers"] = metrics["iqr_outliers"] & ~encoded
    if mtx.ndim == 3:
        var_metrics["measured_obs_pct"] = np.where(encoded, np.nan, metrics["var_measured_obs"] / edata.n_obs * 100)
        var_metrics["median_interval"] = np.where(encoded, np.nan, metrics["median_interval"])

    obs_missing = metrics["obs_missing"] + _compute_missing_values(original, axis=var_axes)
    n_obs_values = math.prod(obs_mtx.shape[1:]) + math.prod(original.shape[1:])
    obs_metrics = pd.DataFrame(index=edata.obs_names)
    obs_metrics["missing_values_abs"] = obs_missing
    obs_metrics["missing_values_pct"] = obs_missing / n_obs_values * 100
    obs_metrics["entropy_of_missingness"] = _entropy(obs_missing / n_obs_values)
    if extended and categorical.any():
        n_categorical_values = math.prod(obs_categorical.shape[1:])
        obs_metrics["unique_values_abs"] = metrics["obs_unique"]
        obs_metrics["unique_values_ratio"] = _percentage(
            metrics["obs_unique"], n_categorical_values - metrics["obs_categorical_missing"]
        )
    elif extended:
        obs_metrics["unique_values_abs"] = np.nan
        obs_metrics["unique_values_ratio"] = np.nan
    for qc_var in qc_vars:
        total = metrics[f"total_features_{qc_var}"]
        obs_metrics[f"total_features_{qc_var}"] = total
        obs_metrics[f"log1p_total_features_{qc_var}"] = np.log1p(total)
        obs_metrics["total_features"] = metrics["total_features"]
        obs_metrics[f"pct_features_{qc_var}"] = total / metrics["total_features"] * 100
    if mtx.ndim == 3:
        obs_metrics["measured_timepoints_abs"] = metrics["measured_timepoints"]
        obs_metrics["measured_timepoints_pct"] = metrics["measured_timepoints"] / mtx.shape[2] * 100
        for name in ("first_measured_time", "last_measured_time"):
            obs_metrics[name] = np.where(np.isfinite(metrics[name]), metrics[name], np.nan)

    return var_metrics, obs_metrics


def _raise_error_when_heterogeneous(mtx: np.ndarray) -> None:
    mtx_df = pd.DataFrame(mtx[:, :, 0] if mtx.ndim == 3 else mtx)
    mixed = []
    for col in mtx_df.columns:
        s = mtx_df[col].dropna()
        if s.empty:
            continue
        types = {type(v) for v in s}

        if all(issubclass(t, (int, float, bool)) for t in types):
            continue
        if all(isinstance(v, str) for v in s):
            continue

        mixed.append(col)
    if mixed:
        raise ValueError(f"Mixed or unsupported types are found in columns {mixed}. Columns must be homogeneous")


def qc_lab_measurements(
    edata: EHRData,
    *,
    layer: str | None = None,
    var_names: list[str] | None = None,
    method: Literal["quantile", "iqr", "zscore", "modified_zscore"] = "iqr",
    score_type: Literal["zscore", "iqr_distance", "percentile"] = "zscore",
    add_flag: bool = True,
    add_score: bool = True,
    groupby: str | None = None,
    max_change: float | Mapping[str, float] | None = None,
    relative_change: bool = False,
    time_key: str | None = None,
    copy: bool = False,
) -> EHRData | None:
    """Flag outliers and compute anomaly scores for numeric variables.

    For each requested variable the function adds up to two columns in
    ``edata.obs``:

    * ``{var}_outlier`` – boolean flag (``True`` = outlier).
    * ``{var}_score``   – continuous anomaly score.

    For 3D data, the reference range and score statistics of a variable are computed across observations and timepoints, an observation is flagged if any of its timepoints is out of range, and its score is the mean score over its timepoints.
    3D data additionally gets the column ``{var}_jump``, which flags observations with an implausible change between two consecutive values of the variable.
    The change per time is implausible if its absolute value exceeds `max_change`, or, for variables without `max_change`, if it lies outside the range of normal changes of the variable estimated with `method`.

    Args:
        edata: Central data object.
        var_names: Variables to evaluate.  ``None`` (default) evaluates all
            variables in ``edata.var_names``.
        layer: Layer to use instead of ``edata.X``.
        method: Outlier detection method.

            * ``"iqr"`` – outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
            * ``"quantile"`` – outside [2.5th, 97.5th] percentiles.
            * ``"zscore"`` – ``|z| > 3``.
            * ``"modified_zscore"`` – ``|modified z| > 3.5`` (median / MAD).
        score_type: Continuous score assigned to each observation.

            * ``"zscore"`` – ``(x − mean) / std``.
            * ``"iqr_distance"`` – ``(x − median) / IQR``.
            * ``"percentile"`` – percentile rank in [0, 100].
        add_flag: Whether to add the ``{var}_outlier`` column, and for 3D data the ``{var}_jump`` column.
        add_score: Whether to add the ``{var}_score`` column.
        groupby: Column in ``edata.obs`` used to stratify the computation so
            that statistics are calculated within each group independently.
            Must not contain missing values.
        max_change: Largest plausible absolute change per time between two consecutive values of a variable of 3D data, for all variables or per variable.
            Variables without a value are flagged by the range of their changes estimated with `method`.
        relative_change: Whether changes are relative to the previous value instead of absolute.
        time_key: Column of `tem` with the time of every timepoint, as numbers, time differences or dates, in which changes per time are measured.
            If `None`, `time_value` is used if `tem` has it, so that times are in the unit of the intervals, else `interval_start_offset`.
            Time differences and dates are measured in seconds.
            If `tem` has no such column, changes are measured per timepoint.
        copy: If ``True``, return a modified copy; otherwise modify in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.qc_lab_measurements(edata, var_names=["potassium_first"])
    """
    if copy:
        edata = edata.copy()

    if var_names is None:
        var_names = list(edata.var_names)

    missing = [v for v in var_names if v not in edata.var_names]
    if missing:
        raise ValueError(f"Variables not found in edata.var_names: {missing}")
    if isinstance(max_change, Mapping) and (unknown := [v for v in max_change if v not in var_names]):
        raise ValueError(f"max_change has variables that are not evaluated: {unknown}")

    if groupby is not None:
        if groupby not in edata.obs.columns:
            raise ValueError(f"groupby columns not found in edata.obs: {groupby!r}")
        if edata.obs[groupby].isna().any():
            raise ValueError(f"groupby key '{groupby}' contains missing values.")

    mtx = edata.X if layer is None else edata.layers[layer]
    if mtx.ndim != 3 and max_change is not None:
        raise ValueError("max_change needs 3D data with a time axis.")
    mtx = to_dense(mtx[:, edata.var_names.get_indexer(var_names)])
    xp = array_namespace(mtx)
    mtx = xp.astype(mtx, xp.float64)
    groups = None if groupby is None else pd.factorize(edata.obs[groupby])[0]

    results = {}
    if add_flag:
        results["outlier"] = _outlier_flags(mtx, groups, method)
        if mtx.ndim == 3:
            if not isinstance(max_change, Mapping):
                max_change = dict.fromkeys(var_names, np.nan if max_change is None else max_change)
            limits = np.array([max_change.get(var, np.nan) for var in var_names], dtype=np.float64)
            changes = _changes(mtx, _tem_times(edata, time_key), relative_change)
            results["jump"] = _jump_flags(changes, groups, method, limits)
    if add_score:
        results["score"] = _anomaly_scores(mtx, groups, score_type)
    results = dict(zip(results, _materialize(*results.values()), strict=True))

    for i, var in enumerate(var_names):
        for suffix, values in results.items():
            edata.obs[f"{var}_{suffix}"] = values[:, i]

    return edata if copy else None


def _reference_range(X: Array, method: str) -> tuple[Array, Array]:
    """Lower and upper limit of the normal values of every variable."""
    xp = array_namespace(X)
    axes = _obs_axes(X)
    match method:
        case "iqr":
            quartiles = nanquantile(X, [0.25, 0.75], axis=axes)
            return _tukey_fences(quartiles[0], quartiles[1])
        case "quantile":
            limits = nanquantile(X, [0.025, 0.975], axis=axes)
            return limits[0], limits[1]
        case "zscore":
            mean, std = xpx.nanmean(X, axis=axes), nanstd(X, axis=axes)
            return mean - 3 * std, mean + 3 * std
        case "modified_zscore":
            median = nanquantile(X, 0.5, axis=axes)
            mad = nanquantile(xp.abs(X - _broadcast_var_stat(median, X)), 0.5, axis=axes)
            spread = xp.where(mad > 0, 3.5 / 0.6745 * mad, xp.nan)
            return median - spread, median + spread
    raise ValueError(f"Unknown method {method!r}.")


def _score_location_scale(X: Array, score_type: str) -> tuple[Array, Array]:
    """Center and scale of the anomaly score of every variable, with a NaN scale where it is not positive."""
    xp = array_namespace(X)
    axes = _obs_axes(X)
    match score_type:
        case "zscore":
            center, scale = xpx.nanmean(X, axis=axes), nanstd(X, axis=axes)
        case "iqr_distance":
            quartiles = nanquantile(X, [0.25, 0.5, 0.75], axis=axes)
            center, scale = quartiles[1], quartiles[2] - quartiles[0]
        case _:
            raise ValueError(f"Unknown score_type {score_type!r}.")
    return center, xp.where(scale > 0, scale, xp.nan)


def _percentile_ranks(X: np.ndarray) -> np.ndarray:
    """Percentile rank of every value within its variable, NaN for variables with fewer than two values."""
    n_valid = np.sum(~np.isnan(X), axis=0)
    ranks = rankdata(X, axis=0, nan_policy="omit")
    return np.where(n_valid >= 2, ranks / np.maximum(n_valid, 1) * 100, np.nan)


def _outlier_flags(X: Array, groups: np.ndarray | None, method: str) -> Array:
    """Whether every value lies outside its variable's reference range, for 3D data at any timepoint."""
    lower, upper = _by_group(X, groups, partial(_reference_range, method=method))
    flags = (X < lower) | (X > upper)
    return array_namespace(X).any(flags, axis=2) if X.ndim == 3 else flags


@singledispatch
def _changes(X: np.ndarray, times: np.ndarray, relative: bool) -> np.ndarray:
    """Change per time from the previous observed value at every observed timepoint, NaN elsewhere."""
    observed = ~np.isnan(X)
    previous = np.take_along_axis(X, np.maximum(_previous_observed(observed), 0), axis=2)
    change = X - previous
    if relative:
        change = change / np.where(previous != 0, np.abs(previous), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return change / _intervals(observed, times)


@_changes.register(DaskArray)
def _(X: DaskArray, times: np.ndarray, relative: bool) -> DaskArray:
    return _map_observation_blocks(X, _changes, times, relative, dtype=np.float64)


def _jump_flags(changes: Array, groups: np.ndarray | None, method: str, limits: np.ndarray) -> Array:
    """Whether any change of every variable exceeds its limit, or lies outside its range of normal changes where the limit is NaN."""
    xp = array_namespace(changes)
    given = _broadcast_var_stat(~np.isnan(limits), changes)
    limit = _broadcast_var_stat(limits, changes)
    exceeds = xp.abs(changes) > limit
    if not given.all():
        lower, upper = _by_group(changes, groups, partial(_reference_range, method=method))
        exceeds = xp.where(given, exceeds, (changes < lower) | (changes > upper))
    return xp.any(exceeds, axis=2)


def _anomaly_scores(X: Array, groups: np.ndarray | None, score_type: str) -> Array:
    """Anomaly score of every value, for 3D data averaged over timepoints."""
    if score_type == "percentile":
        scores = _columnwise(X, groups, _percentile_ranks)
    else:
        center, scale = _by_group(X, groups, partial(_score_location_scale, score_type=score_type))
        scores = (X - center) / scale
    return xpx.nanmean(scores, axis=2) if X.ndim == 3 else scores


def mcar_test(
    edata: EHRData,
    *,
    method: Literal["little", "ttest"] = "little",
    tem_names: Any | Sequence[Any] | slice | None = None,
    agg: Statistic | None = None,
    layer: str | None = None,
) -> float | pd.DataFrame:
    """Statistical hypothesis test for Missing Completely At Random (MCAR).

    Performs Little's MCAR test or pairwise t-tests.

    The null hypothesis of Little's test is that data is Missing Completely At Random (MCAR).
    A small p-value suggests the data is not MCAR.

    We advise to use Little’s MCAR test carefully.
    Rejecting the null hypothesis may not always mean that data is not MCAR, nor is accepting the null hypothesis a guarantee that data is MCAR.
    See Schouten, R. M., & Vink, G. (2021). The Dance of the Mechanisms: How Observed Information Influences the Validity of Missingness Assumptions.
    Sociological Methods & Research, 50(3), 1243-1258. https://doi.org/10.1177/0049124118799376 for a thorough discussion of missingness mechanisms.

    For 3D data, the test uses the values at the timepoint selected by `tem_names`, or `agg` reduces the selected timepoints.

    Args:
        edata: Central data object.
        method: ``"little"`` for a global chi-square test or ``"ttest"`` for pairwise Welch t-tests across all variable combinations.
        tem_names: Labels of `edata.tem.index` or a positional slice that select the timepoints of 3D data.
            If `None` (default), all timepoints are used.
        agg: How the selected timepoints of 3D data are reduced, one of the statistics of :func:`~ehrapy.preprocessing.summarize_measurements`.
            Required if more than one timepoint is selected.
        layer: Layer to apply the test to. Uses ``X`` if ``None``.

    Returns:
        A single p-value if the Little's test was applied or a Pandas DataFrame of the p-value of t-tests for each pair of features.

    Raises:
        ValueError: If Little's test is applied to variables that are not observed together in at least two observations.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(
        ...     n_observations=100, n_variables=5, missing_values=0.1, random_state=0, n_centers=1, base_timepoints=1
        ... )
        >>> ep.pp.mcar_test(edata)
        0.1412...

        Test the mean of the first 6 timepoints of longitudinal data:

        >>> edata = ed.dt.ehrdata_blobs(n_observations=100, n_variables=5, missing_values=0.1, base_timepoints=10)
        >>> p_value = ep.pp.mcar_test(edata, tem_names=slice(0, 6), agg="mean")
    """
    mtx = edata.X if layer is None else edata.layers[layer]
    if mtx.ndim == 3:
        tem_pos, _ = _resolve_axis(pd.Index(edata.tem.index), tem_names, "tem_names")
        mtx = mtx[:, :, tem_pos]
        if agg is None and mtx.shape[2] > 1:
            raise ValueError("mcar_test of 3D data needs `agg` or `tem_names` that select one timepoint.")
        mtx = mtx[:, :, 0] if agg is None else _aggregate_time(mtx, agg)
    elif tem_names is not None or agg is not None:
        raise ValueError("tem_names and agg need 3D data with a time axis.")

    # float64 required: covariance estimation and linear solves need stable floating-point math
    if mtx.dtype != np.float64:
        logger.warning(
            f"Data dtype is {mtx.dtype}, converting to float64 for MCAR test. This may temporarily increase memory usage."
        )
        mtx = mtx.astype(np.float64)

    var_names = np.asarray(edata.var_names)
    if method == "little":
        return _little_mcar_test(_missingness_patterns(mtx), var_names)
    if method == "ttest":
        return _mcar_t_tests(_missingness_patterns(mtx), var_names)
    raise ValueError(f"Unknown method {method!r}. Choose from 'little' or 'ttest'.")


class _MissingnessPatterns(NamedTuple):
    """Sufficient statistics of the observed values for every missingness pattern.

    `patterns` is True where values are missing, `counts`, `sums` and `squares` hold the number of observations and the per-variable sums and sums of squares of each pattern, and `cross_products` holds the cross-products of all variables with missing values as 0.
    """

    patterns: np.ndarray
    counts: np.ndarray
    sums: np.ndarray
    squares: np.ndarray
    cross_products: np.ndarray


def _group_by_pattern(
    patterns: np.ndarray,
    inverse: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray | CSBase,
    squares: np.ndarray | CSBase,
    cross_products: np.ndarray | CSBase,
) -> _MissingnessPatterns:
    """Sum the statistics of all rows that share a pattern, where `inverse` maps every row to its row of `patterns`."""
    indicator = sp.csr_array(
        (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))), shape=(len(patterns), len(inverse))
    )
    return _MissingnessPatterns(
        patterns,
        indicator @ counts,
        to_dense(indicator @ sums),
        to_dense(indicator @ squares),
        to_dense(cross_products),
    )


@singledispatch
def _missingness_patterns(X: np.ndarray) -> _MissingnessPatterns:
    """Sufficient statistics of every missingness pattern of `X`."""
    missing = np.isnan(X)
    observed = np.where(missing, 0.0, X)
    patterns, inverse = np.unique(missing, axis=0, return_inverse=True)
    return _group_by_pattern(patterns, inverse, np.ones(X.shape[0]), observed, observed**2, observed.T @ observed)


@_missingness_patterns.register(CSBase)
def _(X: CSBase) -> _MissingnessPatterns:
    missing = np.isnan(X.data)
    mask = sp.csr_array((missing[missing], (_sparse_rows(X)[missing], _sparse_columns(X)[missing])), shape=X.shape)
    keys = np.array(
        [mask.indices[start:stop].tobytes() for start, stop in itertools.pairwise(mask.indptr)], dtype=object
    )
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    observed = X.copy()
    observed.data[missing] = 0
    return _group_by_pattern(
        mask[first].toarray(),
        inverse,
        np.ones(X.shape[0]),
        observed,
        observed.multiply(observed),
        observed.T @ observed,
    )


@_missingness_patterns.register(DaskArray)
def _(X: DaskArray) -> _MissingnessPatterns:
    import dask

    blocks = dask.compute(*map(dask.delayed(_missingness_patterns), X.rechunk({1: -1}).to_delayed().ravel()))
    patterns, counts, sums, squares, cross_products = zip(*blocks, strict=True)
    unique, inverse = np.unique(np.concatenate(patterns), axis=0, return_inverse=True)
    return _group_by_pattern(
        unique, inverse, np.concatenate(counts), np.concatenate(sums), np.concatenate(squares), sum(cross_products)
    )


def _little_mcar_test(statistics: _MissingnessPatterns, var_names: np.ndarray) -> float:
    # Implements equation (4) from:
    # Li, C. (2013). Little's test of missing completely at random. Stata Journal, 13(4), 795-809.
    # Freely accessible preprint: https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/4/6502/files/2018/06/mcartest-zlxtj7.pdf
    # Original reference: Little, R.J.A. (1988). JASA 83(404), 1198-1202. https://doi.org/10.2307/2290157
    patterns, counts, sums, _, S = statistics
    p = patterns.shape[1]

    if not patterns.any():
        return 1.0

    valid_f = (~patterns).astype(np.float64)
    mu = sums.sum(axis=0) / (counts @ valid_f)

    denom = valid_f.T @ (counts[:, None] * valid_f)
    first, second = np.nonzero(np.triu(denom < 2))
    if len(first):
        pairs = ", ".join(
            repr(var_names[i]) if i == j else f"{var_names[i]!r} and {var_names[j]!r}"
            for i, j in zip(first[:5], second[:5], strict=True)
        )
        raise ValueError(
            f"Little's MCAR test needs every variable and pair of variables observed together in at least two observations, but {len(first)} are not, such as {pairs}. "
            "Select variables that are measured together, or for 3D data reduce more timepoints with `agg`."
        )

    M = sums.T @ valid_f
    cov_global = (S - (M * M.T) / denom) / (denom - 1)
    np.fill_diagonal(cov_global, np.maximum(cov_global.diagonal(), 1e-12))

    d2 = 0.0
    kj = 0
    for j in range(len(patterns)):
        obs_cols = ~patterns[j]
        k = obs_cols.sum()
        if k == 0:
            continue
        kj += k

        delta = sums[j, obs_cols] / counts[j] - mu[obs_cols]
        sigma_j = cov_global[np.ix_(obs_cols, obs_cols)]

        try:
            contribution = delta @ np.linalg.solve(sigma_j, delta)
        except np.linalg.LinAlgError:
            contribution = delta @ np.linalg.pinv(sigma_j) @ delta

        d2 += counts[j] * contribution

    df = kj - p
    if df <= 0:
        return 1.0

    return float(chi2.sf(d2, df))


def _observed_moments(
    statistics: _MissingnessPatterns, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Number of observed values, mean and sample standard deviation of every variable over the selected patterns."""
    n = statistics.counts[selected] @ ~statistics.patterns[selected]
    sums, squares = statistics.sums[selected].sum(axis=0), statistics.squares[selected].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n > 0, sums / n, np.nan)
        # ddof=1 to match ttest_ind equal_var=False (Welch's t-test)
        std = np.where(n > 1, np.sqrt(np.maximum(squares - n * mean**2, 0) / (n - 1)), np.nan)
    return n, mean, std


def _mcar_t_tests(statistics: _MissingnessPatterns, var_names: np.ndarray) -> pd.DataFrame:
    m = statistics.patterns.shape[1]
    result = np.full((m, m), np.nan)

    for i in range(m):
        miss = statistics.patterns[:, i]
        if miss.all() or (~miss).all():
            continue

        n1, mu1, std1 = _observed_moments(statistics, miss)
        n2, mu2, std2 = _observed_moments(statistics, ~miss)

        computable = (n1 >= 1) & (n2 >= 1)
        result[i, computable] = ttest_ind_from_stats(
            mu1[computable],
            std1[computable],
            n1[computable],
            mu2[computable],
            std2[computable],
            n2[computable],
            equal_var=False,
        ).pvalue

    return pd.DataFrame(result, index=var_names, columns=var_names)

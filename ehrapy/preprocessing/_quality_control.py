from __future__ import annotations

import copy
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from ehrdata._logger import logger
from ehrdata.io import to_pandas

from ehrapy._compat import (
    DaskArray,
    _apply_over_time_axis,
    _raise_array_type_not_implemented,
    function_2D_only,
    use_ehrdata,
)
from ehrapy.preprocessing._encoding import _get_encoded_features

if TYPE_CHECKING:
    from collections.abc import Collection


import ehrdata as ed
from anndata import AnnData
from ehrdata import EHRData


@use_ehrdata(deprecated_after="1.0.0")
def qc_metrics(
    edata: EHRData | AnnData,
    qc_vars: Collection[str] = (),
    *,
    layer: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates various quality control metrics.

    Uses the original values to calculate the metrics and not the encoded ones.
    Look at the return type for a more in depth description of the default and extended metrics.
    If :func:`~ehrdata.infer_feature_types` is run first, then extended metrics that require feature type information are calculated in addition to default metrics.


    Args:
        edata: Central data object.
        qc_vars: Optional List of vars to calculate additional metrics for.
        layer: Layer to use to calculate the metrics.

    Returns:
        Two Pandas DataFrames of all calculated QC metrics for `obs` and `var` respectively.

        Default observation level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.
        - `entropy_of_missingness`: Entropy of the missingness pattern for each observation. Higher values indicate a more heterogeneous (less structured) missingness pattern.

        Extended observation level metrics include (only computed if :func:`~ehrdata.infer_feature_types` is run first):
        - `unique_values_abs`: Absolute amount of unique values. Returned as ``NaN`` for numeric features.
        - `unique_values_ratio`: Relative amount of unique values in percent. Returned as ``NaN`` for numeric features.

        Default feature level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.
        - `entropy_of_missingness`: Entropy of the missingness pattern for each feature. Higher values indicate a more heterogeneous (less structured) missingness pattern.
        - `mean`: Mean value of the features.
        - `median`: Median value of the features.
        - `std`: Standard deviation of the features.
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


    Examples:
            >>> import ehrapy as ep
            >>> edata = ed.dt.mimic_2()
            >>> obs_qc, var_qc = ep.pp.qc_metrics(edata)
            >>> obs_qc.head()
            >>> var_qc.head()
    """
    if not isinstance(edata, EHRData) and not isinstance(edata, AnnData):
        raise ValueError(
            f"Central data object should be an EHRData or an AnnData object, but received {type(edata).__name__}"
        )

    feature_type = edata.var.get("feature_type", None)
    extended = True
    if feature_type is None:
        extended = False

    mtx = edata.X if layer is None else edata.layers[layer]

    _raise_error_when_heterogeneous(mtx)

    var_metrics = _compute_var_metrics(mtx, edata, extended=extended)
    obs_metrics = _compute_obs_metrics(mtx, edata, qc_vars=qc_vars, log1p=True, extended=extended)

    edata.var[var_metrics.columns] = var_metrics
    edata.obs[obs_metrics.columns] = obs_metrics

    return obs_metrics, var_metrics


@singledispatch
def _compute_missing_values(mtx, axis):
    _raise_array_type_not_implemented(_compute_missing_values, type(mtx))


@_compute_missing_values.register(np.ndarray)
def _(mtx: np.ndarray, axis) -> np.ndarray:
    return pd.isnull(mtx).sum(axis)


@_compute_missing_values.register(DaskArray)
def _(mtx: DaskArray, axis) -> np.ndarray:
    import dask.array as da

    return da.isnull(mtx).sum(axis).compute()


@singledispatch
def _compute_unique_values(mtx, axis):
    _raise_array_type_not_implemented(_compute_unique_values, type(mtx))


@_compute_unique_values.register(np.ndarray)
def _(mtx: np.ndarray, axis) -> np.ndarray:
    return pd.DataFrame(mtx).nunique(axis=axis, dropna=True).to_numpy()


@_compute_unique_values.register(DaskArray)
def _(mtx: DaskArray, axis) -> np.ndarray:
    import dask.array as da

    def nunique_block(block, axis):
        return pd.DataFrame(block).nunique(axis=axis, dropna=True).to_numpy()

    return da.map_blocks(nunique_block, mtx, axis=axis, dtype=int).compute()


@singledispatch
def _compute_entropy_of_missingness(mtx, axis):
    _raise_array_type_not_implemented(_compute_entropy_of_missingness, type(mtx))


@_compute_entropy_of_missingness.register(np.ndarray)
def _(mtx: np.ndarray, axis) -> np.ndarray:
    missing_mask = pd.isnull(mtx)
    p_miss = missing_mask.mean(axis=axis)
    p = np.clip(p_miss, 1e-10, 1 - 1e-10)  # avoid log(0)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@_compute_entropy_of_missingness.register(DaskArray)
def _(mtx: DaskArray, axis) -> np.ndarray:
    import dask.array as da

    missing_mask = da.isnull(mtx)
    p_miss = missing_mask.mean(axis=axis)
    p = da.clip(p_miss, 1e-10, 1 - 1e-10)  # avoid log(0)
    return -(p * da.log2(p) + (1 - p) * da.log2(1 - p)).compute()


@_apply_over_time_axis
def _row_unique(arr_2d: np.ndarray, axis) -> np.ndarray:
    uniques = _compute_unique_values(arr_2d, axis=axis)
    return np.broadcast_to(uniques[:, None], arr_2d.shape)


@_apply_over_time_axis
def _row_valid(arr_2d: np.ndarray, axis) -> np.ndarray:
    missing = _compute_missing_values(arr_2d, axis=axis)
    valid = arr_2d.shape[axis] - missing
    return np.broadcast_to(valid[:, None], arr_2d.shape)


@singledispatch
def _raise_error_when_heterogeneous(mtx):
    _raise_array_type_not_implemented(_raise_error_when_heterogeneous, type(mtx))


@_raise_error_when_heterogeneous.register(np.ndarray)
@_raise_error_when_heterogeneous.register(DaskArray)
def _(mtx: np.ndarray | DaskArray):
    if mtx.ndim == 3:
        mtx_check = mtx[:, :, 0]
    else:
        mtx_check = mtx
    try:
        mtx_check = mtx_check.compute()
    except AttributeError:
        # numpy arrays don't have .compute()
        pass

    mtx_df = pd.DataFrame(mtx_check)
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


def _compute_obs_metrics(
    mtx,
    edata: EHRData | AnnData,
    *,
    qc_vars: Collection[str] = (),
    log1p: bool = True,
    extended: bool = False,
):
    """Calculates quality control metrics for observations.

    See :func:`~ehrapy.preprocessing._quality_control.calculate_qc_metrics` for a list of calculated metrics.

    Args:
        mtx: Data array.
        edata: Central data object.
        qc_vars: A list of previously calculated QC metrics to calculate summary statistics for.
        log1p: Whether to apply log1p normalization for the QC metrics. Only used with parameter 'qc_vars'.
        extended: Whether to calculate further metrics that require feature type information.

    Returns:
        A Pandas DataFrame with the calculated metrics.
    """
    obs_metrics = pd.DataFrame(index=edata.obs_names)
    var_metrics = pd.DataFrame(index=edata.var_names)

    original_mtx = mtx

    if "encoding_mode" in edata.var:
        for original_values_categorical in _get_encoded_features(edata):
            mtx = mtx.astype(object)
            index = np.where(var_metrics.index.str.contains(original_values_categorical))[0]

            if original_values_categorical not in edata.obs.keys():
                raise KeyError(f"Original values for {original_values_categorical} not found in edata.obs.")
            mtx[:, index[0]] = np.squeeze(
                np.where(
                    edata.obs[original_values_categorical].astype(object) == "nan",
                    np.nan,
                    edata.obs[original_values_categorical].astype(object),
                )
            )

    if mtx.ndim == 3:
        n_obs, n_vars, n_time = mtx.shape
        flat_mtx = mtx.reshape(n_obs, n_vars * n_time)
    if mtx.ndim == 2:
        flat_mtx = mtx

    obs_metrics["missing_values_abs"] = _compute_missing_values(flat_mtx, axis=1)
    obs_metrics["missing_values_pct"] = (obs_metrics["missing_values_abs"] / flat_mtx.shape[1]) * 100
    obs_metrics["entropy_of_missingness"] = _compute_entropy_of_missingness(flat_mtx, axis=1)

    if extended and "feature_type" not in edata.var:
        raise ValueError(
            "Extended QC metrics require `edata.var['feature_type']`. Please run `ehrdata.infer_feature_types(edata)` first"
        )

    if extended:
        feature_type = edata.var["feature_type"]
        categorical_mask = feature_type == "categorical"

        if np.any(categorical_mask):
            cat_mask_np = np.asarray(categorical_mask)

            if original_mtx.ndim == 2:
                mtx_cat = mtx[:, cat_mask_np]  # (n_obs, n_cat_var)
            else:  # ndim == 3
                mtx_cat = original_mtx[:, cat_mask_np, :]  # (n_obs, n_cat_var, n_time)

            unique_arr = _row_unique(mtx_cat, axis=1)
            valid_arr = _row_valid(mtx_cat, axis=1)

            if unique_arr.ndim == 2:
                unique_val_abs = unique_arr[:, 0]
                valid_counts = valid_arr[:, 0]
            else:
                unique_per_time = unique_arr[:, 0, :]
                valid_per_time = valid_arr[:, 0, :]

                unique_val_abs = unique_per_time.sum(axis=1)
                valid_counts = valid_per_time.sum(axis=1)

            unique_val_ratio = np.where(
                valid_counts > 0,
                unique_val_abs / valid_counts * 100,
                np.nan,
            )
        else:
            n_obs = mtx.shape[0]
            unique_val_abs = np.full(n_obs, np.nan)
            unique_val_ratio = np.full(n_obs, np.nan)

        obs_metrics["unique_values_abs"] = unique_val_abs
        obs_metrics["unique_values_ratio"] = unique_val_ratio

    # Specific QC metrics
    for qc_var in qc_vars:
        if mtx.ndim == 3:
            raise ValueError("Only 2D matrices are supported for qc_vars argument")

        obs_metrics[f"total_features_{qc_var}"] = np.ravel(mtx[:, edata.var[qc_var].values].sum(axis=1))
        if log1p:
            obs_metrics[f"log1p_total_features_{qc_var}"] = np.log1p(obs_metrics[f"total_features_{qc_var}"])
        obs_metrics["total_features"] = np.ravel(mtx.sum(axis=1))
        obs_metrics[f"pct_features_{qc_var}"] = (
            obs_metrics[f"total_features_{qc_var}"] / obs_metrics["total_features"] * 100
        )

    return obs_metrics


def _compute_var_metrics(
    mtx,
    edata: EHRData | AnnData,
    extended: bool = False,
):
    """Compute variable metrics for quality control.

    Args:
        mtx: Data array.
        edata: Central data object.
        extended: Whether to calculate further metrics that require feature type information.
    """
    categorical_indices = np.ndarray([0], dtype=int)
    var_metrics = pd.DataFrame(index=edata.var_names)

    if mtx.ndim == 3:
        n_obs, n_vars, n_time = mtx.shape
        mtx = np.moveaxis(mtx, 1, 2).reshape(-1, n_vars)

    if "encoding_mode" in edata.var.keys():
        for original_values_categorical in _get_encoded_features(edata):
            mtx = copy.deepcopy(mtx.astype(object))
            index = np.where(var_metrics.index.str.startswith("ehrapycat_" + original_values_categorical))[0]

            if original_values_categorical not in edata.obs.keys():
                raise KeyError(f"Original values for {original_values_categorical} not found in edata.obs.")
            mtx[:, index] = np.tile(
                np.where(
                    edata.obs[original_values_categorical].astype(object) == "nan",
                    np.nan,
                    edata.obs[original_values_categorical].astype(object),
                ).reshape(-1, 1),
                mtx[:, index].shape[1],
            )
            categorical_indices = np.concatenate([categorical_indices, index])

    non_categorical_indices = np.ones(mtx.shape[1], dtype=bool)
    non_categorical_indices[categorical_indices] = False

    var_metrics["missing_values_abs"] = _compute_missing_values(mtx, axis=0)
    var_metrics["missing_values_pct"] = (var_metrics["missing_values_abs"] / mtx.shape[0]) * 100
    var_metrics["entropy_of_missingness"] = _compute_entropy_of_missingness(mtx, axis=0)

    if extended and "feature_type" not in edata.var:
        raise ValueError(
            "Extended QC metrics require `edata.var['feature_type']`. Please run `ehrdata.infer_feature_types(edata)` first"
        )

    if extended:
        feature_type = edata.var["feature_type"]
        categorical_mask = feature_type == "categorical"

        n_vars = mtx.shape[1]
        unique_val_abs_full = np.full(n_vars, np.nan)
        unique_val_ratio_full = np.full(n_vars, np.nan)

        if np.any(categorical_mask):
            cat_mask_np = np.asarray(categorical_mask)

            mtx_cat = mtx[:, cat_mask_np]

            unique_val_abs = _compute_unique_values(mtx_cat, axis=0)
            missing_cat = _compute_missing_values(mtx_cat, axis=0)
            valid_counts = mtx_cat.shape[0] - missing_cat

            unique_val_ratio = np.where(
                valid_counts > 0,
                unique_val_abs / valid_counts * 100,
                np.nan,
            )

            unique_val_abs_full[cat_mask_np] = unique_val_abs
            unique_val_ratio_full[cat_mask_np] = unique_val_ratio

        var_metrics["unique_values_abs"] = unique_val_abs_full
        var_metrics["unique_values_ratio"] = unique_val_ratio_full

        var_metrics["coefficient_of_variation"] = np.nan
        var_metrics["is_constant"] = np.nan
        var_metrics["constant_variable_ratio"] = np.nan
        var_metrics["range_ratio"] = np.nan

    var_metrics["mean"] = np.nan
    var_metrics["median"] = np.nan
    var_metrics["standard_deviation"] = np.nan
    var_metrics["min"] = np.nan
    var_metrics["max"] = np.nan
    var_metrics["iqr_outliers"] = np.nan

    try:
        # Calculate statistics for non-categorical variables
        var_metrics.loc[non_categorical_indices, "mean"] = np.nanmean(
            mtx[:, non_categorical_indices].astype(np.float64), axis=0
        )
        var_metrics.loc[non_categorical_indices, "median"] = np.nanmedian(
            mtx[:, non_categorical_indices].astype(np.float64), axis=0
        )
        var_metrics.loc[non_categorical_indices, "standard_deviation"] = np.nanstd(
            mtx[:, non_categorical_indices].astype(np.float64), axis=0
        )
        var_metrics.loc[non_categorical_indices, "min"] = np.nanmin(
            mtx[:, non_categorical_indices].astype(np.float64), axis=0
        )
        var_metrics.loc[non_categorical_indices, "max"] = np.nanmax(
            mtx[:, non_categorical_indices].astype(np.float64), axis=0
        )

        # Calculate IQR and define IQR outliers
        q1 = np.nanpercentile(mtx[:, non_categorical_indices], 25, axis=0)
        q3 = np.nanpercentile(mtx[:, non_categorical_indices], 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        var_metrics.loc[non_categorical_indices, "iqr_outliers"] = (
            ((mtx[:, non_categorical_indices] < lower_bound) | (mtx[:, non_categorical_indices] > upper_bound))
            .any(axis=0)
            .astype(float)
        )
        # Fill all non_categoricals with False because else we have a dtype object Series which h5py cannot save
        var_metrics["iqr_outliers"] = var_metrics["iqr_outliers"].astype(bool).fillna(False)

        if extended:
            feature_type = edata.var["feature_type"]
            numeric_mask = feature_type == "numeric"

            numeric_indices = np.asarray(numeric_mask)

            if np.any(numeric_indices):
                var_metrics.loc[non_categorical_indices, "coefficient_of_variation"] = (
                    var_metrics.loc[numeric_indices, "standard_deviation"] / var_metrics.loc[numeric_indices, "mean"]
                ).replace([np.inf, -np.inf], np.nan)

                # Constant column detection
                constant_mask = (var_metrics.loc[numeric_indices, "standard_deviation"] == 0) | (
                    var_metrics.loc[numeric_indices, "max"] == var_metrics.loc[numeric_indices, "min"]
                )

                var_metrics.loc[numeric_indices, "is_constant"] = constant_mask.astype(float)

                var_metrics["constant_variable_ratio"] = constant_mask.mean() * 100

                # Calculate range ratio
                var_metrics.loc[numeric_indices, "range_ratio"] = (
                    (var_metrics.loc[numeric_indices, "max"] - var_metrics.loc[numeric_indices, "min"])
                    / var_metrics.loc[numeric_indices, "mean"]
                ).replace([np.inf, -np.inf], np.nan) * 100

        var_metrics = var_metrics.infer_objects()
    except (TypeError, ValueError):
        # We assume that the data just hasn't been encoded yet
        pass

    return var_metrics


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def qc_lab_measurements(
    edata: EHRData | AnnData,
    *,
    layer: str | None = None,
    var_names: list[str] | None = None,
    method: Literal["quantile", "iqr", "zscore", "modified_zscore"] = "iqr",
    score_type: Literal["zscore", "iqr_distance", "percentile"] = "zscore",
    add_flag: bool = True,
    add_score: bool = True,
    groupby: str | None = None,
    copy: bool = False,
) -> EHRData | AnnData | None:
    """Flag outliers and compute anomaly scores for numeric variables.

    For each requested variable the function adds up to two columns in
    ``edata.obs``:

    * ``{var}_outlier`` – boolean flag (``True`` = outlier).
    * ``{var}_score``   – continuous anomaly score.

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
        add_flag: Whether to add the ``{var}_outlier`` column.
        add_score: Whether to add the ``{var}_score`` column.
        groupby: Column in ``edata.obs`` used to stratify the computation so
            that statistics are calculated within each group independently.
        copy: If ``True``, return a modified copy; otherwise modify in place.

    Returns:
        ``None`` if ``copy=False``, otherwise the updated data object.

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.qc_lab_measurements(edata, var_names=["potassium_first"])
    """
    if copy:
        edata = edata.copy()

    mtx = edata.X if layer is None else edata.layers[layer]

    if var_names is None:
        var_names = list(edata.var_names)

    missing = [v for v in var_names if v not in edata.var_names]
    if missing:
        raise ValueError(f"Variables not found in edata.var_names: {missing}")

    if groupby is not None:
        if groupby not in edata.obs.columns:
            raise ValueError(f"groupby columns not found in edata.obs: {groupby!r}")

    var_idx = {name: i for i, name in enumerate(edata.var_names)}

    for var in var_names:
        col = np.asarray(mtx[:, var_idx[var]], dtype=float).ravel()

        if groupby is None:
            flags, scores = _outlier_flags_and_scores(col, method, score_type)
        else:
            flags = np.zeros(len(col), dtype=bool)
            scores = np.full(len(col), np.nan)
            groups = edata.obs[groupby]
            for group_val in groups.unique():
                mask = (groups == group_val).values
                g_flags, g_scores = _outlier_flags_and_scores(col[mask], method, score_type)
                flags[mask] = g_flags
                scores[mask] = g_scores

        if add_flag:
            edata.obs[f"{var}_outlier"] = flags
        if add_score:
            edata.obs[f"{var}_score"] = scores

    return edata if copy else None


def _outlier_flags_and_scores(
    values: np.ndarray,
    method: str,
    score_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute outlier flags and scores for a single 1-D numeric array."""
    nan_mask = np.isnan(values)
    valid = values[~nan_mask]
    n = len(valid)

    flags = np.zeros(len(values), dtype=bool)
    scores = np.full(len(values), np.nan)

    if n < 2:
        return flags, scores

    # --- outlier flags ---
    if method == "iqr":
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        flags = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    elif method == "quantile":
        lo, hi = np.percentile(valid, [2.5, 97.5])
        flags = (values < lo) | (values > hi)
    elif method == "zscore":
        mean, std = valid.mean(), valid.std()
        if std > 0:
            flags = np.abs((values - mean) / std) > 3
    elif method == "modified_zscore":
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad > 0:
            flags = np.abs(0.6745 * (values - median) / mad) > 3.5

    flags[nan_mask] = False

    # --- scores ---
    if score_type == "zscore":
        mean, std = valid.mean(), valid.std()
        if std > 0:
            scores[~nan_mask] = (values[~nan_mask] - mean) / std
    elif score_type == "iqr_distance":
        q1, q3 = np.percentile(valid, [25, 75])
        iqr = q3 - q1
        median = np.median(valid)
        if iqr > 0:
            scores[~nan_mask] = (values[~nan_mask] - median) / iqr
    elif score_type == "percentile":
        from scipy.stats import rankdata

        ranked = rankdata(values[~nan_mask])
        scores[~nan_mask] = ranked / n * 100

    return flags, scores


@function_2D_only()
@use_ehrdata(deprecated_after="1.0.0")
def mcar_test(
    edata: EHRData | AnnData,
    method: Literal["little", "ttest"] = "little",
    *,
    layer: str | None = None,
) -> float | pd.DataFrame:
    """Statistical hypothesis test for Missing Completely At Random (MCAR).

    The null hypothesis of the Little's test is that data is Missing Completely At Random (MCAR).

    We advise to use Little’s MCAR test carefully.
    Rejecting the null hypothesis may not always mean that data is not MCAR, nor is accepting the null hypothesis a guarantee that data is MCAR.
    See Schouten, R. M., & Vink, G. (2021). The Dance of the Mechanisms: How Observed Information Influences the Validity of Missingness Assumptions.
    Sociological Methods & Research, 50(3), 1243-1258. https://doi.org/10.1177/0049124118799376 for a thorough discussion of missingness mechanisms.

    Args:
        edata: Central data object.
        method: Whether to perform a chi-square test on the entire dataset (“little”) or separate t-tests for every combination of variables (“ttest”).
        layer: Layer to apply the test to. Uses X matrix if set to `None`.

    Returns:
        A single p-value if the Little's test was applied or a Pandas DataFrame of the p-value of t-tests for each pair of features.
    """
    df = ed.io.to_pandas(edata, layer=layer)
    from pyampute.exploration.mcar_statistical_tests import MCARTest

    mt = MCARTest(method=method)

    return mt(df)

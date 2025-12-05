from __future__ import annotations

import copy
import warnings
from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from ehrdata._logger import logger
from scipy.stats import kurtosis, skew
from thefuzz import process

from ehrapy._compat import (
    DaskArray,
    _apply_over_time_axis,
    _raise_array_type_not_implemented,
    function_2D_only,
    use_ehrdata,
)
from ehrapy.anndata import anndata_to_df
from ehrapy.preprocessing._encoding import _get_encoded_features

if TYPE_CHECKING:
    from collections.abc import Collection


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
    reference_table: pd.DataFrame | None = None,
    measurements: list[str] | None = None,
    unit: Literal["traditional", "SI"] | None = None,
    threshold: int = 20,
    age_col: str | None = None,
    age_range: str | None = None,
    sex_col: str | None = None,
    sex: str | None = None,
    ethnicity_col: str | None = None,
    ethnicity: str | None = None,
    layer: str | None = None,
    copy: bool = False,
    verbose: bool = False,
) -> EHRData | AnnData | None:
    """Examines lab measurements for reference ranges and outliers.

    Source:
        The used reference values were obtained from https://accessmedicine.mhmedical.com/content.aspx?bookid=1069&sectionid=60775149 .
        This table is compiled from data in the following sources:

        * Tietz NW, ed. Clinical Guide to Laboratory Tests. 3rd ed. Philadelphia: WB Saunders Co; 1995;
        * Laposata M. SI Unit Conversion Guide. Boston: NEJM Books; 1992;
        * American Medical Association Manual of Style: A Guide for Authors and Editors. 9th ed. Chicago: AMA; 1998:486–503. Copyright 1998, American Medical Association;
        * Jacobs DS, DeMott WR, Oxley DK, eds. Jacobs & DeMott Laboratory Test Handbook With Key Word Index. 5th ed. Hudson, OH: Lexi-Comp Inc; 2001;
        * Henry JB, ed. Clinical Diagnosis and Management by Laboratory Methods. 20th ed. Philadelphia: WB Saunders Co; 2001;
        * Kratz A, et al. Laboratory reference values. N Engl J Med. 2006;351:1548–1563; 7) Burtis CA, ed. Tietz Textbook of Clinical Chemistry and Molecular Diagnostics. 5th ed. St. Louis: Elsevier; 2012.

        This version of the table of reference ranges was reviewed and updated by Jessica Franco-Colon, PhD, and Kay Brooks.

    Limitations:
        * Reference ranges differ between continents, countries and even laboratories (https://informatics.bmj.com/content/28/1/e100419).
          The default values used here are only one of many options.
        * Ensure that the values used as input are provided with the correct units. We recommend the usage of SI values.
        * The reference values pertain to adults. Many of the reference ranges need to be adapted for children.
        * By default if no gender is provided and no unisex values are available, we use the **male** reference ranges.
        * The used reference ranges may be biased for ethnicity. Please examine the primary sources if required.
        * We recommend a glance at https://www.nature.com/articles/s41591-021-01468-6 for the effect of such covariates.

    Additional values:
        * Interleukin-6 based on https://pubmed.ncbi.nlm.nih.gov/33155686/

    If you want to specify your own table as a Pandas DataFrame please examine the existing default table.
    Ethnicity and age columns can be added.
    https://github.com/theislab/ehrapy/blob/main/ehrapy/preprocessing/laboratory_reference_tables/laposata.tsv

    Args:
        edata: Central data object.
        reference_table: A custom DataFrame with reference values. Defaults to the laposata table if not specified.
        measurements: A list of measurements to check.
        unit: The unit of the measurements.
        threshold: Minimum required matching confidence score of the fuzzysearch.
                   0 = no matches, 100 = all must match.
        age_col: Column containing age values.
        age_range: The inclusive age-range to filter for such as 5-99.
        sex_col: Column containing sex values. Column must contain 'U', 'M' or 'F'.
        sex: Sex to filter the reference values for. Use U for unisex which uses male values when male and female conflict.
        ethnicity_col: Column containing ethnicity values.
        ethnicity: Ethnicity to filter for.
        layer: Layer containing the matrix to calculate the metrics for.
        copy: Whether to return a copy.
        verbose: Whether to have verbose stdout. Notifies user of matched columns and value ranges.

    Returns:
        `None` if `copy=False` and modifies the passed edata, else returns an updated data object.

    Examples:
        >>> import ehrapy as ep
        >>> edata = ed.dt.mimic_2()
        >>> ep.pp.qc_lab_measurements(edata, measurements=["potassium_first"], verbose=True)
    """
    if copy:
        edata = edata.copy()

    preprocessing_dir = Path(__file__).parent.resolve()
    if reference_table is None:
        reference_table = pd.read_csv(
            f"{preprocessing_dir}/laboratory_reference_tables/laposata.tsv", sep="\t", index_col="Measurement"
        )

    for measurement in measurements:
        best_column_match, score = process.extractOne(
            query=measurement, choices=reference_table.index, score_cutoff=threshold
        )
        if best_column_match is None:
            logger.warning(f"Unable to find a match for {measurement}")
            continue
        if verbose:
            logger.info(f"Detected '{best_column_match}' for '{measurement}' with score {score}.")

        reference_column = "SI Reference Interval" if unit == "SI" else "Traditional Reference Interval"

        # Fetch all non None columns from the reference statistics
        not_none_columns = [col for col in [sex_col, age_col, ethnicity_col] if col is not None]
        not_none_columns.append(reference_column)
        reference_values = reference_table.loc[[best_column_match], not_none_columns]

        additional_columns = False
        if sex_col or age_col or ethnicity_col:  # check if additional columns were provided
            additional_columns = True

        # Check if multiple reference values occur and no additional information is available:
        if reference_values.shape[0] > 1 and additional_columns is False:
            raise ValueError(
                f"Several options for {best_column_match} reference value are available. Please specify sex, age or "
                f"ethnicity columns and their values."
            )

        try:
            if age_col:
                min_age, max_age = age_range.split("-")
                reference_values = reference_values[
                    (reference_values[age_col].str.split("-").str[0].astype(int) >= int(min_age))
                    and (reference_values[age_col].str.split("-").str[1].astype(int) <= int(max_age))
                ]
            if sex_col:
                sexes = "U|M" if sex is None else sex
                reference_values = reference_values[reference_values[sex_col].str.contains(sexes)]
            if ethnicity_col:
                reference_values = reference_values[reference_values[ethnicity_col].isin([ethnicity])]

            if layer is not None:
                actual_measurements = edata[:, measurement].layers[layer]
            else:
                actual_measurements = edata[:, measurement].X
        except TypeError:
            logger.warning(f"Unable to find specified reference values for {measurement}.")

        check = reference_values[reference_column].values
        check_str: str = np.array2string(check)
        check_str = check_str.replace("[", "").replace("]", "").replace("'", "")
        if "<" in check_str:
            upperbound = float(check_str.replace("<", ""))
            if verbose:
                logger.info(f"Using upperbound {upperbound}")

            upperbound_check_results = actual_measurements < upperbound
            upperbound_check_results_array: np.ndarray = upperbound_check_results.copy()
            edata.obs[f"{measurement} normal"] = upperbound_check_results_array
        elif ">" in check_str:
            lower_bound = float(check_str.replace(">", ""))
            if verbose:
                logger.info(f"Using lowerbound {lower_bound}")

            lower_bound_check_results = actual_measurements > lower_bound
            lower_bound_check_results_array = lower_bound_check_results.copy()
            edata.obs[f"{measurement} normal"] = lower_bound_check_results_array
        else:  # "-" range case
            min_value = float(check_str.split("-")[0])
            max_value = float(check_str.split("-")[1])
            if verbose:
                logger.info(f"Using minimum of {min_value} and maximum of {max_value}")

            range_check_results = (actual_measurements >= min_value) & (actual_measurements <= max_value)
            range_check_results_array: np.ndarray = range_check_results.copy()
            edata.obs[f"{measurement} normal"] = range_check_results_array

    return edata if copy else None


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
    Sociological Methods & Research, 50(3), 1243-1258. https://doi.org/10.1177/0049124118799376
    for a thorough discussion of missingness mechanisms.

    Args:
        edata: Central data object.
        method: Whether to perform a chi-square test on the entire dataset (“little”) or separate t-tests for every combination of variables (“ttest”).
        layer: Layer to apply the test to. Uses X matrix if set to `None`.

    Returns:
        A single p-value if the Little's test was applied or a Pandas DataFrame of the p-value of t-tests for each pair of features.
    """
    df = anndata_to_df(edata, layer=layer)
    from pyampute.exploration.mcar_statistical_tests import MCARTest

    mt = MCARTest(method=method)

    return mt(df)

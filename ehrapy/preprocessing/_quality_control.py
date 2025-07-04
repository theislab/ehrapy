from __future__ import annotations

from functools import singledispatch
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from lamin_utils import logger
from thefuzz import process

from ehrapy._compat import DaskArray, _raise_array_type_not_implemented
from ehrapy.anndata import anndata_to_df
from ehrapy.preprocessing._encoding import _get_encoded_features

if TYPE_CHECKING:
    from collections.abc import Collection

    from anndata import AnnData


def qc_metrics(
    adata: AnnData, qc_vars: Collection[str] = (), layer: str = None
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Calculates various quality control metrics.

    Uses the original values to calculate the metrics and not the encoded ones.
    Look at the return type for a more in depth description of the calculated metrics.

    Args:
        adata: Annotated data matrix.
        qc_vars: Optional List of vars to calculate additional metrics for.
        layer: Layer to use to calculate the metrics.

    Returns:
        Two Pandas DataFrames of all calculated QC metrics for `obs` and `var` respectively.

        Observation level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.

        Feature level metrics include:

        - `missing_values_abs`: Absolute amount of missing values.
        - `missing_values_pct`: Relative amount of missing values in percent.
        - `mean`: Mean value of the features.
        - `median`: Median value of the features.
        - `std`: Standard deviation of the features.
        - `min`: Minimum value of the features.
        - `max`: Maximum value of the features.

    Examples:
            >>> import ehrapy as ep
            >>> adata = ep.dt.mimic_2(encoded=True)
            >>> obs_qc, var_qc = ep.pp.qc_metrics(adata)
            >>> obs_qc["missing_values_pct"].plot(kind="hist", bins=20)
    """
    mtx = adata.X if layer is None else adata.layers[layer]
    var_metrics = _compute_var_metrics(mtx, adata)
    obs_metrics = _compute_obs_metrics(mtx, adata, qc_vars=qc_vars, log1p=True)

    adata.var[var_metrics.columns] = var_metrics
    adata.obs[obs_metrics.columns] = obs_metrics

    return obs_metrics, var_metrics


@singledispatch
def _compute_missing_values(mtx, axis):
    _raise_array_type_not_implemented(_compute_missing_values, type(mtx))


@_compute_missing_values.register
def _(mtx: np.ndarray, axis) -> np.ndarray:
    return pd.isnull(mtx).sum(axis)


@_compute_missing_values.register
def _(mtx: DaskArray, axis) -> np.ndarray:
    import dask.array as da

    return da.isnull(mtx).sum(axis).compute()


def _compute_obs_metrics(
    mtx,
    adata: AnnData,
    *,
    qc_vars: Collection[str] = (),
    log1p: bool = True,
):
    """Calculates quality control metrics for observations.

    See :func:`~ehrapy.preprocessing._quality_control.calculate_qc_metrics` for a list of calculated metrics.

    Args:
        mtx: Data array.
        adata: Annotated data matrix.
        qc_vars: A list of previously calculated QC metrics to calculate summary statistics for.
        log1p: Whether to apply log1p normalization for the QC metrics. Only used with parameter 'qc_vars'.

    Returns:
        A Pandas DataFrame with the calculated metrics.
    """
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    var_metrics = pd.DataFrame(index=adata.var_names)

    if "encoding_mode" in adata.var:
        for original_values_categorical in _get_encoded_features(adata):
            mtx = mtx.astype(object)
            index = np.where(var_metrics.index.str.contains(original_values_categorical))[0]

            if original_values_categorical not in adata.obs.keys():
                raise KeyError(f"Original values for {original_values_categorical} not found in adata.obs.")
            mtx[:, index[0]] = np.squeeze(
                np.where(
                    adata.obs[original_values_categorical].astype(object) == "nan",
                    np.nan,
                    adata.obs[original_values_categorical].astype(object),
                )
            )

    obs_metrics["missing_values_abs"] = _compute_missing_values(mtx, axis=1)
    obs_metrics["missing_values_pct"] = (obs_metrics["missing_values_abs"] / mtx.shape[1]) * 100

    # Specific QC metrics
    for qc_var in qc_vars:
        obs_metrics[f"total_features_{qc_var}"] = np.ravel(mtx[:, adata.var[qc_var].values].sum(axis=1))
        if log1p:
            obs_metrics[f"log1p_total_features_{qc_var}"] = np.log1p(obs_metrics[f"total_features_{qc_var}"])
        obs_metrics["total_features"] = np.ravel(mtx.sum(axis=1))
        obs_metrics[f"pct_features_{qc_var}"] = (
            obs_metrics[f"total_features_{qc_var}"] / obs_metrics["total_features"] * 100
        )

    return obs_metrics


def _compute_var_metrics(
    mtx,
    adata: AnnData,
):
    """Compute variable metrics for quality control.

    Args:
        mtx: Data array.
        adata: Annotated data matrix.
    """
    categorical_indices = np.ndarray([0], dtype=int)
    var_metrics = pd.DataFrame(index=adata.var_names)

    if "encoding_mode" in adata.var.keys():
        encoded_features = _get_encoded_features(adata)
        if encoded_features:
            mtx = mtx.astype(object)

            all_indices = []
            replacement_values = []

            for original_values_categorical in encoded_features:
                index = np.where(var_metrics.index.str.startswith("ehrapycat_" + original_values_categorical))[0]

                if original_values_categorical not in adata.obs.keys():
                    raise KeyError(f"Original values for {original_values_categorical} not found in adata.obs.")

                values = np.where(
                    adata.obs[original_values_categorical].astype(object) == "nan",
                    np.nan,
                    adata.obs[original_values_categorical].astype(object),
                )

                all_indices.append(index)
                replacement_values.append(values)

            for indices, values in zip(all_indices, replacement_values, strict=False):
                mtx[:, indices] = values.reshape(-1, 1)

            categorical_indices = np.concatenate(all_indices)

    non_categorical_indices = np.ones(mtx.shape[1], dtype=bool)
    non_categorical_indices[categorical_indices] = False

    var_metrics["missing_values_abs"] = _compute_missing_values(mtx, axis=0)
    var_metrics["missing_values_pct"] = (var_metrics["missing_values_abs"] / mtx.shape[0]) * 100

    var_metrics["mean"] = np.nan
    var_metrics["median"] = np.nan
    var_metrics["standard_deviation"] = np.nan
    var_metrics["min"] = np.nan
    var_metrics["max"] = np.nan
    var_metrics["iqr_outliers"] = np.nan

    try:
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
        var_metrics = var_metrics.infer_objects()
    except (TypeError, ValueError):
        # We assume that the data just hasn't been encoded yet
        pass

    return var_metrics


def qc_lab_measurements(
    adata: AnnData,
    reference_table: pd.DataFrame = None,
    measurements: list[str] = None,
    unit: Literal["traditional", "SI"] = None,
    layer: str = None,
    threshold: int = 20,
    age_col: str = None,
    age_range: str = None,
    sex_col: str = None,
    sex: str = None,
    ethnicity_col: str = None,
    ethnicity: str = None,
    copy: bool = False,
    verbose: bool = False,
) -> AnnData:
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
        adata: Annotated data matrix.
        reference_table: A custom DataFrame with reference values. Defaults to the laposata table if not specified.
        measurements: A list of measurements to check.
        unit: The unit of the measurements.
        layer: Layer containing the matrix to calculate the metrics for.
        threshold: Minimum required matching confidence score of the fuzzysearch.
                   0 = no matches, 100 = all must match.
        age_col: Column containing age values.
        age_range: The inclusive age-range to filter for such as 5-99.
        sex_col: Column containing sex values. Column must contain 'U', 'M' or 'F'.
        sex: Sex to filter the reference values for. Use U for unisex which uses male values when male and female conflict.
        ethnicity_col: Column containing ethnicity values.
        ethnicity: Ethnicity to filter for.
        copy: Whether to return a copy.
        verbose: Whether to have verbose stdout. Notifies user of matched columns and value ranges.

    Returns:
        A modified AnnData object (copy if specified).

    Examples:
        >>> import ehrapy as ep
        >>> adata = ep.dt.mimic_2(encoded=True)
        >>> ep.pp.qc_lab_measurements(adata, measurements=["potassium_first"], verbose=True)
    """
    if copy:
        adata = adata.copy()

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
                actual_measurements = adata[:, measurement].layers[layer]
            else:
                actual_measurements = adata[:, measurement].X
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
            adata.obs[f"{measurement} normal"] = upperbound_check_results_array
        elif ">" in check_str:
            lower_bound = float(check_str.replace(">", ""))
            if verbose:
                logger.info(f"Using lowerbound {lower_bound}")

            lower_bound_check_results = actual_measurements > lower_bound
            lower_bound_check_results_array = lower_bound_check_results.copy()
            adata.obs[f"{measurement} normal"] = lower_bound_check_results_array
        else:  # "-" range case
            min_value = float(check_str.split("-")[0])
            max_value = float(check_str.split("-")[1])
            if verbose:
                logger.info(f"Using minimum of {min_value} and maximum of {max_value}")

            range_check_results = (actual_measurements >= min_value) & (actual_measurements <= max_value)
            range_check_results_array: np.ndarray = range_check_results.copy()
            adata.obs[f"{measurement} normal"] = range_check_results_array

    if copy:
        return adata


def mcar_test(
    adata: AnnData, method: Literal["little", "ttest"] = "little", *, layer: str = None
) -> float | pd.DataFrame:
    """Statistical hypothesis test for Missing Completely At Random (MCAR).

    The null hypothesis of the Little's test is that data is Missing Completely At Random (MCAR).

    We advise to use Little’s MCAR test carefully.
    Rejecting the null hypothesis may not always mean that data is not MCAR, nor is accepting the null hypothesis a guarantee that data is MCAR.
    See Schouten, R. M., & Vink, G. (2021). The Dance of the Mechanisms: How Observed Information Influences the Validity of Missingness Assumptions.
    Sociological Methods & Research, 50(3), 1243-1258. https://doi.org/10.1177/0049124118799376
    for a thorough discussion of missingness mechanisms.

    Args:
        adata: Annotated data matrix.
        method: Whether to perform a chi-square test on the entire dataset (“little”) or separate t-tests for every combination of variables (“ttest”).
        layer: Layer to apply the test to. Uses X matrix if set to `None`.

    Returns:
        A single p-value if the Little's test was applied or a Pandas DataFrame of the p-value of t-tests for each pair of features.
    """
    df = anndata_to_df(adata, layer=layer)
    from pyampute.exploration.mcar_statistical_tests import MCARTest

    mt = MCARTest(method=method)

    return mt(df)

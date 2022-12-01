from __future__ import annotations

from pathlib import Path
from typing import Collection, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from rich import print

from ehrapy import logging as logg
from ehrapy.core.str_matching import StrMatcher


def qc_metrics(
    adata: AnnData, qc_vars: Collection[str] = (), layer: str = None, inplace: bool = True
) -> pd.DataFrame | None:
    """Calculates various quality control metrics.

    Uses the original values to calculate the metrics and not the encoded ones.
    Look at the return type for a more in depth description of the calculated metrics.

    Args:
        adata: Annotated data matrix.
        qc_vars: Optional List of vars to calculate additional metrics for.
        layer: Layer to use to calculate the metrics.
        inplace: Whether to add the metrics to obs/var or to solely return a Pandas DataFrame.

    Returns:
        Pandas DataFrame of all calculated QC metrics.

        Observation level metrics include:

        `missing_values_abs`
            Absolute amount of missing values.
        `missing_values_pct`
            Relative amount of missing values in percent.

        Feature level metrics include:

        `missing_values_abs`
            Absolute amount of missing values.
        `missing_values_pct`
            Relative amount of missing values in percent.
        `mean`
            Mean value of the features.
        `median`
            Median value of the features.
        `std`
            Standard deviation of the features.
        `min`
            Minimum value of the features.
        `max`
            Maximum value of the features.

        Example:
            .. code-block:: python

                import ehrapy as ep
                import seaborn as sns
                import matplotlib.pyplot as plt

                adata = ep.dt.mimic_2(encode=True)
                ep.pp.qc_metrics(adata)
                sns.displot(adata.obs["missing_values_abs"])
                plt.show()
    """
    obs_metrics = _obs_qc_metrics(adata, layer, qc_vars)
    var_metrics = _var_qc_metrics(adata, layer)

    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
        adata.var[var_metrics.columns] = var_metrics
        logg.info("Added the calculated metrics to AnnData's `obs` and `var`.")

    return obs_metrics, var_metrics


def _missing_values(
    arr: np.ndarray, shape: tuple[int, int] = None, df_type: Literal["obs", "var"] = "obs"
) -> np.ndarray:
    """Calculates the absolute or relative amount of missing values.

    Args:
        arr: Numpy array containing a data row which is a subset of X (mtx).
        shape: Shape of X (mtx).
        df_type: Whether to calculate the proportions for obs or var. One of 'obs' or 'var' (default: 'obs').

    Returns:
        Absolute or relative amount of missing values.
    """
    # Absolute number of missing values
    if shape is None:
        return pd.isnull(arr).sum()
    # Relative number of missing values in percent
    else:
        n_rows, n_cols = shape
        if df_type == "obs":
            return (pd.isnull(arr).sum() / n_cols) * 100
        else:
            return (pd.isnull(arr).sum() / n_rows) * 100


def _obs_qc_metrics(
    adata: AnnData, layer: str = None, qc_vars: Collection[str] = (), log1p: bool = True
) -> pd.DataFrame:
    """Calculates quality control metrics for observations.

    See :func:`~ehrapy.preprocessing._quality_control.calculate_qc_metrics` for a list of calculated metrics.

    Args:
        adata: Annotated data matrix.
        layer: Layer containing the actual data matrix.
        qc_vars: A list of previously calculated QC metrics to calculate summary statistics for.
        log1p: Whether to apply log1p normalization for the QC metrics. Only used with parameter 'qc_vars'.

    Returns:
        A Pandas DataFrame with the calculated metrics.
    """
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    mtx = adata.X if layer is None else adata.layers[layer]

    obs_metrics["missing_values_abs"] = np.apply_along_axis(_missing_values, 1, mtx)
    obs_metrics["missing_values_pct"] = np.apply_along_axis(_missing_values, 1, mtx, shape=mtx.shape, df_type="obs")

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


def _var_qc_metrics(adata: AnnData, layer: str = None) -> pd.DataFrame:
    """Calculates quality control metrics for features.

    See :func:`~ehrapy.preprocessing._quality_control.calculate_qc_metrics` for a list of calculated metrics.

    Args:
        adata: Annotated data matrix.
        layer: Layer containing the matrix to calculate the metrics for.

    Returns:
        Pandas DataFrame with the calculated metrics.
    """
    # TODO we need to ensure that we are calculating the QC metrics for the original -> look at adata.uns
    var_metrics = pd.DataFrame(index=adata.var_names)
    mtx = adata.X if layer is None else adata.layers[layer]

    var_metrics["missing_values_abs"] = np.apply_along_axis(_missing_values, 0, mtx)
    var_metrics["missing_values_pct"] = np.apply_along_axis(_missing_values, 0, mtx, shape=mtx.shape, df_type="var")
    try:
        var_metrics["mean"] = np.nanmean(mtx, axis=0)
        var_metrics["median"] = np.nanmedian(mtx, axis=0)
        var_metrics["standard_deviation"] = np.nanstd(mtx, axis=0)
        var_metrics["min"] = np.nanmin(mtx, axis=0)
        var_metrics["max"] = np.nanmax(mtx, axis=0)
    except TypeError:
        print("[bold yellow]TypeError! Setting quality control metrics to nan. Did you encode your data?")
        var_metrics["mean"] = np.nan
        var_metrics["median"] = np.nan
        var_metrics["standard_deviation"] = np.nan
        var_metrics["min"] = np.nan
        var_metrics["max"] = np.nan

    return var_metrics


def qc_lab_measurements(
    adata: AnnData,
    reference_table: pd.DataFrame = None,
    measurements: list[str] = None,
    unit: Literal["traditional", "SI"] = None,
    layer: str = None,
    threshold: float = 0.2,
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
    https://github.com/theislab/ehrapy/ehrapy/preprocessing/laboratory_reference_tables/laposata.tsv

    Args:
        adata: Annotated data matrix.
        reference_table: A custom DataFrame with reference values. Defaults to the laposata table if not specified.
        measurements: A list of measurements to check.
        unit: The unit of the measurements. (default: SI)
        layer: Layer containing the matrix to calculate the metrics for.
        threshold: Minimum required matching confidence score of the bigrams.
                   0 = low requirements, 1 = high requirements.
        age_col: Column containing age values.
        age_range: The inclusive age-range to filter for. e.g. 5-99
        sex_col: Column containing sex values.
        sex: Sex to filter the reference values for. Use U for unisex which uses male values when male and female conflict.
             (default: U|M)
        ethnicity_col: Column containing ethnicity values.
        ethnicity: Ethnicity to filter for.
        copy: Whether to return a copy (default: False).
        verbose: Whether to have verbose stdout. Notifes user of matched columns and value ranges.

    Returns:
        A modified AnnData object (copy if specified).

    Example:
        .. code-block:: python

            import ehrapy as ep

            ep.pp.qc_lab_measurements(adata, measurements=["Interleukin-6[pg/ml]"], verbose=True)
    """
    if copy:
        adata = adata.copy()

    preprocessing_dir = Path(__file__).parent.resolve()
    if reference_table is None:
        reference_table = pd.read_csv(
            f"{preprocessing_dir}/laboratory_reference_tables/laposata.tsv", sep="\t", index_col="Measurement"
        )

    str_matcher = StrMatcher(list(reference_table.index))

    for measurement in measurements:
        score, best_column_match = str_matcher.best_match(query=measurement, threshold=threshold)
        if best_column_match is None:
            print(f"[bold yellow]Unable to find a match for {measurement}")
            continue
        if verbose:
            print(
                f"[bold blue]Detected [green]{best_column_match}[blue] for [green]{measurement}[blue] with score [green]{score}."
            )

        reference_column = "SI Reference Interval" if unit == "SI" else "Traditional Reference Interval"

        # Fetch all non None columns from the reference measures
        not_none_columns = [col for col in [sex_col, age_col, ethnicity_col] if col is not None]
        not_none_columns.append(reference_column)
        reference_values = reference_table.loc[[best_column_match], not_none_columns]

        # Fetch reference values
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
            print(f"[bold yellow]Unable to find specified reference values for {measurement}.")

        # Check whether the measurements are inside the reference ranges
        check = reference_values[reference_column].values
        check_str: str = np.array2string(check)
        check_str = check_str.replace("[", "").replace("]", "").replace("'", "")
        if "<" in check_str:
            upperbound = float(check_str.replace("<", ""))
            if verbose:
                print(f"[bold blue]Using upperbound [green]{upperbound}")

            upperbound_check_results = actual_measurements < upperbound
            upperbound_check_results_array: np.ndarray = upperbound_check_results.copy()
            adata.obs[f"{measurement} normal"] = upperbound_check_results_array
        elif ">" in check_str:
            lower_bound = float(check_str.replace(">", ""))
            if verbose:
                print(f"[bold blue]Using lowerbound [green]{lower_bound}")

            lower_bound_check_results = actual_measurements > lower_bound
            lower_bound_check_results_array = lower_bound_check_results.copy()
            adata.obs[f"{measurement} normal"] = lower_bound_check_results_array
        else:  # "-" range case
            min_value = float(check_str.split("-")[0])
            max_value = float(check_str.split("-")[1])
            if verbose:
                print(f"[bold blue]Using minimum of [green]{min_value}[blue] and maximum of [green]{max_value}")

            range_check_results = (actual_measurements >= min_value) & (actual_measurements <= max_value)
            range_check_results_array: np.ndarray = range_check_results.copy()
            adata.obs[f"{measurement} normal"] = range_check_results_array

    if copy:
        return adata

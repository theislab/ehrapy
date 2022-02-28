from __future__ import annotations

from typing import Collection, Literal

import numpy as np
import pandas as pd
from anndata import AnnData


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
                ep.pp.calculate_qc_metrics(adata)
                sns.displot(adata.obs["missing_values_abs"])
                plt.show()
    """
    obs_metrics = _obs_qc_metrics(adata, layer, qc_vars)
    var_metrics = _var_qc_metrics(adata, layer)

    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics
        adata.var[var_metrics.columns] = var_metrics

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
        var_metrics["mean"] = mtx.mean(axis=0)
        var_metrics["median"] = np.median(mtx, axis=0)
        var_metrics["standard_deviation"] = mtx.std(axis=0)
        var_metrics["min"] = mtx.min(axis=0)
        var_metrics["max"] = mtx.max(axis=0)
    except TypeError:
        var_metrics["mean"] = np.nan
        var_metrics["median"] = np.nan
        var_metrics["standard_deviation"] = np.nan
        var_metrics["min"] = np.nan
        var_metrics["max"] = np.nan

    return var_metrics

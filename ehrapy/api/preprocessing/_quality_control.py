from __future__ import annotations

from typing import Collection

import numpy as np
import pandas as pd
from anndata import AnnData


def calculate_qc_metrics(
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

        `test`
            bla
        `test_2`
            bla

        Feature level metrics include:
    """
    # TODO we need to ensure that we are calculating the QC metrics for the original
    obs_metrics = _obs_qc_metrics(adata, layer, qc_vars)

    if inplace:
        adata.obs[obs_metrics.columns] = obs_metrics

    return obs_metrics


def _obs_qc_metrics(
    adata: AnnData, layer: str = None, qc_vars: Collection[str] = (), log1p: bool = True
) -> pd.DataFrame:
    obs_metrics = pd.DataFrame(index=adata.obs_names)
    mtx = adata.X if layer is None else adata.layers[layer]

    def missing_values(arr: np.ndarray, shape: tuple[int, int] = None):
        if shape is None:
            return np.isnan(arr).sum()
        else:
            n_rows, n_cols = shape
            return np.isnan(arr).sum() / n_cols

    # Missing values absolute
    obs_metrics["missing_values_abs"] = np.apply_along_axis(missing_values, 1, mtx)

    # Missing values percentage
    obs_metrics["missing_values_pct"] = np.apply_along_axis(missing_values, 1, mtx, shape=mtx.shape)

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


def _var_qc_metrics(adata: AnnData, qc_vars: Collection[str]):
    qc_vars = adata.var_names if qc_vars is None else qc_vars
    # adata.uns["original_values_categoricals"] <-

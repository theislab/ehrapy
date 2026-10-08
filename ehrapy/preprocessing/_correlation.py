from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ehrapy._compat import _raise_if_dask_with_sparse_chunks
from ehrapy.preprocessing._summarize_measurements import _aggregate_time

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def _aggregate_variable_values(
    edata: EHRData,
    layer: str | None = None,
    *,
    var_names: Sequence[str] | None = None,
    agg: Literal["mean", "last", "first"] = "mean",
) -> tuple[np.ndarray, Sequence[str]]:
    """Aggregate variable values from a EHRData layer over time with specified aggregation method."""
    if layer is not None:
        if layer not in edata.layers:
            raise KeyError(f"Layer {layer} not found in edata.layers. Available: {edata.layers.keys()}")
        mtx = edata.layers[layer]
    else:
        mtx = edata.X

    if isinstance(mtx, CSBase):
        raise NotImplementedError(
            "variable_correlations does not support sparse arrays because pairwise deletion needs dense masks of missing values."
        )
    _raise_if_dask_with_sparse_chunks(mtx, "variable_correlations")

    # only include numeric or encoded variables
    numeric_var_names = set(edata.var_names) if np.issubdtype(mtx.dtype, np.number) else set()

    if var_names is None:
        var_names = [v for v in edata.var_names if v in numeric_var_names]
    else:  # when user provides the var_names
        available_vars = set(edata.var_names)
        missing = set(var_names) - available_vars
        if missing:
            raise KeyError(f"Variables not found: {missing}, {available_vars}")
        non_numeric = set(var_names) - numeric_var_names
        if non_numeric:
            raise ValueError(f"Non-numeric variables were requested {non_numeric}")

    values = mtx[:, edata.var_names.get_indexer(var_names)]
    if mtx.ndim == 3:
        if agg not in {"mean", "last", "first"}:
            raise ValueError(f"Unknown aggregation method: {agg}")
        values = _aggregate_time(values, agg)

    return to_dense(values, to_cpu_memory=True), var_names


def variable_correlations(
    edata: EHRData,
    *,
    layer: str | None = None,
    var_names: Sequence[str] | None = None,
    method: Literal["spearman", "pearson", "kendall"] = "pearson",
    agg: Literal["mean", "last", "first"] = "mean",
    correction_method: Literal["bonferroni", "fdr_bh", "fdr_tsbh", "holm", "none"] = "bonferroni",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrix with statistical testing and multiple testing correction.

    This function computes pairwise correlations between variables in the given EHRData object,
    automatically handling missing values through pairwise deletion.
    For 3D time-series data, values are aggregated across time before computing correlations.
    Dask arrays are computed once, after the aggregation.
    Sparse arrays are not supported because pairwise deletion needs dense masks of missing values.

    Args:
        edata: Central data object.
        layer: Layer to extract data from. If None, `.X` will be used.
        var_names: List of variable names to compute correlation of. If None, uses all numeric variables.
        method: Correlation method, "spearman", "kendall" or "pearson".
        agg: How to aggregate time dimension: "mean", "last" or "first".
        correction_method: Multiple testing correction method:
                    * `'bonferroni'` conservative Bonferroni correction.
                    * `'fdr_bh'` Benjamini-Hochberg false discovery rate (FDR) control.
                    * `'fdr_tsbh'` two-stage Benjamini-Hochberg, better calibrated when many variables are truly correlated.
                    * `'holm'` Holm-Bonferroni correction.
                    * `'none'` no multiple-testing correction.
        alpha: Significance threshold after correction.

    Returns:
        Correlation coefficient matrix, raw p-value matrix and boolean significance matrix after correction for each variable pair.

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_centers=5, n_observations=200, base_timepoints=3)
        >>> corr, pval, sig = ep.pp.variable_correlations(
        ...     edata, layer="tem_data", method="pearson", agg="mean", correction_method="fdr_bh", alpha=0.02
        ... )
    """
    arr, var_names = _aggregate_variable_values(edata, layer, var_names=var_names, agg=agg)

    if arr.shape[1] < 2:
        raise ValueError("For correlation matrix, at least 2 numeric variables are needed.")

    n_vars = len(var_names)

    if method not in {"spearman", "kendall", "pearson"}:
        raise ValueError(f"Unsupported correlation method: {method}")

    corr_mtx = np.full((n_vars, n_vars), np.nan)
    np.fill_diagonal(corr_mtx, 1.0)

    pval_mtx = np.ones((n_vars, n_vars))
    np.fill_diagonal(pval_mtx, 0.0)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x = arr[:, i]
            y = arr[:, j]

            mask = ~(np.isnan(x) | np.isnan(y))

            if mask.sum() < 3:
                # There should be at least 3 observations that have a value for variables i and j
                corr_mtx[i, j] = np.nan
                corr_mtx[j, i] = np.nan
                pval_mtx[i, j] = 1.0
                pval_mtx[j, i] = 1.0
                continue

            if method == "spearman":
                corr_val, pval = stats.spearmanr(x[mask], y[mask])
            elif method == "kendall":
                corr_val, pval = stats.kendalltau(x[mask], y[mask])
            else:
                corr_val, pval = stats.pearsonr(x[mask], y[mask])

            corr_mtx[i, j] = corr_val
            corr_mtx[j, i] = corr_val
            pval_mtx[i, j] = pval
            pval_mtx[j, i] = pval

    corr_df = pd.DataFrame(corr_mtx, index=var_names, columns=var_names)
    pval_df = pd.DataFrame(pval_mtx, index=var_names, columns=var_names)
    # Multiple testing correction
    if correction_method != "none":
        indices = np.triu_indices(n_vars, k=1)
        pvals_upper = pval_mtx[indices]

        _, pval_corrected, _, _ = multipletests(pvals_upper, alpha=alpha, method=correction_method)
        sig_mtx = np.zeros((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(sig_mtx, True)

        for idx, (i, j) in enumerate(zip(*indices, strict=False)):
            is_sig = pval_corrected[idx] < alpha
            sig_mtx[i, j] = is_sig
            sig_mtx[j, i] = is_sig
    else:
        sig_mtx = pval_mtx < alpha

    sig_df = pd.DataFrame(sig_mtx, index=var_names, columns=var_names)

    return corr_df, pval_df, sig_df

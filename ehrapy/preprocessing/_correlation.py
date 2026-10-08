from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase, DaskArray
from scipy import special, stats
from statsmodels.stats.multitest import multipletests

from ehrapy._compat import _aggregate_time, sparse_nan_min_max

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def _aggregate_variable_values(
    edata: EHRData,
    layer: str | None = None,
    *,
    var_names: Sequence[str] | None = None,
    agg: Literal["mean", "last", "first"] = "mean",
) -> tuple[np.ndarray | CSBase, Sequence[str]]:
    """Aggregate variable values from a EHRData layer over time with specified aggregation method."""
    if layer is not None:
        if layer not in edata.layers:
            raise KeyError(f"Layer {layer} not found in edata.layers. Available: {edata.layers.keys()}")
        mtx = edata.layers[layer]
    else:
        mtx = edata.X

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

    return (values.compute() if isinstance(values, DaskArray) else values), var_names


@singledispatch
def _correlations(X: np.ndarray | CSBase, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Correlation coefficient and p-value of every pair of variables over the observations where both are observed."""
    n_vars = X.shape[1]
    corr_mtx = np.full((n_vars, n_vars), np.nan)
    np.fill_diagonal(corr_mtx, 1.0)

    pval_mtx = np.ones((n_vars, n_vars))
    np.fill_diagonal(pval_mtx, 0.0)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x, y = to_dense(X[:, [i, j]], to_cpu_memory=True).T

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

    return corr_mtx, pval_mtx


@_correlations.register(CSBase)
def _(X: CSBase, method: str) -> tuple[np.ndarray, np.ndarray]:
    if method != "pearson":
        return _correlations.dispatch(np.ndarray)(X.tocsc(), method)
    is_nan = np.isnan(X.data)
    filled, missing = X.astype(np.float64), X.astype(np.float64)
    filled.data[is_nan] = 0
    missing.data = is_nan.astype(np.float64)
    n_missing = (missing.T @ missing).toarray()
    n = X.shape[0] - np.diag(n_missing)[:, None] - np.diag(n_missing) + n_missing
    sums, squares = (
        np.asarray(a.sum(axis=0)).reshape(-1, 1) - (a.T @ missing).toarray() for a in (filled, filled.power(2))
    )
    variances = n * squares - sums**2
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.clip((n * (filled.T @ filled).toarray() - sums * sums.T) / np.sqrt(variances * variances.T), -1, 1)
    # rounding can leave a nonzero variance for constant variables
    minimum, maximum = sparse_nan_min_max(X)
    corr[:, minimum == maximum] = corr[minimum == maximum] = np.nan
    pval = 2 * special.betaincc(n / 2 - 1, n / 2 - 1, (np.abs(corr) + 1) / 2)
    corr[n < 3], pval[n < 3] = np.nan, 1.0
    np.fill_diagonal(corr, 1.0)
    np.fill_diagonal(pval, 0.0)
    return corr, pval


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
        ...     edata, method="pearson", agg="mean", correction_method="fdr_bh", alpha=0.02
        ... )
    """
    arr, var_names = _aggregate_variable_values(edata, layer, var_names=var_names, agg=agg)

    if arr.shape[1] < 2:
        raise ValueError("For correlation matrix, at least 2 numeric variables are needed.")

    n_vars = len(var_names)

    if method not in {"spearman", "kendall", "pearson"}:
        raise ValueError(f"Unsupported correlation method: {method}")

    corr_mtx, pval_mtx = _correlations(arr, method)

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

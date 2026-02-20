from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import array_api_compat
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ehrdata import EHRData


def _extract_variable_values(
    edata: EHRData,
    layer: str,
    var_names: Sequence[str] | None = None,
    agg: Literal["mean", "last", "first"] = "mean",
) -> pd.DataFrame:
    """Extract variable values from a EHRData layer aggregating over time with specified aggregation method."""
    if layer not in edata.layers:
        raise KeyError(f"Layer {layer} not found in edata.layers. Available: {edata.layers.keys()}")

    mtx = edata.layers[layer]
    xp = array_api_compat.array_namespace(mtx)

    if var_names is None:
        var_names = list(edata.var_names)
    else:
        available_vars = set(edata.var_names)
        missing = set(var_names) - available_vars
        if missing:
            raise KeyError(f"Variables not found: {missing}")

    var_indices = [list(edata.var_names).index(v) for v in var_names]

    if mtx.ndim == 2:
        n_obs, n_var = mtx.shape
        mtx_2d = mtx[:, var_indices]
        mtx_2d_np = array_api_compat.numpy.asarray(mtx_2d)
        df = pd.DataFrame(mtx_2d_np, columns=var_names, index=edata.obs_names)

    else:
        n_obs, n_var, n_time = mtx.shape
        if agg == "mean":
            mtx_2d = xp.mean(mtx[:, var_indices, :], axis=2)
            mtx_2d_np = array_api_compat.numpy.asarray(mtx_2d)
            df = pd.DataFrame(mtx_2d_np, columns=var_names, index=edata.obs_names)
        elif agg == "last" or agg == "first":
            mtx_sub = mtx[:, var_indices, :]
            mtx_sub = xp.astype(mtx_sub, xp.float64)
            valid_mask = ~xp.isnan(mtx_sub)
            if agg == "last":
                mtx_sub = xp.flip(mtx_sub, axis=2)  # for argmax to find the last valid value
                valid_mask = xp.flip(valid_mask, axis=2)

            first_valid = xp.argmax(valid_mask, axis=2)
            is_valid = xp.any(valid_mask, axis=2)

            obs_idx = xp.arange(n_obs)[:, None]
            var_idx = xp.arange(len(var_indices))[None, :]
            mtx_2d = mtx_sub[obs_idx, var_idx, first_valid]

            mtx_2d = xp.where(is_valid, mtx_2d, xp.full(mtx_2d.shape, xp.nan, dtype=mtx_2d.dtype))
            mtx_2d_np = array_api_compat.numpy.asarray(mtx_2d)
            df = pd.DataFrame(mtx_2d_np, columns=var_names, index=edata.obs_names)
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def compute_variable_correlations(
    edata: EHRData,
    *,
    layer: str,
    var_names: Sequence[str] | None = None,
    method: Literal["spearman", "pearson", "kendall"] = "pearson",
    agg: Literal["mean", "last", "first"] = "mean",
    correction_method: Literal["bonferroni", "fdr_bh", "fdr_tsbh", "holm", "none"] = "bonferroni",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrix with statistical testing and multiple testing correction.

    This function computes pairwise correlations between variables in the given EHRData object,
    automatically handling missing values through pairwise deletion. For 3D
    time-series data, values are aggregated across time before computing correlations.

    Args:
        edata: Central data object.
        layer: Layer to extract data from.
        var_names: List of variable names to compute correlation of. If None, uses all numeric variables.
        method: Correlation method, "spearman", "kendall" or "pearson".
        agg: How to aggregate time dimension: "mean", "last" or "first".
        correction_method: Multiple testing correction method:
            -   "bonferroni": conservative Bonferroni correction
            -   "fdr_bh": Benjamini Hochberg FDR
            -   "fdr_tsbh": two-stage Benjamini-Hochberg, better calibrated when many variables are truly correlated
            -   "holm": Holm-Bonferroni
            -   "none": no correction
        alpha: Significance threshold after correction.

    Returns:
        corr_df : :class:`~pandas.DataFrame`
            Correlation coefficient matrix
        pval_df : :class:`~pandas.DataFrame`
            Raw p-value matrix
        sig_df : :class:`~pandas.DataFrame`
            Boolean significance matrix after correction

    Examples:
        >>> import ehrdata as ed
        >>> import ehrapy as ep
        >>> edata = ed.dt.ehrdata_blobs(n_variables=10, n_centers=5, n_observations=200, base_timepoints=3)
        >>> corr, pval, sig = ep.tl.compute_variable_correlations(
        ...     edata, layer="tem_data", method="pearson", agg="mean", correction_method="fdr_bh", alpha=0.02
        ... )
    """
    df = _extract_variable_values(edata, layer=layer, var_names=var_names, agg=agg)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("For correlation matrix, at least 2 numeric variables are needed.")

    df = df[numeric_cols]
    n_vars = len(numeric_cols)

    if method == "spearman" or method == "kendall" or method == "pearson":
        corr_df = df.corr(method=method)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    pval_mtx = np.ones((n_vars, n_vars))
    np.fill_diagonal(pval_mtx, 0.0)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x = df.iloc[:, i].values
            y = df.iloc[:, j].values

            mask = ~(np.isnan(x) | np.isnan(y))

            if mask.sum() < 3:
                # There should be at least 3 observations that have a value for variables i and j
                corr_df.iloc[i, j] = np.nan
                corr_df.iloc[j, i] = np.nan
                pval_mtx[i, j] = 1.0
                pval_mtx[j, i] = 1.0
                continue

            if method == "spearman":
                _, pval = stats.spearmanr(x[mask], y[mask])
            elif method == "kendall":
                _, pval = stats.kendalltau(x[mask], y[mask])
            else:
                _, pval = stats.pearsonr(x[mask], y[mask])

            pval_mtx[i, j] = pval
            pval_mtx[j, i] = pval

    pval_df = pd.DataFrame(pval_mtx, index=numeric_cols, columns=numeric_cols)
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

    sig_df = pd.DataFrame(sig_mtx, index=numeric_cols, columns=numeric_cols)

    return corr_df, pval_df, sig_df

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    """Extract variable values from a 3D EHRData layer."""
    if layer not in edata.layers:
        raise KeyError(f"Layer {layer} not found in edata.layers. Available: {edata.layers.keys()}")

    mtx = edata.layers[layer]

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
        df = pd.DataFrame(mtx_2d, columns=var_names, index=edata.obs_names)

    else:
        n_obs, n_var, n_time = mtx.shape
        if agg == "mean":
            mtx_2d = np.nanmean(mtx[:, var_indices, :], axis=2)
            df = pd.DataFrame(mtx_2d, columns=var_names, index=edata.obs_names)
        elif agg == "last" or agg == "first":
            mtx_2d = np.empty((n_obs, len(var_indices)))
            for i, _var_idx in enumerate(var_names):
                for obs_idx in range(n_obs):
                    values = mtx[obs_idx, var_indices[i], :]
                    non_nan_mask = ~np.isnan(values.astype(float))
                    if non_nan_mask.any():
                        if agg == "last":
                            mtx_2d[obs_idx, i] = values[non_nan_mask][-1]
                        elif agg == "first":
                            mtx_2d[obs_idx, i] = values[non_nan_mask][0]
                    else:
                        mtx_2d[obs_idx, i] = np.nan
            df = pd.DataFrame(mtx_2d, columns=var_names, index=edata.obs_names)
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def compute_variable_correlations(
    edata: EHRData,
    *,
    layer: str,
    var_names: Sequence[str] | None = None,
    method: Literal["spearman", "pearson"] = "pearson",
    agg: Literal["mean", "last", "first"] = "mean",
    correction_method: Literal["bonferroni", "fdr_bh", "holm", "none"] = "bonferroni",
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
        method: Correlation method, "spearman" or "pearson".
        agg: How to aggregate time dimension: "mean", "last" or "first".
        correction_method: Multiple testing correction method:
            -   "bonferroni": conservative Bonferroni correction
            -   "fdr_bh": Benjamini Hochberg FDR
            -   "holm": Holm-Bonferroni
            -   "none": no correction
        alpha: Significance threshold after correction.

    Returns:
        tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame)
            - corr_mtx: Correlation coefficient matrix (pd.DataFrame)
            - pval_mtx: Raw p-value matrix (pd.DataFrame)
            - sig_mtx: Boolean significance matrix after correction (pd.DataFrame)
    """
    df = _extract_variable_values(edata, layer=layer, var_names=var_names, agg=agg)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError("For correlation matrix, at least 2 numeric variables are needed.")

    df = df[numeric_cols]
    n_vars = len(numeric_cols)

    # Correlation computation using pandas
    if method == "spearman" or method == "pearson":
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
